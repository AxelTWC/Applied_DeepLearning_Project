import os
import sys
import json

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import gc
import re
import copy
from typing import List
from trainer.trainer_utils import init_distributed_mode, Logger, init_model, llm_checkpoint, setup_seed, get_lr, SkipBatchSampler, is_main_process
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from dataset.dataset import RAGDataset
from rag_pipeline.retriever import FaissLocalRetriever
from rag_pipeline.utils import break_condition
from utils.metrics import normalize_text
from rag_pipeline.prompt import ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT
import torch.nn.functional as F
from tqdm import tqdm

def calc_reward(args, prompts: List[str], responses: List[List[str]], contexts: List[List[[str]]], ground_truth: List[List[str]]) -> torch.Tensor:
    def gt_reward(contexts: List[str], gt: str) -> float:
        gt = [normalize_text(v) for v in gt]
        for ctx in contexts:
            if normalize_text(ctx) in gt:
                return 0.5
        return 0.0
    
    def ctx_penalty(contexts: List[str], gt: str, current_step: int, gt_step: int) -> float:
        gt = [normalize_text(v) for v in gt]
        if gt_step != -1 and current_step - gt_step > 1:
            for ctx in contexts:
                if normalize_text(ctx) in gt:
                    return -0.1
                return -0.25
        return 0.0
    
    rewards = torch.zeros(len(responses), device=args.device)
    batch_size = len(prompts)
    for i in range(batch_size):
        for j in range(args.num_generations):
            response_idx = i * args.num_generations + j
            response = responses[response_idx]
            context = contexts[response_idx]
            gt = ground_truth[response_idx]
            gt_step = -1
            for idx, r in enumerate(response):
                if break_condition(r):
                    if idx != len(response) - 1:
                        rewards[response_idx] -= 0.2
                    continue
                ctx = context[idx]
                rewards[response_idx] += gt_reward(ctx, gt) + ctx_penalty(ctx, gt, idx, gt_step)
                if gt_reward(ctx, gt) > 0 and gt_step == -1:
                    gt_step = idx
            rewards[response_idx] = rewards[response_idx] / len(response)
    return rewards

def collate_fn(batch):
    return {
        'messages': [x['messages'] for x in batch],
        'ground_truth': [x['ground_truth'] for x in batch],
        'question': [x['question'] for x in batch]
    }

def gspo_train_epoch(epoch, loader, model, tokenizer, ref_model, retriever, start_step, wandb, optimizer, scheduler, device, args):
    for batch in tqdm(loader, initial=start_step + 1, desc=f"Epoch {epoch}"):
        messages = batch['messages']
        question = batch['question']
        gen_inputs = [copy.deepcopy(messages) for _ in range(args.num_generations)]
        batch_contexts = []
        batch_responses = []
        batch_raw_outputs = []
        with torch.no_grad():
            # use .module for DDP model
            gen_model = model.module if isinstance(model, DistributedDataParallel) else model
            for generation_idx in range(args.num_generations):
                contexts = [[] for _ in range(args.batch_size)]
                responses = [[] for _ in range(args.batch_size)]
                raw_outputs = [[] for _ in range(args.batch_size)]
                # Mask to keep track of active generations. 1 means active, 0 means finished.
                active_mask = [True] * args.batch_size 
                
                for step in range(args.max_step):
                    # Check if all generations are finished
                    if not any(active_mask):
                        break
                        
                    gen_messages =  gen_inputs[generation_idx] # [num_generations, messages]
                    
                    # Prepare prompts only for active generations
                    active_indices = [i for i, active in enumerate(active_mask) if active]
                    active_messages = [gen_messages[i] for i in active_indices]
                    
                    if not active_messages:
                        break

                    gen_prompts = tokenizer.apply_chat_template(
                        active_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                        
                    prompt_inputs = tokenizer(gen_prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                    padding_side="left", add_special_tokens=False).to(args.device)
                                    
                    outputs = gen_model.generate(
                        **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=1.0, top_p=0.9, top_k=40,
                        num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
                    for i in range(args.batch_size):
                        # raw_outputs[i].append(outputs.cpu())
                        continue
                        
                    response_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
                    response_tokens = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    
                    # Map back results to original indices
                    current_response_idx = 0
                    for i in range(args.batch_size):
                        if not active_mask[i]:
                            continue
                            
                        response_token = response_tokens[current_response_idx]
                        current_response_idx += 1 
                        
                        # Store response
                        responses[i].append(response_token)
                        gen_inputs[generation_idx][i].append({"role": "assistant", "content": response_token})
                        if break_condition(response_token):
                            active_mask[i] = False
                            continue
                            
                        sub_topics = [topic.strip() for topic in response_token.split(",") if topic.strip()]
                        if len(sub_topics) == 0:
                            sub_topics = [question[i]] # Use question[i] for specific batch item
                            
                        question_ids = [f"{question[i]}_subq_{i}_{step}_{k}" for k in range(len(sub_topics))]
                        raw_contexts, retrieval_contexts = retriever.retrieve_batch(sub_topics, question_ids=question_ids, top_k=5)
                        
                        step_contexts = []
                        references = ""
                        for sub_topic in sub_topics:
                            ctx, scores = retrieval_contexts[f"{question[i]}_subq_{i}_{step}_{sub_topics.index(sub_topic)}"]
                            step_contexts.extend(ctx) # Flatten contexts
                            references += f"\nSub-topics: {sub_topic}\nRetrieved Contexts: {ctx}\n"
                        
                        contexts[i].append(step_contexts) # Store all contexts for this step
                        gen_inputs[generation_idx][i].append({"role": "user", "content": ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT.format(question=question[i], references=references)})
                
                batch_contexts.append(contexts)
                batch_responses.append(responses)
                batch_raw_outputs.append(raw_outputs)
        # Flatten batch_contexts and batch_responses to [batch_size * num_generations]
        # Original structure: [num_generations, batch_size, steps]
        # Target structure: [batch_size * num_generations, steps]
        
        flat_batch_responses = []
        flat_batch_contexts = []
        flat_batch_gts = []
        
        for i in range(args.batch_size):
            for j in range(args.num_generations):
                flat_batch_responses.append(batch_responses[j][i])
                flat_batch_contexts.append(batch_contexts[j][i])
                flat_batch_gts.append(batch['ground_truth'][i])
        outputs = []
        for i in range(args.batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                output = {
                    "messages": gen_inputs[j][i],
                    "question": question[i],
                    "responses": flat_batch_responses[response_idx],
                    "contexts": flat_batch_contexts[response_idx],
                    "gts": flat_batch_gts[response_idx],
                }
                outputs.append(output)
        with open("responses.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)
        raw_output = {
            "messages": gen_inputs,
            "question": question,
            "responses": flat_batch_responses,
            "contexts": flat_batch_contexts,
            "gts": flat_batch_gts,
        }
        with open("raw_output.json", "w", encoding="utf-8") as f:
            json.dump(raw_output, f, ensure_ascii=False, indent=4)
        rewards = calc_reward(args, question, flat_batch_responses, flat_batch_contexts, flat_batch_gts)
        rewards.view(-1, args.num_generations)
        print(rewards)
        break
        # batch_loss = batch_loss / args.batch_size
        # batch_loss.backward()
        # # Update Parameters
        # if (step + 1) % args.accumulation_steps == 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
            
        # if step % args.log_interval == 0 or step == iters:
        #     policy_loss_val = batch_loss.item()
        #     avg_reward_val = rewards.mean().item()
        #     avg_len_val = completion_mask.sum(dim=1).float().mean().item()
        #     current_lr = optimizer.param_groups[0]['lr']

        #     Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
        #            f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
        #            f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

        #     if wandb and is_main_process():
        #         wandb.log({
        #             "policy_loss": policy_loss_val,
        #             "reward": avg_reward_val,
        #             "avg_response_len": avg_len_val,
        #             "advantages_mean": advantages.mean().item(),
        #             "learning_rate": current_lr
        #         })

        # if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
        #     model.eval()
        #     ckp = f'{args.save_dir}/{args.save_name}.pth'
        #     state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        #     torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
        #     llm_checkpoint(args.model, save_name=args.save_name, model=model, optimizer=optimizer, 
        #                  epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
        #     model.train()

        # torch.cuda.empty_cache()
        # gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="The directory to save the results")
    parser.add_argument("--save_name", type=str, default="grpo", help="The name of the save file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="The model to use")
    parser.add_argument("--num_generations", type=int, default=4, help="The number of generations to sample")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--max_step", type=int, default=3, help="The max step for Adaptive RAG")
    parser.add_argument("--max_gen_len", type=int, default=512, help="The max generation length")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate")
    parser.add_argument("--device", type=str, default="mps", help="The device to use")
    parser.add_argument("--dtype", type=str, default="float32", help="The data type to use")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers to load the data")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="The number of steps to accumulate the gradients")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="The gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=10, help="The number of steps to warmup the learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay")
    parser.add_argument("--log_interval", type=int, default=1, help="The interval to log the results")
    parser.add_argument("--save_interval", type=int, default=10, help="The interval to save the model")
    parser.add_argument("--from_checkpoint", default=0, type=int, choices=[0, 1], help="Whether to load the model from a checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument("--wandb_project", type=str, default="AdaptiveRAG-GRPO", help="wandb project name")
    parser.add_argument("--data_path", type=str, default="../dataset/nq_1000.jsonl", help="The path to load the data")
    args = parser.parse_args()
    
    # ================================ Initialize Env ================================
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f'cuda:{local_rank}'
    setup_seed(1508 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # Initialize Retriever
    retriever = FaissLocalRetriever(embedding_model="BAAI/bge-m3", faiss_index_path="../faiss_store")
    
    # ================================ Checkpoint Loading ================================
    checkpoint_data = llm_checkpoint(args.model, save_name=args.save_name, save_dir=args.save_dir) if args.from_checkpoint==1 else None
    
    # ================================ Configure wandb ================================
    if args.use_wandb:
        import wandb
        wandb_id = checkpoint_data.get('wandb_id') if checkpoint_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"AdaptiveRAG-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
        
    # ================================ Initialize Model ================================
    # Policy模型
    model, tokenizer = init_model(args.model, args.save_name, device=args.device)
    # Reference模型
    # ref_model, _ = init_model(args.model, args.save_name, device=args.device)
    ref_model = model
    ref_model = ref_model.eval().requires_grad_(False)
    # 数据和优化器
    train_ds = RAGDataset(jsonl_path=args.data_path, tokenizer=tokenizer)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = None
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    scheduler = None
    
    # ================================ Recover from Checkpoint ================================
    start_epoch, start_step = 0, 0
    if checkpoint_data:
        model.load_state_dict(checkpoint_data['model'])
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        start_epoch = checkpoint_data['epoch']
        start_step = checkpoint_data.get('step', 0)
        Logger(f"Recovered from checkpoint at epoch {start_epoch} and step {start_step}")
        
    # ================================ DDP Wrapper ================================
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ================================ Train ================================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping {start_step} steps, starting from step {start_step + 1}')
            gspo_train_epoch(epoch, loader, model, tokenizer, ref_model, retriever, start_step, wandb if args.use_wandb else None, optimizer, scheduler, args.device, args)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: starting from scratch')
            gspo_train_epoch(epoch, loader, model, tokenizer, ref_model, retriever, 0, wandb if args.use_wandb else None, optimizer, scheduler, args.device, args)