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
from trainer.trainer_utils import init_distributed_mode, Logger, setup_seed, SkipBatchSampler, is_main_process, deepspeed_checkpoint, setup_logging
import torch.distributed as dist
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from dataset.dataset import RAGDataset
from rag_pipeline.retriever import FaissLocalRetriever
from rag_pipeline.utils import break_condition, extract_subtopics
from utils.metrics import normalize_text
from rag_pipeline.prompt import ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT, ADAPTIVE_GENERATOR_QUERY_PROMPT, ADAPTIVE_GENERATOR_SYSTEM_PROMPT
import torch.nn.functional as F
from tqdm import tqdm
import deepspeed
from contextlib import nullcontext

def get_ds_config(args):
    ds_config = {
        "train_batch_size": args.batch_size * args.num_generations * dist.get_world_size() * args.accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size * args.num_generations,
        "gradient_accumulation_steps": args.accumulation_steps,
        "steps_per_print": args.log_interval,
        "gradient_clipping": args.grad_clip,
        "fp16": { "enabled": args.dtype == "float16" },
        "bf16": { "enabled": args.dtype == "bfloat16" },
        
        # ================= ZeRO-3 core configuration =================
        "zero_optimization": {
            "stage": args.stage,

            # offload parameters to CPU memory
            "offload_param": {
                "device": "cpu", 
                "pin_memory": True
            } if args.offload else None, 

            # offload optimizer to CPU memory
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if args.offload else None,
            "allgather_partitions": False,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_param_persistence_threshold": 1e4, 
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_max_live_parameters": 3e7,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        # ===================================================

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.total_optimizer_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        }
    }
    return ds_config

def calc_reward(args, prompts: List[str], responses: List[List[str]], contexts: List[List[str]], ground_truth: List[List[str]], answers: List[str]) -> torch.Tensor:
    def context_gt_reward(contexts: List[str], gt: List[str], current_step: int, total_step: int) -> float:
        for ctx in contexts:
            for v in gt:
                if normalize_text(v) in normalize_text(ctx):
                    return 2.0
        return 0.0
    
    def answer_gt_reward(answer: str, gt: List[str]) -> float:
        for v in gt:
            if normalize_text(v) in normalize_text(answer):
                return 3.0
        return -2.0
    
    def ctx_penalty(contexts: List[str], gt: List[str], current_step: int, gt_step: int) -> float:
        if gt_step != -1 and current_step - gt_step > 1:
            for ctx in contexts:
                for v in gt:
                    if normalize_text(v) in normalize_text(ctx):
                        return -0.5
            return -1.0
        return 0.0
    
    def foramt_reward(response: str):
        pattern = r"<subtopic>(.*?)</subtopic>"
        matches = re.findall(pattern, response, re.DOTALL)
        rewards = 0.0
        if len(matches) != 0:
            rewards += 1.0
        if len(matches) > 5:
            rewards -= 1.5
        if len(matches) == 0 and response != "<terminate>":
            rewards -= 2.5
        match_set = set()
        for match in matches:
            if match.strip() in match_set:
                rewards -= 0.3
            else:
                match_set.add(match.strip())
        def mark_num(text: str, matches: List[str]) -> int:
            reward = 0
            if text.count("<subtopic>") == text.count("</subtopic>"):
                reward += 0.1
            if text.count("<subtopic>") != len(matches):
                reward -= 0.1 * abs(text.count("<subtopic>") - len(matches))
            if text.count("</subtopic>") != len(matches):
                reward -= 0.1 * abs(text.count("</subtopic>") - len(matches))
            return reward
        rewards += mark_num(response, matches)
        if response.endswith("</subtopic>"):
            rewards += 0.05            
        return rewards
    
    rewards = torch.zeros(len(responses), device=args.device)
    batch_size = len(prompts)
    for i in range(batch_size):
        for j in range(args.num_generations):
            response_idx = i * args.num_generations + j
            response = responses[response_idx]
            context = contexts[response_idx]
            answer = answers[response_idx]
            gt = ground_truth[response_idx]
            gt_step = -1
            response_set = set()
            for idx, r in enumerate(response):
                if r in response_set:
                    rewards[response_idx] -= 0.3
                else:
                    response_set.add(r)
                rewards[response_idx] += foramt_reward(r)
                if break_condition(r) or idx == len(response) - 1:
                    if idx != len(response) - 1:
                        rewards[response_idx] -= 0.2
                    rewards[response_idx] += answer_gt_reward(answer, gt)
                    # terminate word reward
                    if "<terminate>" == r:
                        rewards[response_idx] += 0.4
                    else:
                        rewards[response_idx] -= 0.3
                    continue
                ctx = context[idx]
                rewards[response_idx] += context_gt_reward(ctx, gt, idx, len(response)) + ctx_penalty(ctx, gt, idx, gt_step)
                if context_gt_reward(ctx, gt, idx, len(response)) > 0 and gt_step == -1:
                    gt_step = idx
                if r.lower().endswith("<terminate>"):
                    rewards[response_idx] -= 0.05
            rewards[response_idx] -= 1.0 * len(response)
    return rewards

def collate_fn(batch):
    return {
        'messages': [x['messages'] for x in batch],
        'ground_truth': [x['ground_truth'] for x in batch],
        'question': [x['question'] for x in batch]
    }
def get_per_token_logps(model, input_ids, n_keep):
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token_logps)

def compute_logps_merged(model, input_ids, attention_mask, completion_mask):
    """
    Calculate the log probabilities of the entire sequence at once.
    input_ids: [B, Seq_Len]
    completion_mask: [B, Seq_Len] (1 where we calculate loss, 0 otherwise)
    """
    # 1. one forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :] # [B, Seq-1, V]
    
    # 2. align labels
    labels = input_ids[:, 1:]
    
    # 3. calculate Log Softmax
    # gather: only keep the log probability of the ground truth tokens
    per_token_logps = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)
    
    # 4. align mask 
    mask = completion_mask[:, 1:]
    
    return per_token_logps * mask, mask

def stitch_trajectory(tokenizer, messages):
    """
    input: messages List[Dict] - complete conversation history [{"role": "user"...}, {"role": "assistant"...}...]
    output:
       input_ids: Tensor [Seq_Len]
       mask: Tensor [Seq_Len] (1 for assistant responses, 0 for user/system parts)
    """
    input_ids = []
    mask = []
    prev_ids = []
    current_history = []
    
    for msg in messages:
        current_history.append(msg)
        current_ids = tokenizer.apply_chat_template(
            current_history, 
            tokenize=True, 
            add_generation_prompt=False
        )
        
        # calculate the new tokens
        new_token_ids = current_ids[len(prev_ids):]
        input_ids.extend(new_token_ids)
        
        # determine the mask: if the message is from the assistant, mask=1; otherwise mask=0
        if msg['role'] == 'assistant':
            mask.extend([1] * len(new_token_ids))
        else:
            mask.extend([0] * len(new_token_ids))
            
        # update prev_ids
        prev_ids = current_ids

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def stitch_trajectory_step(tokenizer, messages: List[Dict], target_assistant_idx: int, max_seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct training samples for the "target_assistant_idx-th assistant output"
    Args:
    - tokenizer: The tokenizer to use
    - messages: The complete conversation history [{"role": "user"...}, {"role": "assistant"...}...]
    - target_assistant_idx: The index of the target assistant message, starting from 0
    - max_seq_len: Optional, cut input_ids and mask to max_seq_len to prevent OOM
    Returns:
    - input_ids: The input ids of the training samples
    - mask: The mask of the training samples
    """
    input_ids = []
    mask = []

    prev_ids = []
    current_history = []
    assistant_counter = 0

    for msg in messages:
        current_history.append(msg)

        current_ids = tokenizer.apply_chat_template(
            current_history,
            tokenize=True,
            add_generation_prompt=False
        )

        new_token_ids = current_ids[len(prev_ids):]

        new_mask = [0] * len(new_token_ids)

        if msg["role"] == "assistant":
            # mark the tokens with target_assistant_idx as 1
            if assistant_counter == target_assistant_idx:
                new_mask = [1] * len(new_token_ids)
            assistant_counter += 1

        input_ids.extend(new_token_ids)
        mask.extend(new_mask)

        prev_ids = current_ids

    # TOnly keep the last max_seq_len tokens
    if max_seq_len is not None and len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]
        mask = mask[-max_seq_len:]

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def grpo_train_epoch(epoch, loader, model_engine, tokenizer, ref_model, ans_model, retriever, start_step, wandb, args):
    for global_step, batch in enumerate(tqdm(loader, initial=start_step + 1, desc=f"Epoch {epoch}"), start=start_step + 1):
        messages = batch['messages']
        question = batch['question']
        gen_inputs = [copy.deepcopy(messages) for _ in range(args.num_generations)]
        
        total_samples = args.batch_size * args.num_generations
        batch_contexts = []
        batch_responses = []
        batch_answers = []
        with torch.no_grad():
            # use .module for DDP model
            gen_model = model_engine
            for generation_idx in range(args.num_generations):
                contexts = [[] for _ in range(args.batch_size)]
                responses = [[] for _ in range(args.batch_size)]
                batch_references = [[] for _ in range(args.batch_size)]
                answers = ["" for _ in range(args.batch_size)]
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
                    response_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
                    response_tokens = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    # Map back results to original indices
                    current_response_idx = 0
                    for i in range(args.batch_size):
                        if not active_mask[i]:
                            continue
                        
                        sample_idx = generation_idx * args.batch_size + i
                        
                        response_token = response_tokens[current_response_idx]
                        current_response_idx += 1
                        
                        # Randomly decide to terminate based on step
                        base_prob = 0.02
                        step_factor = 0.06
                        current_prob = base_prob + (step * step_factor)
                        current_prob = min(current_prob, 0.50)
                        
                        if torch.rand(1).item() < current_prob: 
                            response_token = "<terminate>"
                        
                        # Store response
                        responses[i].append(response_token)
                        gen_inputs[generation_idx][i].append({"role": "assistant", "content": response_token})
                            
                        sub_topics = extract_subtopics(response_token)
                        if len(sub_topics) > 5:
                            sub_topics = sub_topics[:5]
                        if break_condition(response_token) or len(sub_topics) == 0:
                            active_mask[i] = False
                            references = "\n".join(batch_references[i]) if len(batch_references[i]) > 0 else ""
                            generator_messages = [
                                {"role": "system", "content": ADAPTIVE_GENERATOR_SYSTEM_PROMPT},
                                {"role": "user", "content": ADAPTIVE_GENERATOR_QUERY_PROMPT.format(query=question[i], references=references)}
                            ]
                            input_text = tokenizer.apply_chat_template(generator_messages, tokenize=False, add_generation_prompt=True)
                            generator_inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
                            response = ans_model.generate(**generator_inputs, max_new_tokens=args.max_gen_len, temperature=1.0, top_p=0.9, top_k=40, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
                            response = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(generator_inputs.input_ids, response)
                            ]
                            answer = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
                            # print(answer)
                            answers[i] = answer
                            continue  
                            
                        question_ids = [f"{question[i]}_subq_{i}_{step}_{k}" for k in range(len(sub_topics))]
                        raw_contexts, retrieval_contexts = retriever.retrieve_batch(sub_topics, question_ids=question_ids, top_k=5)
                        
                        step_contexts = []
                        references = ""
                        for sub_topic in sub_topics:
                            ctx, scores = retrieval_contexts[f"{question[i]}_subq_{i}_{step}_{sub_topics.index(sub_topic)}"]
                            step_contexts.append(ctx[0]) # Flatten contexts
                            references += f"\nSub-topics: {sub_topic}\nRetrieved Contexts: {ctx[0]}\n"
                            batch_references[i].append(references)
                        
                        contexts[i].append(step_contexts) # Store all contexts for this step
                        if step != args.max_step - 1:
                            gen_inputs[generation_idx][i].append({"role": "user", "content": ADAPTIVE_ROUTER_SEQUENTIAL_PROMPT.format(question=question[i], references=references)})
                        else:
                            references = "\n".join(batch_references[i]) if len(batch_references[i]) > 0 else ""
                            generator_messages = [
                                {"role": "system", "content": ADAPTIVE_GENERATOR_SYSTEM_PROMPT},
                                {"role": "user", "content": ADAPTIVE_GENERATOR_QUERY_PROMPT.format(query=question[i], references=references)}
                            ]
                            input_text = tokenizer.apply_chat_template(generator_messages, tokenize=False, add_generation_prompt=True)
                            generator_inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
                            response = ans_model.generate(**generator_inputs, max_new_tokens=args.max_gen_len, temperature=1.0, top_p=0.9, top_k=40, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
                            response = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(generator_inputs.input_ids, response)
                            ]
                            answer = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
                            # print(answer)
                            answers[i] = answer
                            
                
                batch_contexts.append(contexts)
                batch_responses.append(responses)
                batch_answers.append(answers)
        # Flatten batch_contexts and batch_responses to [batch_size * num_generations]
        # Original structure: [num_generations, batch_size, steps]
        # Target structure: [batch_size * num_generations, steps]
        flat_batch_responses = []
        flat_batch_contexts = []
        flat_batch_gts = []
        flat_batch_messages = []
        flat_batch_answers = []
        for i in range(args.batch_size):
            for j in range(args.num_generations):
                flat_batch_responses.append(batch_responses[j][i])
                flat_batch_contexts.append(batch_contexts[j][i])
                flat_batch_gts.append(batch['ground_truth'][i])
                flat_batch_messages.append(gen_inputs[j][i])
                flat_batch_answers.append(batch_answers[j][i])
        rewards = calc_reward(args, question, flat_batch_responses, flat_batch_contexts, flat_batch_gts, flat_batch_answers)
        grouped_rewards = rewards.view(-1, args.num_generations)
        
        if args.debug:
            sampler = {
                "messages": flat_batch_messages,
                "response": flat_batch_responses,
                "context": flat_batch_contexts,
                "answers": flat_batch_answers,
                "ground_truth": flat_batch_gts,            
            }
            with open(f'debug/{epoch}/grpo_debug_epoch{epoch}_step{global_step}.json', 'w', encoding='utf-8') as f:
                json.dump(sampler, f, indent=4)
            
        # Advantages
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]
        # print(f"Final Shapes - LogP: {policy_logps.shape}, Adv: {advantages.shape}")
        
        
        # Flatten the trajectories into step samples
        step_input_ids_list = []
        step_masks_list = []
        traj_index_for_step = [] 
        
        for traj_idx, msgs in enumerate(flat_batch_messages):
            num_assistant = sum(1 for m in msgs if m["role"] == "assistant")
            for step_idx in range(num_assistant):
                ids, mask = stitch_trajectory_step(
                    tokenizer,
                    msgs,
                    target_assistant_idx=step_idx,
                    max_seq_len=args.max_seq_len
                )
                step_input_ids_list.append(ids)
                step_masks_list.append(mask)
                traj_index_for_step.append(traj_idx)

        
        if len(step_input_ids_list) == 0:
            continue

        traj_index_for_step = torch.tensor(traj_index_for_step, dtype=torch.long, device=args.device)
        advantages_step = advantages[traj_index_for_step]  # [num_steps]
        advantages_step = (advantages_step - advantages_step.mean()) / (advantages_step.std() + 1e-8)

        total_valid_tokens = 0
        for m in step_masks_list:
            total_valid_tokens += int(m.sum().item())
        total_valid_tokens = max(total_valid_tokens, 1)


        # Calculate logprob / KL / loss in chunks, to avoid OOM
        from torch.nn.utils.rnn import pad_sequence

        total_loss_tokens = 0.0 
        num_steps = len(step_input_ids_list)
        step_chunk_size =1
        for chunk_start in range(0, num_steps, step_chunk_size):
            chunk_end = min(chunk_start + step_chunk_size, num_steps)
            ids_chunk = step_input_ids_list[chunk_start:chunk_end]
            masks_chunk = step_masks_list[chunk_start:chunk_end]
            adv_chunk = advantages_step[chunk_start:chunk_end]

            input_ids = pad_sequence(
                ids_chunk, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(args.device)
            completion_mask = pad_sequence(
                masks_chunk, batch_first=True, padding_value=0
            ).to(args.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(args.device)

            policy_logps, valid_mask = compute_logps_merged(
                model_engine, input_ids, attention_mask, completion_mask
            )
            with torch.no_grad():
                ref_logps, _ = compute_logps_merged(
                    ref_model, input_ids, attention_mask, completion_mask
                )

            kl_div = ref_logps - policy_logps
            per_token_kl = torch.exp(kl_div) - kl_div - 1

            per_token_loss = -(
                torch.exp(policy_logps - policy_logps.detach()) * adv_chunk.unsqueeze(1)
                - args.beta * per_token_kl
            )

            loss_tokens = (per_token_loss * valid_mask).sum()
            total_loss_tokens += loss_tokens

        loss = (total_loss_tokens / total_valid_tokens) / args.accumulation_steps
        model_engine.backward(loss)
        model_engine.step()
        # torch.cuda.empty_cache()
            
        if global_step % args.log_interval == 0 or global_step == iters:
            if is_main_process():
                policy_loss_val = loss.item()
                avg_reward_val = rewards.mean().item()
                avg_len_val = valid_mask.sum(dim=1).float().mean().item()
                current_lr = model_engine.get_lr()[0]

                Logger(f'Epoch: {epoch+1}, Step: {global_step}/{iters}, '
                    f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                    f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

                if wandb and is_main_process():
                    wandb.log({
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": advantages.mean().item(),
                        "learning_rate": current_lr
                    })

        if (global_step % args.save_interval == 0 or global_step == iters - 1):
            wandb_id = wandb.run.id if (wandb is not None and wandb.run is not None) else None
            client_state = {'epoch': epoch, 'step': global_step, 'wandb_id': wandb_id}
            tag = f"epoch{epoch}_step{global_step}"
            deepspeed_checkpoint(args.model, args.save_name, model_engine, client_state, save_dir=args.save_dir, tag=tag)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="The directory to save the results")
    parser.add_argument("--save_name", type=str, default="grpo", help="The name of the save file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="The model to use")
    parser.add_argument("--num_generations", type=int, default=2, help="The number of generations to sample")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size")
    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs")
    parser.add_argument("--max_step", type=int, default=3, help="The max step for Adaptive RAG")
    parser.add_argument("--max_gen_len", type=int, default=512, help="The max generation length")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="The learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type to use")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers to load the data")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="The number of steps to accumulate the gradients")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="The gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=10, help="The number of steps to warmup the learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay")
    parser.add_argument("--log_interval", type=int, default=1, help="The interval to log the results")
    parser.add_argument("--save_interval", type=int, default=10, help="The interval to save the model")
    parser.add_argument("--from_checkpoint", default=0, type=int, choices=[0, 1], help="Whether to load the model from a checkpoint")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument("--wandb_project", type=str, default="AdaptiveRAG-GRPO", help="wandb project name")
    parser.add_argument("--data_path", type=str, default="../dataset/triviaqa_1000.jsonl", help="The path to load the data")
    parser.add_argument("--beta", type=float, default=0.02, help="The beta for the KL divergence")
    parser.add_argument("--offload", type=int, default=0, choices=[0,1], help="Whether to offload parameters and optimizer states to CPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed required argument")
    parser.add_argument("--stage", type=int, default=2, choices=[0,1,2,3], help="ZeRO optimization stage")
    parser.add_argument("--debug", type=int, default=1, choices=[0,1], help="Whether to debug training process by output into debug dir")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="The maximum sequence length for training")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # ================================ Initialize Env ================================
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f'cuda:{local_rank}'
    setup_seed(1508 + (dist.get_rank() if dist.is_initialized() else 0))
    setup_logging()
    
    # Initialize Retriever
    retriever = FaissLocalRetriever(embedding_model="BAAI/bge-m3", faiss_index_path="/root/autodl-tmp")
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if args.device == "mps" or args.device == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    if args.debug:
        if os.path.exists('debug'):
            if not args.from_checkpoint:
                import shutil
                shutil.rmtree('debug')
        os.makedirs('debug', exist_ok=True)
        
    # ================================ Initialize Model ================================
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = RAGDataset(jsonl_path=args.data_path, tokenizer=tokenizer)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    args.total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    
    ds_config = get_ds_config(args)
    
    # load the model
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        trust_remote_code=True
    )
    model_raw.gradient_checkpointing_enable()
    model_raw.config.use_cache = False
    
    # initialize the model with deepspeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model_raw,
        model_parameters=model_raw.parameters(),
        config=ds_config
    )
    # initialize the reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    )
    ref_model.config.use_cache = False
    ref_model.to(args.device).eval()
    
    ans_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    )
    # ans_model = deepspeed.init_inference(
    #     model=ans_model_raw,
    #     mp_size=dist.get_world_size(),
    #     dtype=torch.bfloat16,
    #     replace_method='auto',
    #     max_tokens=4096
    # )
    # ans_model.eval()
    ans_model.to(args.device).eval()
    # ans_model = None
    
    # ================================ Recover from Checkpoint ================================
    start_epoch, start_step = 0, 0
    client_state = {}
    if args.from_checkpoint and os.path.exists(f"{args.save_dir}/{args.save_name}"):
        _, client_state = model_engine.load_checkpoint(f"{args.save_dir}/{args.save_name}", tag=None)
        start_epoch = client_state.get('epoch', 0)
        start_step = client_state.get('step', 0)
        Logger(f"Resume training from checkpoint: epoch {start_epoch}, step {start_step}")
        
    # ================================ Configure wandb ================================
    if args.use_wandb:
        import wandb
        wandb_id = client_state.get('wandb_id', None)
        resume = 'must' if wandb_id else None
        wandb_run_name = f"AdaptiveRAG-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        if dist.get_rank() == 0:
            wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ================================ Train ================================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if args.debug:
                os.makedirs(f'debug/{epoch}', exist_ok=True)
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping {start_step} steps, starting from step {start_step + 1}')
            grpo_train_epoch(epoch, loader, model_engine, tokenizer, ref_model, ans_model, retriever, start_step, wandb if args.use_wandb else None, args)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: starting from scratch')
            grpo_train_epoch(epoch, loader, model_engine, tokenizer, ref_model, ans_model, retriever, 0, wandb if args.use_wandb else None, args)