import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
import math
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import deepspeed
from torch.utils.data.sampler import Sampler

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        logging.info(content)
        
def setup_logging():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank != 0:
        logging.getLogger().setLevel(logging.CRITICAL) 

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            deepspeed.init_distributed(dist_backend="nccl")
        else:
            dist.init_process_group(backend="gloo")

        dist.barrier()
        return local_rank
    else:
        print('Not using distributed mode')
        
        
def get_lr(warmup_steps: int, current_step: int, total_steps: int, lr: float) -> float:
    if current_step < warmup_steps:
        return lr * current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * progress))
    
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def deepspeed_checkpoint(model_name, save_name, model_engine, client_state, save_dir='../checkpoints', tag=None):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{save_name}"
    model_engine.save_checkpoint(save_path, client_state=client_state, tag=tag)
    Logger(f"Checkpoint saved to {save_path}")

def llm_checkpoint(model_name: str, save_name: str = 'full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = f"{save_dir}/{model_name}_{save_name}.pth"
    resume_path = f"{save_dir}/{model_name}_{save_name}.pth"
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        ckp_tmp = checkpoint_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, checkpoint_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
                
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
                resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:
        if os.path.exists(resume_path):
            checkpoint_data = torch.load(resume_path, map_location='cpu')
            saved_ws = checkpoint_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                checkpoint_data['step'] = checkpoint_data['step'] * saved_ws // current_ws
                Logger(f"GPU number changed {saved_ws}→{current_ws}, step reset to {checkpoint_data['step']}")
            return checkpoint_data
        return None
    
def init_model(model_name: str, save_name:str, save_dir: str = '../out', device: str = 'auto', dtype: str = 'float32'):
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    torch_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if save_name is not None:
        weight_path = f'{save_dir}/{model_name}{save_name}.pth'
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            Logger(f"Loaded weights from {weight_path} to {model_name}")
    model.to(device)
    Logger(f"Model {model_name} initialized on {device} with dtype {dtype}")
    return model, tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)