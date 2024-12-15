### Code partially inspired from https://github.com/karpathy/nanoGPT 
import os
import sys
import time
import datetime
from contextlib import nullcontext
import wandb
import numpy as np
import yaml
import random
from pathlib import Path
import transformers

import json
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader

from fastargs.decorators import param
from fastargs import Param, Section
from transformers import AutoTokenizer

import loralib as lora

from token_sublora.nn.create_model import get_model
from token_sublora.bounds.bound_utils import quantize_model, compute_bound_scores, compute_bound_metrics, compute_token_bound_metrics
from token_sublora.bounds.compute_bounds import llm_subsampling_bound 
from token_sublora.utils import get_lr, get_batch, sample_single_document, sample_nonoverlapping_sequences, sample_token_batch, sample_token_batch_from_llm360, sample_token_batch_from_oas

from token_sublora.nn.antibody_llm.language_models import setup_model

from data.oas.data import (
    setup_clone_datasets,
    DataCollatorForSeqLabelsDataset,
)


Section("training", "training details").params(
    gradient_accumulation_steps=Param(int, "used to simulate larger batch sizes", default=40),
    backend=Param(str, "ddp setting; 'nccl', 'gloo', etc.", default='nccl'),
    eval_interval=Param(int, "", default=500),
    log_interval=Param(int, "", default=1),
    eval_iters=Param(int, "", default=200),
    eval_only=Param(bool, "if True, script exits right after the first eval", default=False),
    always_save_checkpoint=Param(bool, "if True, always save a checkpoint after each eval", default=True),
    max_iters=Param(int, "total number of training iterations", default=600000),
)

Section("login", "login details").params(
    wandb_log=Param(bool, "disabled by default", default=False),
    wandb_project=Param(str, "name of the project", default='gpt-2'),
    wandb_run_name=Param(str, "name of the run", default='train'),
    out_dir=Param(str, "where to save results?", default=None),
    create_new_output_dir=Param(bool, "default is True", default='True'),
)

Section("data", "data details").params(
    dataset_dir=Param(str, "name of the dataset", default=None),
    dataset=Param(str, "where to find the dataset?", default='openwebtext'),
    batch_size=Param(int, "f gradient_accumulation_steps > 1, this is the micro-batch size", default=12),
    block_size=Param(int, "size of the sequence", default=1024),
    vocab_size=Param(int, "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)", default=50257),
    data_size=Param(int, "Number of sequences in the dataset of size = block_size", default=8823811),
    num_docs=Param(int, "Number of documents in the dataset", default=8009762),
    eot_token=Param(int, "Set EOT tokens for identifying a single document within openwebtext. {'<|endoftext|>': 50256}", default=50256),
    perturb_word_order_window_size=Param(int, "perturbations window within which we apply random permutations", default=0),
)

Section("model", "model details").params(
    n_layer=Param(int, "", default=12),
    n_head=Param(int, "", default=12),
    n_embd=Param(int, "", default=768),
    dropout=Param(float, "for pretraining 0 is good, for finetuning try 0.1+", default=0.0),
    bias=Param(bool, "do we use bias inside LayerNorm and Linear layers?", default=False),
    linear_head_bias=Param(bool, "do we use bias in the linear head or not?", default=False),
    use_mergedlinear=Param(bool, "merged linear or linear for the attention layers?", default=False),
    apply_rope=Param(bool, "apply rope instead of learned positional embeddings", default=False),
    use_mistral_sliding_window=Param(bool, "apply rope instead of learned positional embeddings", default=False),
    init_from=Param(str, "'scratch' or 'best_ckpt' if computing the bounds", default='scratch'),
    best_checkpoint_path=Param(str, "path to best checkpoint for bound eval", default=None),
    finetuned_quip=Param(str, "dataset used for finetuning QuIP models", default=None),
    init_model_path=Param(str, "path to the model to be used for initialization", default=None),
)

Section("optimizer", "optimizer details").params(
    learning_rate=Param(float, "adamw optimizer lr", default=6e-4),
    weight_decay=Param(float, "", default=1e-1),
    beta1=Param(float, "", default=0.9),
    beta2=Param(float, "", default=0.95),
    grad_clip=Param(float, "# clip gradients at this value, or disable if == 0.0", default=1.0),
    correct_bias=Param(bool, "", default=False),
    adam_epislon=Param(float, "", default=1e-8),
    no_decay_bias=Param(bool, "", default=True),
)

Section("learning_rate", "learning rate decay settings").params(
    decay_lr=Param(bool, "whether to decay the learning rate", default=True),
    warmup_iters=Param(int, "how many steps to warm up for", default=2000),
    lr_decay_iters=Param(int, "should be ~= max_iters per Chinchilla", default=600000),
    min_lr=Param(float, "minimum learning rate, should be ~= learning_rate/10 per Chinchilla", default=6e-5),
)

Section("sublora", "LoRA and subspace Settings").params(
    use_lora=Param(bool, "true if any LoRA layer is used", default=False),
    use_struct_approx_kron=Param(bool, "true if any Kronecker structure approximation layer is used", default=False),
    use_struct_approx_monarch=Param(bool, "true if any Monarch matrix approximation layer is used", default=False),
    layers_applied=Param(str, "specify which layers to apply Monarch or Kronecker compression", default='attn_and_lm_head'),
    monarch_nblocks=Param(int, "specify number of blocks in Monarch blockdiag matrices formulation", default=4),
    kron_factorized_mode=Param(int, "choose from 1 to 8; specify the mode of Kronecker factorization (6 total mode, each mode represent one way of n=a*b for different a and b per mode)", default=4),
    lora_alpha=Param(int, "default value", default=32),
    lora_dropout=Param(float, "default value", default=0.1),
    attention_linear_use_lora=Param(bool, "default value", default=False),
    attention_linear_lora_r=Param(int, "", default=1),
    linear_head_lora_r=Param(int, "", default=1),
    linear_head_enable_lora=Param(bool, "", default=False),
    mlp_lora_r=Param(int, "", default=1),
    mlp_enable_lora=Param(bool, "", default=False),
    intrinsic_dim=Param(int, "subspace intrinsic dimensionality", default=0),
    proj_kron_order=Param(int, "order of the kronecker factorization", default=2),
)

Section("system", "system details").params(
    device=Param(str, "examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks", default='cuda'),
    dtype=Param(str, "'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler",
                default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'),
    compile=Param(bool, "use PyTorch 2.0 to compile the model to be faster", default=False), 
)

Section("bounds", "bound computation details").params(
    use_kmeans=Param(bool, "Using kmeans during the quantization", default=False),
    quant_lr=Param(float, "Learning rate for quantization-aware training", default=5e-5),
    eval_batch_size=Param(int, "Batch size for quantization-aware training and bound evaluation", default=6),
    max_quant_iters=Param(int, "Number of iterations for quantization-aware training", default=0),
    levels=Param(int, "Number of quantization levels, should be odd", default=11),
    bound_samples=Param(int, "number of samples used in the subsampling bounds", default=10000),
    bound_type=Param(str, "'document_level' bounds or 'sequence_level' bounds", default="document_level"),
    sliding_window_size=Param(int, "the length of the sliding window in the evaluation of a doc > 1024 tokens", default=100),
    misc_extra_bits=Param(int, "number of extra bits to be paid for sweeping over multiple hyperparameters", default=5),
    use_quip=Param(bool, "use quip-trained model or not", default=False),
    quip_model=Param(str, "select from quip model zoo", default='TOADD'),
    quip_model_cache_dir=Param(str, "quip model cache dir", default='TOADD'),
    zip_message_len=Param(int, "The message length of a zipped checkpoint", default=0),
    optimize_alpha=Param(bool, "if True: optimize alpha at the token level", default=False),
    optimize_alpha_strategy=Param(str, "how to optimize alpha at the token level", default='not_optimized'),
    alpha_optim_scale=Param(float, "Scale of the loss terms in the alpha optimization objective.", default=0.8),
)

Section("alpha_optim", "alpha optimization details").params(
    warmup_iters=Param(int, "how many steps to warm up for", default=0),
    learning_rate=Param(float, "adamw optimizer lr", default=2e-4),
    lr_decay_iters=Param(int, "should be ~= max_iters per Chinchilla", default=1000),
    min_lr=Param(float, "minimum learning rate, should be ~= learning_rate/10 per Chinchilla", default=2e-5),
    decay_lr=Param(bool, "whether to decay the learning rate", default=True),
    weight_decay=Param(float, "", default=1e-2),
    beta1=Param(float, "", default=0.9),
    beta2=Param(float, "", default=0.95),
    correct_bias=Param(bool, "", default=False),
    adam_epislon=Param(float, "", default=1e-8),
    no_decay_bias=Param(bool, "", default=True),
    eval_interval=Param(int, "", default=20),
    gradient_accumulation_steps=Param(int, "used to simulate larger batch sizes", default=10),
    grad_clip=Param(float, "# clip gradients at this value, or disable if == 0.0", default=1.0),
    log_interval=Param(int, "", default=10),
    max_iters=Param(int, "total number of training iterations", default=200),
)

Section("analysis", "explorative analysis").params(
    analyze_quantization=Param(bool, "explorative analysis on quantization", default=False), 
    debug=Param(bool, "turn to True if debugging", default=False), 
)

class SubLoRA():
    @param("data.dataset")
    @param("data.dataset_dir")
    @param("data.block_size")
    @param("data.batch_size")
    @param("data.perturb_word_order_window_size")
    @param("model.init_from")
    @param("model.n_layer")
    @param("model.n_head")
    @param("model.n_embd")
    @param("model.bias")
    @param("model.dropout")
    @param("model.use_mergedlinear")
    @param("model.apply_rope")
    @param("model.use_mistral_sliding_window")
    @param("sublora.use_lora")
    @param("sublora.use_struct_approx_kron")
    @param("sublora.use_struct_approx_monarch")
    @param("sublora.layers_applied")
    @param("sublora.monarch_nblocks")
    @param("sublora.kron_factorized_mode")
    @param("sublora.lora_alpha")
    @param("sublora.lora_dropout")
    @param("sublora.intrinsic_dim")
    @param("sublora.attention_linear_use_lora")
    @param("sublora.attention_linear_lora_r")
    @param("sublora.linear_head_lora_r")
    @param("sublora.linear_head_enable_lora")
    @param("sublora.mlp_lora_r")
    @param("sublora.mlp_enable_lora")
    @param("bounds.optimize_alpha")
    @param("bounds.optimize_alpha_strategy")
    @param("bounds.alpha_optim_scale")
    @param("sublora.proj_kron_order")
    @param("model.linear_head_bias")
    @param("analysis.debug")
    @param("model.finetuned_quip")
    @param("model.init_model_path")
    @param("model.best_checkpoint_path")
    def __init__(self, yaml_config, dataset, dataset_dir, block_size, batch_size, perturb_word_order_window_size, init_from, 
                 n_layer, n_head, n_embd, bias, dropout, use_mergedlinear, apply_rope, use_mistral_sliding_window, use_lora, 
                 use_struct_approx_kron, use_struct_approx_monarch, layers_applied, monarch_nblocks, kron_factorized_mode,
                 lora_alpha, lora_dropout, intrinsic_dim, attention_linear_use_lora, attention_linear_lora_r, linear_head_lora_r,
                 linear_head_enable_lora, mlp_lora_r, mlp_enable_lora, optimize_alpha, optimize_alpha_strategy, alpha_optim_scale, 
                 proj_kron_order, linear_head_bias, debug, finetuned_quip, init_model_path, best_checkpoint_path=None):        

        self.yaml_config = yaml_config
        self.debug = debug
        self.optimize_alpha = optimize_alpha
        self.optimize_alpha_strategy = optimize_alpha_strategy
        self.alpha_optim_scale = alpha_optim_scale
        self.block_size = block_size
        self.batch_size = batch_size
        self.perturb_word_order_window_size = perturb_word_order_window_size
        self.intrinsic_dim = intrinsic_dim
        self.use_lora = use_lora
        self.use_struct_approx_kron = use_struct_approx_kron
        self.use_struct_approx_monarch = use_struct_approx_monarch
        self.layers_applied = layers_applied
        self.monarch_nblocks = monarch_nblocks
        self.kron_factorized_mode = kron_factorized_mode
        print("Setting up the ddp.")
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        print("Preparing the common setup.")
        self.prepare_common_setup()
        print("Loading the data.")
        self.dataset = dataset
        self.finetuned_quip = finetuned_quip

        if 'openwebtext' in dataset:
            self.data_dir = os.path.join(dataset_dir, dataset)
            self.train_data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            self.val_data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        elif 'oas' in dataset:
            self.data_dir = None 
            self.train_data = None 
            self.val_data = None

            self.model, self.tokenizer = setup_model(
                MAX_LENGTH=2048, 
                model_name=init_from, 
                vocab_txt_file_path="ADDPATH/data/oas/vocab.txt", 
                load_model=True, 
                load_path=init_model_path,
            )
            self.model = self.model.cuda()

            datasets = setup_clone_datasets(chain_type="heavy", data_path=Path("/"), min_clone_size=25, use_clone_attention=False, task="train", tokenizer=self.tokenizer)
            self.total_dataset_files = datasets['train'].filenames+datasets['val'].filenames+datasets['test'].filenames

            with open("ADDPATH/data/oas/merged_file_counts_dict.json", 'r') as file:
                self.list_of_number_of_tokens_in_the_file = json.load(file)

            categorical_dist_prob = np.array(list(self.list_of_number_of_tokens_in_the_file.values())) / np.sum(np.array(list(self.list_of_number_of_tokens_in_the_file.values())))
            self.categorical_dist = torch.distributions.categorical.Categorical(torch.from_numpy(categorical_dist_prob))

        else:
            self.data_dir = None 
            self.train_data = None 
            self.val_data = None

            with open('ADDPATH/data/llm360/total_dataset_files.json', 'r') as f_list:
                total_dataset_files = json.load(f_list)
                self.total_dataset_files = [filename[:-4] if filename.endswith(".len") else filename for filename in total_dataset_files]

            self.list_of_number_of_tokens_in_the_file = np.load('ADDPATH/data/llm360/list_of_number_of_tokens_in_the_file.npy')
            categorical_dist_prob = self.list_of_number_of_tokens_in_the_file / np.sum(self.list_of_number_of_tokens_in_the_file)

            self.categorical_dist = torch.distributions.categorical.Categorical(torch.from_numpy(categorical_dist_prob))

            self.tokenizer = AutoTokenizer.from_pretrained(init_from)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.init_from = init_from 

        if 'oas' in dataset:
            self.iter_num, self.best_val_loss, self.model_args, self.nparams = None, None, None, None
        else:
            print("Loading the model.")
            self.model, self.iter_num, self.best_val_loss, self.model_args, self.nparams  = get_model(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                                                                                    bias=bias, dropout=dropout, use_mergedlinear=use_mergedlinear,
                                                                                                    apply_rope=apply_rope, use_mistral_sliding_window=use_mistral_sliding_window,
                                                                                                    use_lora=use_lora, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                                                                    attention_linear_use_lora=attention_linear_use_lora, attention_linear_lora_r=attention_linear_lora_r,
                                                                                                    linear_head_lora_r=linear_head_lora_r, linear_head_enable_lora=linear_head_enable_lora, 
                                                                                                    mlp_lora_r=mlp_lora_r, mlp_enable_lora=mlp_enable_lora,
                                                                                                    intrinsic_dim=intrinsic_dim, block_size=block_size, data_dir=self.data_dir, out_dir=self.out_dir,
                                                                                                    init_from=init_from, master_process=self.master_process, device=self.device,
                                                                                                    best_checkpoint_path=best_checkpoint_path, optimize_alpha=self.optimize_alpha,
                                                                                                    optimize_alpha_strategy=optimize_alpha_strategy, kron_order=proj_kron_order,
                                                                                                    linear_head_bias=linear_head_bias,use_struct_approx_kron=self.use_struct_approx_kron, 
                                                                                                    use_struct_approx_monarch=self.use_struct_approx_monarch, layers_applied=self.layers_applied, 
                                                                                                    monarch_nblocks=self.monarch_nblocks, kron_factorized_mode=self.kron_factorized_mode, debug=self.debug, 
                                                                                                    finetuned_quip=self.finetuned_quip)

    @param("optimizer.weight_decay")
    @param("optimizer.learning_rate")
    @param("optimizer.beta1")
    @param("optimizer.beta2")
    @param("optimizer.correct_bias")
    @param("optimizer.adam_epislon")
    @param("optimizer.no_decay_bias")
    @param("system.dtype")
    @param("learning_rate.decay_lr")
    @param("learning_rate.warmup_iters")
    @param("learning_rate.lr_decay_iters")
    @param("learning_rate.min_lr")
    @param("training.eval_interval")
    @param("training.always_save_checkpoint")
    @param("training.eval_only")
    @param("training.gradient_accumulation_steps")
    @param("optimizer.grad_clip")
    @param("training.log_interval")
    @param("training.max_iters")
    @param("system.compile")
    def train(self, weight_decay, learning_rate, beta1, beta2, correct_bias, adam_epislon, no_decay_bias, dtype, decay_lr,
              warmup_iters, lr_decay_iters, min_lr, eval_interval, always_save_checkpoint, eval_only, gradient_accumulation_steps,
              grad_clip, log_interval, max_iters, compile,):
        print("Training beings...")
        # clear file contents
        iter_num = self.iter_num
        best_val_loss = self.best_val_loss
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), self.device_type,
                                               correct_bias, adam_epislon, no_decay_bias)
        checkpoint = None # free up memory

        # compile the model
        if compile and (not self.use_lora and not self.use_struct_approx_kron and not self.use_struct_approx_monarch) and self.intrinsic_dim == 0:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0
        
        # wrap model into DDP container

        if self.ddp:
            if self.use_lora or self.use_struct_approx_kron or self.use_struct_approx_monarch:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank], find_unused_parameters=True)
            else:
                self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        
        if self.use_lora:
            if self.intrinsic_dim == 0:
                pass 
        if self.ddp:
            total_num_params = int(self.model.module.get_num_params())
            num_trainable_params = int(self.model.module.get_num_params(only_trainable=True))
        else:
            total_num_params = int(self.model.get_num_params())
            num_trainable_params = int(self.model.get_num_params(only_trainable=True))
            
        if self.wandb_log and self.master_process:
            wandb.log({"num_params": total_num_params, "num_trainable_params": num_trainable_params})

        print("\n# === final trainable parameters === #\n")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n)
        print("\n# === final trainable parameters === #\n")

        torch.manual_seed(1337 + self.seed_offset)
        # training loop
        X, Y, ix = get_batch('train', self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if self.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    if losses['val'] < best_val_loss:
                        _best_checkpoint = True
                    else:
                        _best_checkpoint = False

                    best_val_loss = losses['val']
                    if iter_num > 0:
                        if self.use_lora:
                            raw_model_state_dict = raw_model.state_dict()
                            lora_state_dict = lora.lora_state_dict(self.model)
                        else:
                            raw_model_state_dict = raw_model.state_dict()
                            lora_state_dict = None

                        checkpoint = {
                            'raw_model': raw_model_state_dict,
                            'lora_model': lora_state_dict,
                            'optimizer': optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': self.model_args,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        
                        if _best_checkpoint:
                            torch.save(checkpoint, os.path.join(self.out_dir, f'best_ckpt.pt'))
                        else:
                            torch.save(checkpoint, os.path.join(self.out_dir, f'ckpt_{iter_num}.pt'))
            if iter_num == 0 and eval_only:
                break

            for micro_step in range(gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                if self.intrinsic_dim == 0:
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                        loss = loss / gradient_accumulation_steps 
                else:
                    logits, loss = self.model(X, Y)
                    loss = loss / gradient_accumulation_steps 
                
                X, Y, ix = get_batch('train', self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
                
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)            
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and self.master_process:
                lossf = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, {dt*1000:.2f}ms")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break

        if self.ddp:
            destroy_process_group()


    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    @param("training.eval_iters")
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            og_losses = torch.zeros(eval_iters)
            alphas_mean = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, ix = get_batch(split, self.train_data, self.val_data, self.batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
                
                if self.intrinsic_dim == 0:
                    if self.optimize_alpha:
                        with self.ctx:
                            _, loss, alphas, og_loss = self.model(X, Y)        
                    else:
                        with self.ctx:
                            if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
                                output = self.model(input_ids=X, labels=X)
                                loss = output.loss
                            else:
                                _, loss = self.model(X, Y)
                else:
                    if self.optimize_alpha:
                        _, loss, alphas, og_loss = self.model(X, Y)
                    else:
                        _, loss = self.model(X, Y)
                losses[k] = loss.item()
                if self.optimize_alpha: 
                    og_losses[k] = og_loss.item()
                    alphas_mean[k] = alphas.mean().item() 
            out[split] = losses.mean()
            if self.optimize_alpha: 
                label = split + "_og"
                out[label] = og_losses.mean()
                label = split + "_mean_alphas"
                out[label] = alphas_mean.mean()
        self.model.train()
        return out


    @torch.no_grad()
    def estimate_loss_llm360(self, eot_token, eval_iters=10):
        out = {}
        self.model.eval()
        for split in ['llm360']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, ix, lengths, attention_mask = sample_token_batch_from_llm360(self.categorical_dist,
                self.total_dataset_files,
                self.list_of_number_of_tokens_in_the_file,
                self.batch_size, self.block_size, eot_token)
                
                if "relaxml" in self.yaml_config['quip_model'] or "meta-llama" in self.yaml_config['quip_model']:
                    output = self.model(input_ids=X)
                    loss = F.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)), Y.reshape(-1), ignore_index=-1)
                else:
                    with self.ctx:
                        _, loss = self.model(X, Y)

                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    @param("system.dtype")
    @param("login.wandb_log")
    @param("login.wandb_project")
    @param("login.wandb_run_name")
    @param("login.create_new_output_dir")
    @param("login.out_dir")
    @param("sublora.intrinsic_dim")
    @param("optimizer.learning_rate")
    @param("sublora.attention_linear_lora_r")
    def prepare_common_setup(self, dtype, wandb_log, wandb_project, wandb_run_name, create_new_output_dir, out_dir,
                             intrinsic_dim, learning_rate, attention_linear_lora_r):
        self.maybe_launch_ddp()
        if self.debug:
            wandb_log = False
        self.wandb_log = wandb_log
        self.out_dir = out_dir
        wandb_run_name = "id{}_lr{}".format(intrinsic_dim, learning_rate)

        if wandb_log and self.master_process:
            wandb.init(project=wandb_project, name=wandb_run_name, config=self.yaml_config)
        if create_new_output_dir:
            now = datetime.datetime.now()
            formatted_date = now.strftime('%Y-%m-%d')
            formatted_time = now.strftime('%H-%M')
            logging_directory = f'{formatted_date}/{formatted_time}'
            self.out_dir = os.path.join(self.out_dir, wandb_project, wandb_run_name, logging_directory)

        if self.master_process and (self.yaml_config["action"] == "train" or self.yaml_config["action"] == "memorization"):
            os.makedirs(self.out_dir, exist_ok=True)
        torch.manual_seed(137)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
                
    @param("training.gradient_accumulation_steps")
    @param("training.backend")
    @param("system.device")
    def maybe_launch_ddp(self, gradient_accumulation_steps, backend, device): 
        self.device = device
        if self.ddp:
            init_process_group(backend=backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 
            self.seed_offset = self.ddp_rank 
            assert gradient_accumulation_steps % self.ddp_world_size == 0
            gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.ddp_rank = 0
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            
    @param("alpha_optim.warmup_iters")
    @param("alpha_optim.learning_rate")
    @param("alpha_optim.lr_decay_iters")
    @param("alpha_optim.min_lr")
    @param("alpha_optim.decay_lr")
    @param("alpha_optim.weight_decay")
    @param("alpha_optim.beta1")
    @param("alpha_optim.beta2")
    @param("alpha_optim.correct_bias")
    @param("alpha_optim.adam_epislon")
    @param("alpha_optim.no_decay_bias")
    @param("alpha_optim.eval_interval")
    @param("alpha_optim.gradient_accumulation_steps")
    @param("alpha_optim.grad_clip")
    @param("alpha_optim.log_interval")
    @param("alpha_optim.max_iters")
    @param("bounds.eval_batch_size")
    @param("system.dtype")
    def optimize_alpha(self, warmup_iters, learning_rate, lr_decay_iters, min_lr, decay_lr, weight_decay, beta1, beta2, 
                       correct_bias, adam_epislon, no_decay_bias, eval_interval, gradient_accumulation_steps, grad_clip, 
                       log_interval, max_iters, eval_batch_size, dtype):
        
        print("Optimizing alpha beings...")
        grad_params = []
        # turn off the params we don't want to optimize
        for n, p in self.model.named_parameters():
            if "alpha" not in n:
                p.requires_grad = False
                grad_params.append(n)
        print("\n# === final trainable parameters === #\n")
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n)
        print("\n# === final trainable parameters === #\n")
        
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), self.device_type,
                                               correct_bias, adam_epislon, no_decay_bias)
        
        torch.manual_seed(1337 + self.seed_offset)
        
        X, Y, ix = get_batch('train', self.train_data, self.val_data, eval_batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
        t0 = time.time()
        local_iter_num = 0 
        raw_model = self.model 
        while True:
            lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {iter_num}: train bound loss {losses['train']:.4f}, val bound loss {losses['val']:.4f}, NLL train loss {losses['train_og']:.4f}, NLL val loss {losses['val_og']:.4f}")

                if self.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "train/og_loss": losses['train_og'],
                        "val/og_loss": losses['val_og'],
                        "train/mean_alphas": losses['train_mean_alphas'],
                        "val/mean_alphas": losses['val_mean_alphas'],
                        "lr": lr,
                    })
                    
                if losses['val'] < best_val_loss:
                    if losses['val'] < best_val_loss:
                        _best_checkpoint = True
                    else:
                        _best_checkpoint = False

                    best_val_loss = losses['val']
                    if iter_num > 0:
                        raw_model_state_dict = raw_model.state_dict()
                        lora_state_dict = None

                        checkpoint = {
                            'raw_model': raw_model_state_dict,
                            'lora_model': lora_state_dict,
                            'optimizer': optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': self.model_args,
                        }
                        
                        print(f"saving checkpoint to {self.out_dir}")
                        
                        if _best_checkpoint:
                            torch.save(checkpoint, os.path.join(self.out_dir, f'best_alpha_ckpt.pt'))

            for micro_step in range(gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                if self.intrinsic_dim == 0:
                    with self.ctx:
                        logits, loss, alphas, og_loss = self.model(X, Y)
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                        og_loss = og_loss / gradient_accumulation_steps 
                        if self.optimize_alpha_strategy=="weighted_loss": 
                            loss = self.alpha_optim_scale * og_loss + (1 - self.alpha_optim_scale) * loss 
                else:
                    logits, loss, alphas, og_loss = self.model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    og_loss = og_loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    if self.optimize_alpha_strategy=="weighted_loss": 
                        loss = self.alpha_optim_scale * og_loss + (1 - self.alpha_optim_scale) * loss   
                
                X, Y, ix = get_batch('train', self.train_data, self.val_data, eval_batch_size, self.block_size,
                                     self.device_type, self.device, self.perturb_word_order_window_size)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and self.master_process:
                lossf = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num}: bound loss {lossf:.4f}, OG loss time {og_loss:.4f}, {dt*1000:.2f}ms")
            iter_num += 1
            local_iter_num += 1
            
            if iter_num > max_iters:
                break
            
        # turn the gradient on again 
        for n, p in self.model.named_parameters():
            if n in grad_params: 
                p.requires_grad = True
    
    @param("bounds.max_quant_iters")
    @param("bounds.use_kmeans")
    @param("bounds.levels")
    @param("bounds.quant_lr")
    @param("bounds.eval_batch_size")
    @param("bounds.bound_samples")
    @param("bounds.bound_type")
    @param("bounds.misc_extra_bits")
    @param("bounds.sliding_window_size")
    @param("model.best_checkpoint_path")
    @param("data.data_size")
    @param("data.eot_token")
    @param("data.vocab_size")
    @param("data.num_docs")
    @param("analysis.analyze_quantization")
    @param("bounds.use_quip")
    @param("bounds.quip_model")
    @param("bounds.quip_model_cache_dir")
    @param("model.finetuned_quip")
    @param("bounds.zip_message_len")
    def get_bounds(self, max_quant_iters, use_kmeans, levels, quant_lr, eval_batch_size, bound_samples, bound_type, 
                   misc_extra_bits, sliding_window_size, best_checkpoint_path, data_size, eot_token, vocab_size, num_docs, 
                   analyze_quantization, use_quip, quip_model, quip_model_cache_dir, finetuned_quip, zip_message_len):
        if use_quip:
            pass
        else:
            # wrap model into DDP container
            if self.ddp:
                if self.use_lora or self.use_struct_approx_kron or self.use_struct_approx_monarch:
                    self.model = DDP(self.model, device_ids=[self.ddp_local_rank], find_unused_parameters=True)
                else:
                    self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

            if self.use_lora:
                if self.intrinsic_dim == 0:
                    lora.mark_only_lora_as_trainable(self.model)

        
        if self.optimize_alpha: 
            self.optimize_alpha()
        
        if analyze_quantization:
            torch.manual_seed(5)
            list_of_train_val_loss_before_quantize = [losses['train'], losses['val']]
            list_of_train_losses = []
            list_of_val_losses = []
            list_of_levels_i = [_ for _ in range(2, 100, 5)]

            if self.intrinsic_dim > 0:
                curr_module = self.model.module if self.ddp else self.model
                curr_module_subspace_params_copy = curr_module.subspace_params.data
            else:
                raise NotImplementedError

            for levels_i in list_of_levels_i:
                self.model, prefix_message_len = quantize_model(self.model, self.train_data, self.block_size, self.intrinsic_dim,
                                                                self.device_type, self.device, self.ddp, self.perturb_word_order_window_size,
                                                                eval_batch_size, max_quant_iters, use_kmeans, levels_i, quant_lr)
                print("EVALUATING THE MODEL AFTER QUANTIZATION")
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, levels_i={levels_i}")

                list_of_train_losses.append(losses['train'])
                list_of_val_losses.append(losses['val'])

                self.model.subspace_params.data = curr_module_subspace_params_copy
            
            list_of_train_val_loss_before_quantize = torch.tensor(list_of_train_val_loss_before_quantize)
            list_of_train_losses = torch.tensor(list_of_train_losses)
            list_of_val_losses = torch.tensor(list_of_val_losses)
            list_of_levels_i = torch.tensor(list_of_levels_i)

        else:
            if use_quip:
                pass
            else:
                self.model, prefix_message_len = quantize_model(self.model, self.train_data, self.block_size, self.intrinsic_dim,
                                                                self.device_type, self.device, self.ddp, self.perturb_word_order_window_size,
                                                                eval_batch_size, max_quant_iters, use_kmeans, levels, quant_lr)

        
        if "Llama" in self.init_from: 
            # compute the message length from the zipped checkpoint 
            message_len = zip_message_len 
            prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0


        if self.dataset == "llm360":
            losses = self.estimate_loss_llm360(eot_token)
            print(f"llm360: loss {losses}")        
        else:
            losses = self.estimate_loss()
            
            print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")        
        
        if use_quip:
            pass
        else:
            raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
            if self.use_lora:
                raw_model_state_dict = raw_model.state_dict()
                lora_state_dict = lora.lora_state_dict(self.model)
            else:
                raw_model_state_dict = raw_model.state_dict()
                lora_state_dict = None

            checkpoint = {
                'raw_model': raw_model_state_dict,
                'lora_model': lora_state_dict,
                'optimizer': None,
                'model_args': self.model_args,
                'iter_num': self.iter_num,
                'best_val_loss': None,
                'config': self.yaml_config,
                'prefix_message_len': prefix_message_len, 
            }
            print(f"saving checkpoint to {best_checkpoint_path}")

            torch.save(checkpoint, os.path.join(best_checkpoint_path, f'quant_ckpt_levels{levels}_iters{max_quant_iters}.pt'))
        
        if not self.optimize_alpha:
            alpha_array = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]    
                
        with torch.no_grad():
            self.model.eval()
            curr_iter_i = 0
            metrics_dict = {}
            for k in range(1,10+1):
                metrics_dict[f'top_{k}_acc'] = 0
            metrics_dict[f'top_50_acc'] = 0
            metrics_dict[f'top_100_acc'] = 0
            
            if self.optimize_alpha:
                bounds_dict["best_bpd_bound"] = 0
            else:
                for alpha in alpha_array:
                    metrics_dict[f'bpd_alpha_{alpha}'] = 0
             
            metrics_dict["n_train"] = 0
            metrics_dict["curr_iter_i"] = 0
            
            while bound_samples > metrics_dict["n_train"]:
                
                lengths = None

                if bound_type == "sequence_level":
                    X, Y, ix = sample_nonoverlapping_sequences("train", self.train_data, self.val_data, eval_batch_size, self.block_size,
                                                               self.device_type, self.device, data_size)
                elif bound_type == "document_level":
                    X, Y, ix = sample_single_document("train", self.train_data, self.val_data, eot_token, self.device_type,
                                                      self.device, self.init_from)
                elif bound_type == "token_level":                    
                    if self.dataset == "llm360":
                        X, Y, ix, lengths, attention_mask = sample_token_batch_from_llm360(self.categorical_dist, self.total_dataset_files, self.list_of_number_of_tokens_in_the_file, self.batch_size, self.block_size, eot_token)
                    else:
                        X, Y, ix, lengths = sample_token_batch("train", self.train_data, self.val_data, eval_batch_size, self.block_size, eot_token, 
                                                    self.device_type, self.device, self.init_from)
                else:
                    raise NotImplemented


                top_k_indices, percentile_vec, selected_prob_scores = compute_bound_scores(self.model, X, Y, self.device,
                                                                                           self.intrinsic_dim, self.block_size,
                                                                                           sliding_window_size, self.ctx, use_quip)                
                    
                if bound_type == "token_level": 
                    metrics_dict = compute_token_bound_metrics(metrics_dict, top_k_indices, selected_prob_scores, alpha_array, eval_batch_size, vocab_size, self.block_size, lengths)
                else:
                    metrics_dict = compute_bound_metrics(metrics_dict, top_k_indices, selected_prob_scores, alpha_array,
                                            bound_type, eval_batch_size, vocab_size, len_x=X.shape[1])
                    
                if self.wandb_log:
                    wandb.log(metrics_dict)
                if curr_iter_i % 100 == 0:
                    print("\n".join("{}\t{}".format(k, v) for k, v in metrics_dict.items()))
                curr_iter_i += 1                

        if use_quip:     
            pass
        else:
            prefix_message_len = torch.load(os.path.join(best_checkpoint_path, f'quant_ckpt_levels{levels}_iters{max_quant_iters}.pt'))['prefix_message_len']
    
        sample_size = metrics_dict["n_train"]

        bounds_dict = {}
        bounds_dict["prefix_message_len"] = float(prefix_message_len)
        
        best_bpd_bound = np.inf
        
        if bound_type == "sequence_level":
            total_sample_size = data_size
        elif bound_type == "document_level":
            total_sample_size = num_docs
        elif bound_type == "token_level":
            if self.dataset == "llm360":
                total_sample_size = self.list_of_number_of_tokens_in_the_file.sum()
            else:
                total_sample_size = len(self.train_data)
                    
        for k in metrics_dict.keys():
            if k != "n_train" and k != "curr_iter_i":
                if "acc" in k:
                    train_error = 1. - metrics_dict[k] 
                    divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
                    bounds_dict["acc_divergence"] = float(divergence)
                    bounds_dict[f"bound_{k}"] = float(llm_subsampling_bound(train_error=train_error,
                                                        div=divergence,
                                                        data_size=total_sample_size,
                                                        sample_size=sample_size,
                                                        delta=1.))
                else:
                    misc_extra_bits += np.ceil(len(alpha_array))
                    divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
                    bounds_dict["bpd_divergence"] = float(divergence)
                    alpha = float(k.replace("bpd_alpha_", ""))
                    delta = np.log2(1 + (1 - alpha) * vocab_size / alpha)
                    train_error = metrics_dict[k]
                    bounds_dict[f"bound_{k}"] = float(llm_subsampling_bound(train_error=train_error,
                                                                            div=divergence,
                                                                            data_size=total_sample_size,
                                                                            sample_size=sample_size,
                                                                            delta=delta))
                    
                    if best_bpd_bound > bounds_dict[f"bound_{k}"]:
                        best_bpd_bound = bounds_dict[f"bound_{k}"]
                                
        bounds_dict["best_bpd_bound"] = best_bpd_bound
        
        print("\n".join("{}\t{}".format(k, v) for k, v in bounds_dict.items()))

        if self.wandb_log:
            wandb.log(bounds_dict)

