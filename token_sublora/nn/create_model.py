import os 
import pickle 
import torch 

import loralib as lora
from collections import OrderedDict

from token_sublora.nn.model import GPTConfig, GPT
from token_sublora.nn.projectors import create_intrinsic_model
from token_sublora.nn.cola_nn import colafy
from token_sublora.nn.quip.lib.utils.unsafe_import import model_from_hf_path

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(n_layer, n_head, n_embd, bias, dropout, use_mergedlinear, apply_rope, use_mistral_sliding_window, 
              use_lora, lora_alpha, lora_dropout, attention_linear_use_lora, attention_linear_lora_r,linear_head_lora_r, 
              linear_head_enable_lora, mlp_lora_r, mlp_enable_lora, intrinsic_dim, block_size, data_dir, out_dir, init_from, master_process, device,
              best_checkpoint_path, optimize_alpha, optimize_alpha_strategy, kron_order, linear_head_bias, use_struct_approx_kron, use_struct_approx_monarch, 
              layers_applied, monarch_nblocks, kron_factorized_mode, debug, finetuned_quip):
        
        curr_user = os.getenv('USER')
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        if data_dir != None:
            meta_path = os.path.join(data_dir, 'meta.pkl')
            meta_vocab_size = None
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                meta_vocab_size = meta['vocab_size']
                print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, bias=bias, vocab_size=None, dropout=dropout, optimize_alpha=optimize_alpha,
                          use_mergedlinear=use_mergedlinear, apply_rope=apply_rope, use_lora=use_lora, lora_alpha=lora_alpha, optimize_alpha_strategy=optimize_alpha_strategy,
                          lora_dropout=lora_dropout, attention_linear_use_lora=attention_linear_use_lora, block_size=block_size, linear_head_bias=linear_head_bias,
                          attention_linear_lora_r=attention_linear_lora_r, linear_head_lora_r=linear_head_lora_r, intrinsic_dim=intrinsic_dim, 
                          linear_head_enable_lora=linear_head_enable_lora, mlp_lora_r=mlp_lora_r, mlp_enable_lora=mlp_enable_lora, use_mistral_sliding_window=use_mistral_sliding_window, use_struct_approx_kron=use_struct_approx_kron, 
                          use_struct_approx_monarch=use_struct_approx_monarch, layers_applied=layers_applied, monarch_nblocks=monarch_nblocks, kron_factorized_mode=kron_factorized_mode, debug=debug)
        if init_from == 'scratch':
            # init a new model from scratch
            print("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if meta_vocab_size is None:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            
            if use_struct_approx_kron: # TODO: manual config here; need to put config into config files
                model = colafy(model=model, struct='kron',layers=layers_applied, device=device, kron_factorized_mode=kron_factorized_mode)                
                for n, p in model.named_parameters():
                    if 'transformer.wte.weight' in n:
                        p.requires_grad = False
                model_copy = GPT(gptconf)
                model_copy = colafy(model=model_copy, struct='kron',layers=layers_applied, device=device, kron_factorized_mode=kron_factorized_mode)
                model_copy.load_state_dict(model.state_dict())  # Copy the state (parameters and buffers)
                
                for n, p in model_copy.named_parameters():
                    if 'transformer.wte.weight' in n:
                        p.requires_grad = False
            elif use_struct_approx_monarch:
                monarch_keys_to_check = ['lm_head', 'blkdiag1', 'blkdiag2', 'bias']

                model = colafy(model=model, struct='monarch',layers=layers_applied, device=device, monarch_nblocks=monarch_nblocks, debug=debug)

                for n, p in model.named_parameters():
                    if all(key not in n for key in monarch_keys_to_check): 
                        p.requires_grad = False
                
                model_copy = GPT(gptconf)
                model_copy = colafy(model=model_copy, struct='monarch',layers=layers_applied, device=device, monarch_nblocks=monarch_nblocks)
                model_copy.load_state_dict(model.state_dict())  # Copy the state (parameters and buffers)

                for n, p in model_copy.named_parameters():
                    if all(key not in n for key in monarch_keys_to_check): 
                        p.requires_grad = False
            else:
                model_copy = None

            nparams = int(model.get_num_params())
            
            if master_process:
                torch.save(model.state_dict(), os.path.join(out_dir, 'forward_ckpt_at_random_initialization.pt'))
            
            print("INTRINSIC DIM IS: ", intrinsic_dim)
            
            if block_size < model.config.block_size:
                model._forward_net[0].crop_block_size(block_size)
                model_args['block_size'] = block_size # so that the checkpoint will have the right value
            
            if use_lora:
                for n, p in model.named_parameters():
                    if 'transformer.wte.weight' in n:
                        p.requires_grad = False
            
            if intrinsic_dim > 0:              
                model = create_intrinsic_model(base_net=model, ckpt_path=None, intrinsic_mode="rdkronqr", intrinsic_dim=intrinsic_dim,
                                               seed=137, device=device, order=kron_order, optimize_alpha=optimize_alpha, model_copy=model_copy)
                
            # crop down the model block size if desired, using model surgery
            if intrinsic_dim == 0:
                if block_size < model.config.block_size:
                    model._forward_net[0].crop_block_size(block_size)
                    model_args['block_size'] = block_size # so that the checkpoint will have the right value

            if master_process:
                torch.save(model.state_dict(), os.path.join(out_dir, 'ckpt_at_random_initialization.pt'))
                if intrinsic_dim > 0:
                    torch.save(model.trainable_initparams, os.path.join(out_dir, 'trainable_initparams.pt'))
                    torch.save(model.names, os.path.join(out_dir, 'names.pt'))


        elif init_from == 'best_ckpt':
            print(f"loading best training checkpoint from {best_checkpoint_path} for pretraining bound metrics eval") 
            ckpt_path = os.path.join(best_checkpoint_path, "best_ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'use_lora', 'lora_alpha', 'lora_dropout',
                      'attention_linear_use_lora', 'attention_linear_lora_r', 'linear_head_lora_r', 'linear_head_enable_lora']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

            if use_struct_approx_kron: # TODO: manual config here; need to put config into config files
                # colafy base model
                model = colafy(model=model, struct='kron',layers=layers_applied, device=device, kron_factorized_mode=kron_factorized_mode)
                for n, p in model.named_parameters():
                    if 'matrix_params' not in n:
                        p.requires_grad = False
                
                model_copy = GPT(gptconf)
                model_copy = colafy(model=model_copy, struct='kron',layers=layers_applied, device=device, kron_factorized_mode=kron_factorized_mode)
                model_copy.load_state_dict(model.state_dict())  # Copy the state (parameters and buffers)
                for n, p in model_copy.named_parameters():
                    if 'matrix_params' not in n:
                        p.requires_grad = False
            elif use_struct_approx_monarch:
                monarch_keys_to_check = ['lm_head', 'blkdiag1', 'blkdiag2', 'bias']

                model = colafy(model=model, struct='monarch',layers=layers_applied, device=device, monarch_nblocks=monarch_nblocks)
                for n, p in model.named_parameters():
                    if all(key not in n for key in monarch_keys_to_check): 
                        p.requires_grad = False
                
                model_copy = GPT(gptconf)
                model_copy = colafy(model=model_copy, struct='monarch',layers=layers_applied, device=device, monarch_nblocks=monarch_nblocks)
                model_copy.load_state_dict(model.state_dict())  # Copy the state (parameters and buffers)
                for n, p in model_copy.named_parameters():
                    if all(key not in n for key in monarch_keys_to_check): 
                        p.requires_grad = False
            else:
                model_copy = None


            nparams = int(model.get_num_params())
            
            if use_lora: 
                assert (not use_struct_approx_kron)
                assert (not use_struct_approx_monarch)
                lora.mark_only_lora_as_trainable(model)
                            
            if intrinsic_dim > 0:
                #### loading the random initialization of all the weights 
                init_ckpt_path = os.path.join(best_checkpoint_path, "forward_ckpt_at_random_initialization.pt")
                init_checkpoint = torch.load(init_ckpt_path, map_location=device)
                unwanted_prefix = '_orig_mod.'
                for k,v in list(init_checkpoint.items()):
                    if k.startswith(unwanted_prefix):
                        init_checkpoint[k[len(unwanted_prefix):]] = init_checkpoint.pop(k)

                if optimize_alpha:
                    model.load_state_dict(init_checkpoint, strict=False)
                else:
                    model.load_state_dict(init_checkpoint)
                    
                if optimize_alpha:
                    for n, p in model.named_parameters():
                        if "lm_alpha_head" in n:
                            p.requires_grad = True
                
                model = create_intrinsic_model(base_net=model,
                                            ckpt_path=None,
                                            intrinsic_mode="rdkronqr",
                                            intrinsic_dim=intrinsic_dim,
                                            seed=137,
                                            device=device,
                                            order=kron_order,
                                            optimize_alpha=optimize_alpha,
                                            )

            state_dict = checkpoint['raw_model']
            
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    
            if intrinsic_dim > 0:   
                if optimize_alpha:
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict)
                    
                print('subspace_params loaded!')

                model.trainable_initparams = torch.load(os.path.join(best_checkpoint_path, "trainable_initparams.pt"), map_location=device)
                model.names = torch.load(os.path.join(best_checkpoint_path, "names.pt"))
                
            else:
                state_dict = change_keys_of_ordered_dict(state_dict)
                model.load_state_dict(state_dict)
                                
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            
            checkpoint = None # free up memory
        elif init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
            
            if "finetune" in best_checkpoint_path:
                if init_from == "gpt2":
                    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

                elif init_from == "gpt2-large":
                    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
                else:
                    raise ValueError

                model_ckpt = torch.load(os.path.join(best_checkpoint_path)) 
                
                model.load_state_dict(model_ckpt, strict=True)

                iter_num = None 
                best_val_loss = None
                nparams = sum(p.numel() for p in model.parameters())

            else:
                override_args = dict(dropout=dropout)
                model = GPT.from_pretrained(init_from, override_args)
            
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                    model_args[k] = getattr(model.config, k)

                iter_num = None 
                best_val_loss = None
                nparams = int(model.get_num_params())


        elif "relaxml" in init_from:
            model, model_str = model_from_hf_path(init_from, use_cuda_graph=False,use_flash_attn=False, cache_dir=f'/scratch/{curr_user}/cache')

            iter_num = None 
            best_val_loss = None
            model_args = None
            nparams = None 

            # finetuned_quip
            from collections import OrderedDict

            if finetuned_quip == "GSM8K":
                
                finetune_ckpt = torch.load(best_checkpoint_path)

                # Qidxs_0 -> Qidxs
                new_dict = OrderedDict()

                for old_key, value in finetune_ckpt.items():
                    if old_key.endswith('Qidxs_0'):
                        new_key = old_key[:-2]
                    else:
                        new_key = old_key
                    new_dict[new_key] = value

                model.load_state_dict(new_dict, strict=True)            

        elif init_from == "meta-llama/Llama-2-7b-chat-hf":
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            iter_num = None 
            best_val_loss = None
            model_args = None
            nparams = None 
        else:
            raise NotImplemented
        
        model.to(device)
        return model, iter_num, best_val_loss, model_args, nparams

def get_quip_model(init_from):
    curr_user = os.getenv('USER')
    model, model_str = model_from_hf_path(init_from, use_cuda_graph=False, use_flash_attn=False, cache_dir=f'/scratch/{curr_user}/cache')
    return model, model_str

def change_keys_of_ordered_dict(original_dict):
    new_dict = OrderedDict()
    for old_key, value in original_dict.items():
        if "matrix_params.0" in old_key:
            new_key = old_key.replace("matrix_params.0", "matrix_params_U")
        elif "matrix_params.1" in old_key:
            new_key = old_key.replace("matrix_params.1", "matrix_params_V")
        else:
            new_key = old_key
        new_dict[new_key] = value
    return new_dict
