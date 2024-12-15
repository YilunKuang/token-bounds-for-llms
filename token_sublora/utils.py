import os
import torch 
import numpy as np 
import math 
import bisect
import json
import pandas as pd
import random 

def get_doc_indices(train_data, eot_token, init_from, base_path="TOADD"):

    openwebtext_train_eot_indices_file = os.path.join(base_path, 'openwebtext_train_eot_indices_file_full.npy') 
    empirical_document_length_distribution_file = os.path.join(base_path, 'empirical_document_length_distribution_full.npy')

    if os.path.exists(openwebtext_train_eot_indices_file):
        openwebtext_train_eot_indices = np.load(openwebtext_train_eot_indices_file)
        empirical_document_length_distribution = np.load(empirical_document_length_distribution_file)
    else:
        # openwebtext_train_eot_indices
        openwebtext_train_eot_indices =  np.where(train_data==eot_token)
        openwebtext_train_eot_indices = openwebtext_train_eot_indices[0]
        openwebtext_train_eot_indices_shift_left_by_1 = np.insert(openwebtext_train_eot_indices[:-1], 0, 0)
        # empirical length distribution
        empirical_document_length_distribution = openwebtext_train_eot_indices - openwebtext_train_eot_indices_shift_left_by_1

        with open(openwebtext_train_eot_indices_file, 'wb') as f_openwebtext_train_eot_indices_file: 
            np.save(f_openwebtext_train_eot_indices_file, openwebtext_train_eot_indices)
        
        with open(empirical_document_length_distribution_file, 'wb') as f_empirical_document_length_distribution:
            np.save(f_empirical_document_length_distribution, empirical_document_length_distribution)
            
    return openwebtext_train_eot_indices, empirical_document_length_distribution

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # TODO force-coded this, maybe revisit
    min_lr = learning_rate / 10
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters 
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(split, train_data, val_data, batch_size, block_size, device_type, device, perturb_word_order_window_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    newx = torch.stack([torch.from_numpy((data[i:i+1+block_size]).astype(np.int64)) for i in ix])
        
    if perturb_word_order_window_size > 1:
        for i, batch_i in enumerate(newx):
            if perturb_word_order_window_size==1024:
                # 100% shuffling
                newx[i] = newx[i][torch.randperm(len(newx[i]))]
            elif perturb_word_order_window_size < 1024:
                num_of_windows = block_size // perturb_word_order_window_size
                counter_i = 0
                while counter_i < num_of_windows:
                    sequence_segment = newx[i][counter_i*perturb_word_order_window_size:(counter_i+1)*perturb_word_order_window_size]
                    shuffled_indices = torch.randperm(perturb_word_order_window_size)
                    shuffled_sequence_segment = sequence_segment[shuffled_indices]
                    newx[i][counter_i*perturb_word_order_window_size:(counter_i+1)*perturb_word_order_window_size] = shuffled_sequence_segment
                    counter_i += 1
            else:
                raise ValueError
    x = newx[:,:-1]
    y = newx[:,1:]
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix

def sample_single_document(split, train_data, val_data, eot_token, device_type, device, init_from):
    '''
    This function is used for bounds evaluation where we're sampling a single document `doc_i` at a time to get log p(`doc_i`)
    '''
    
    openwebtext_train_eot_indices, empirical_document_length_distribution = get_doc_indices(train_data, eot_token, init_from)
    
    # specify data split
    data = train_data if split == 'train' else val_data

    # sample a random document from openwebtext with replacement
    random_iter = np.random.randint(0, int((len(openwebtext_train_eot_indices))))
        
    # get document start and end index & document length
    ix = (openwebtext_train_eot_indices[random_iter]-empirical_document_length_distribution[random_iter], openwebtext_train_eot_indices[random_iter])
    length_ix = empirical_document_length_distribution[random_iter]

    x = torch.from_numpy((data[ix[0]+1:ix[0]+length_ix]).astype(np.int64)) 
    y = torch.from_numpy((data[ix[0]+1+1:ix[0]+length_ix+1]).astype(np.int64))
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x.unsqueeze(0), y.unsqueeze(0), torch.tensor(ix)

def sample_nonoverlapping_sequences(split, train_data, val_data, batch_size, block_size, device_type, device, data_size):
    
    upper_bound = (data_size-1)//block_size
    lower_bound = 0
    chunk_idx = np.random.randint(lower_bound, upper_bound, size=(batch_size))
        
    data = train_data if split == 'train' else val_data
    ix = (chunk_idx[:,None]*block_size+np.arange(block_size)) # a (bs, block_size) set of ids
    x = torch.from_numpy((data[ix]).astype(np.int64)) # assuming the broadcasting is correct
    y = torch.from_numpy((data[ix+1]).astype(np.int64))
        
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, ix

def sample_token_batch(split, train_data, val_data, batch_size, block_size, eot_token, device_type, device, init_from):
    
    data = train_data if split == 'train' else val_data
    
    openwebtext_train_eot_indices, _ = get_doc_indices(train_data, eot_token, init_from)
    
    upper_bound = len(data)
    lower_bound = 0
    ix = np.random.randint(lower_bound, upper_bound, size=(batch_size))
    
    newx = torch.full((batch_size, block_size+1), eot_token)
    
    lengths = []
    
    for i, idx in enumerate(ix):
        begin_idx = openwebtext_train_eot_indices[bisect.bisect_left(openwebtext_train_eot_indices, idx)-1] + 1
        token_array = torch.from_numpy((data[begin_idx:idx+1]).astype(np.int64))[-(block_size+1):]
        newx[i, :len(token_array)] = token_array 
        lengths.append(len(token_array))
        
    x = newx[:,:-1]
    y = newx[:,1:]

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
                
    return x, y, ix, lengths 

def sample_token_batch_from_llm360(categorical_dist, total_dataset_files, list_of_number_of_tokens_in_the_file, batch_size, block_size, eot_token):
    list_of_x = []
    list_of_y = []
    list_of_starting_index = []

    for i in range(batch_size):
        curr_index = categorical_dist.sample().item()
        curr_file = total_dataset_files[curr_index]

        with open(curr_file + '.len', 'r') as file:
            curr_file_statistics = json.load(file)

        curr_file_token_count = list_of_number_of_tokens_in_the_file[curr_index]
        curr_data = np.memmap(curr_file, dtype=np.uint16, mode='r')
        assert curr_data.shape[0]==curr_file_statistics['number_of_tokens_in_the_file']

        # get index for token
        random_token_index = torch.randint(low=0, high=curr_file_statistics['number_of_tokens_in_the_file'],size=(1,)).item()

        # edge case 
        if random_token_index <= block_size:
            starting_index = 0
        else:
            starting_index = random_token_index - block_size
        
        if random_token_index == curr_data.shape[0]-1:
            # account for the label Y
            ending_index = random_token_index - 1
        else: 
            ending_index = random_token_index

        x = curr_data[starting_index:ending_index]

        if eot_token in x:
            starting_index_increment = np.argwhere(x==eot_token)[0].item()

            # edge case
            if starting_index_increment == block_size-1:
                ending_index -= 1
            else:
                starting_index += starting_index_increment
        
        # update x by eliminating edge cases
        x = curr_data[starting_index:ending_index]
        y = curr_data[starting_index+1:ending_index+1]

        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))

        list_of_x.append(x)
        list_of_y.append(y)
        list_of_starting_index.append(starting_index)

    x = torch.nn.utils.rnn.pad_sequence(list_of_x, batch_first=True, padding_value=eot_token)
    y = torch.nn.utils.rnn.pad_sequence(list_of_y, batch_first=True, padding_value=-1)
    attention_mask = x.ne(eot_token)
        
    # assuming cuda all the time! 
    x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
    ix = np.array([list_of_starting_index])
    lengths = list(map(lambda x: len(x), list_of_x))

    return x, y, ix, lengths, attention_mask
