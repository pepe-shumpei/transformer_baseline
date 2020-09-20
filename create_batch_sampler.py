import random

import numpy as np
import torch 

#token数でbatchを作る
def create_token_batch_sampler(src, trg, max_token):
    indices = torch.arange(len(src)).tolist()
    random.shuffle(indices)
    indices = sorted(indices, key=lambda idx: len(src[idx]))
    sorted_indices = sorted(indices, key=lambda idx: len(trg[idx]))

    batch_indices = []
    idx = 0
    while True:
        indices = []
        num_trg_token = 0        
        while True:
            indices.append(sorted_indices[idx])
            num_trg_token += len(trg[sorted_indices[idx]])
            idx += 1
            if len(src) == idx:
                break
            if num_trg_token > max_token:
                break
        
        batch_indices.append(indices)

        if len(src) == idx:
            break

    return batch_indices

def create_sentence_batch_sampler(src, trg, batch_size):
    indices = torch.arange(len(src)).tolist()
    random.shuffle(indices)
    indices = sorted(indices, key=lambda idx: len(src[idx]))
    sorted_indices = sorted(indices, key=lambda idx: len(trg[idx]))

    sorted_indices = np.array(sorted_indices)
    n_divide = len(sorted_indices)//batch_size
    sorted_indices = list(np.array_split(sorted_indices, n_divide))
    for i in range(len(sorted_indices)):
        sorted_indices[i] = sorted_indices[i].tolist()
        
    return sorted_indices