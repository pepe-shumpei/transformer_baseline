import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext import data
import numpy as np

import random

from Models import Transformer


def no_peak_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask == 0
    return mask.unsqueeze(0) 


def create_masks(src, trg, device, pad_idx):
    
    #src_mask = (src != opt.src_pad).unsqueeze(-2)
    #1 -> padding_idxに修正する
    src_mask = (src != pad_idx).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = no_peak_mask(size).to(device)
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def lr_schedule(step):
    step = step + 1
    a = 512 ** (-0.5)
    b = min([step ** (-0.5), step*4000**(-1.5)])
    return a * b 

#batch_size = 1のみ   
def generate_sentence(seq, Dict, file):
    seq_len = len(seq)
    sentence_list = []
    for s in range(seq_len):
         word = Dict[seq[s].item()]
         if word == "<eos>":
             break
         if word == "<sos>":
             continue
         sentence_list.append(word)
    sentence = " ".join(sentence_list)
    file.write(sentence + "\n")
    return sentence

#ミニバッチ対応
def output_file(seq, Dict, file):
    #file = open(file_name, "w", encoding="utf8")
    batch_size = seq.shape[0]
    seq_len = seq.shape[1]
    for b in range(batch_size):
        sentence_list = []
        for s in range(seq_len-1):
            word = Dict[seq[b][s+1].item()]
            if word == "<eos>":
                break
            sentence_list.append(word)
        sentence = " ".join(sentence_list)
        file.write(sentence + "\n")

def sentence_to_list(sentence_list, seq, Dict, ref=False):
    batch_size = seq.shape[0]
    seq_len = seq.shape[1]
    for b in range(batch_size):
        sentence = []
        for s in range(seq_len-1):
            word = Dict[seq[b][s+1].item()]
            if word == "<eos>":
                break
            #refとoutのunkを別語彙にする
            elif not ref and word == "<unk>":
                word = "UNK"

            sentence.append(word)
        sentence = remove_at(sentence)
        if ref:
            sentence_list.append([sentence])
        else:
            sentence_list.append(sentence)
#remove @@
def remove_at(sentence_list):
    sentence_str = " ".join(sentence_list)
    sentence_str = sentence_str.replace("@@ ","")
    sentence_list = sentence_str.split()
    return sentence_list

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

"""
#token数でbatchを作る
def create_batch_sampler(src, trg, max_token):
    indices = torch.arange(len(src)).tolist()
    random.shuffle(indices)
    indices = sorted(indices, key=lambda idx: len(src[idx]))
    sorted_indices = sorted(indices, key=lambda idx: len(trg[idx]))

    batch_indices = []
    idx = 0
    while True:
        indices = []
        num_src_token = 0
        num_trg_token = 0
        src_max_length = 0 
        #max_token = 5000
        
        while True:
            indices.append(sorted_indices[idx])
            num_trg_token += len(trg[sorted_indices[idx]])
            num_src_token += len(src[sorted_indices[idx]])
            if src_max_length < len(src[sorted_indices[idx]]):
                src_max_length = len(src[sorted_indices[idx]])

            idx += 1
            if len(src) == idx:
                break
            if num_trg_token > max_token:
                break
            if len(indices)*src_max_length > max_token:
                break
        
        batch_indices.append(indices)

        if len(src) == idx:
            break

    return batch_indices
"""

def create_batch_sampler(src, trg, batch_size):
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

def load_model(model_num, opt):
    model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout)
    model.load_state_dict(torch.load(opt.save_model + "/n_" + str(model_num) + ".model"))
    return model

def average_model(end_point, opt):
    end_point = end_point+1
    start_point = end_point -5
    models = [load_model(m, opt) for m in range(start_point, end_point)]

    opt.model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(opt.device)
    state_dict = opt.model.state_dict()
    state_dict0 = models[0].state_dict()
    state_dict1 = models[1].state_dict()
    state_dict2 = models[2].state_dict()
    state_dict3 = models[3].state_dict()
    state_dict4 = models[4].state_dict()

    for k in state_dict.keys():
        state_dict[k] = state_dict0[k] + state_dict1[k] + state_dict2[k] + state_dict3[k] + state_dict4[k]
        state_dict[k] = state_dict[k]/5            

    opt.model.load_state_dict(state_dict)