import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def no_peak_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask == 0
    return mask.unsqueeze(0) 

def create_masks(src, trg, device, pad_idx):
    
    src_mask = (src != pad_idx).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = no_peak_mask(size).to(device)
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

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