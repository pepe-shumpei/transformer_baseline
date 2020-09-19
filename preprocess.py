import torch
from preprocessing import *
from dataset import MyDataset
from utils import create_batch_sampler

import random
import numpy as np

def preprocess(opt):
    
    source_vocab_path = "RESULT/" + opt.save + "/vocab/source_vocab"
    target_vocab_path = "RESULT/" + opt.save + "/vocab/target_vocab"

    #create vocab
    source_vocab = GetVocab(opt.train_src, opt.word_cut)
    target_vocab = GetVocab(opt.train_trg, opt.word_cut)
    with open(source_vocab_path, "w") as f:
        for key in source_vocab.keys():
            f.write(key + "\n")

    with open(target_vocab_path, "w") as f:
        for key in target_vocab.keys():
            f.write(key + "\n")
   
    #create dict
    pre_data = Preprocess()
    source_dict = pre_data.getVocab(source_vocab_path)
    target_dict = pre_data.getVocab(target_vocab_path)

    TrgDict = {}
    for key, value in target_dict.items():
        TrgDict[value] = key

    SrcDict = {}
    for key, value in source_dict.items():
        SrcDict[value] = key

    src_size = len(source_dict)
    trg_size = len(target_dict)

    padding_idx = source_dict["<pad>"]
    trg_sos_idx = source_dict["<sos>"]
    trg_eos_idx = source_dict["<eos>"]

    train_source = pre_data.load(opt.train_src , 1, source_dict)
    train_target = pre_data.load(opt.train_trg , 1, target_dict)
    valid_source = pre_data.load(opt.valid_src , 1, source_dict)
    valid_target = pre_data.load(opt.valid_trg , 1, target_dict)
    test_source = pre_data.load(opt.test_src , 1, source_dict)
    test_target = pre_data.load(opt.test_trg , 1, target_dict)

    train_batch_sampler = create_batch_sampler(train_source, train_target, opt.batch_size)
    valid_batch_sampler = create_batch_sampler(valid_source, valid_target, opt.valid_batch_size)
    
    #create dataset and dataloader
    train_data_set = MyDataset(train_source, train_target)
    valid_data_set = MyDataset(valid_source, valid_target)
    valid_iter = DataLoader(valid_data_set, batch_sampler=valid_batch_sampler, collate_fn=valid_data_set.collater)
    test_data_set = MyDataset(test_source, test_target)
    test_iter = DataLoader(test_data_set, batch_size=1, collate_fn=test_data_set.collater, shuffle=False)

    opt.valid_iter = valid_iter
    opt.test_iterator = test_iter
    opt.train_data_set = train_data_set
    opt.train_batch_sampler = train_batch_sampler
    opt.padding_idx = padding_idx
    opt.trg_sos_idx = trg_sos_idx
    opt.trg_eos_idx = trg_eos_idx
    opt.src_size = src_size
    opt.trg_size = trg_size
    opt.SrcDict = SrcDict
    opt.TrgDict = TrgDict
    
