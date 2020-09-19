import torch
from torch.utils.data import DataLoader

import random
import numpy as np

from preprocessing import Preprocess
from dataset import MyDataset
from utils import create_batch_sampler

def preprocess(opt):
    
    source_vocab_path = "RESULT/" + opt.save + "/vocab/source_vocab"
    target_vocab_path = "RESULT/" + opt.save + "/vocab/target_vocab"

    SRC = Preprocess()
    TRG = Preprocess()

    train_source , valid_source, test_source = \
        SRC.load(train=opt.train_src,
                valid=opt.valid_src, 
                test = opt.test_src, 
                mode=1, 
                vocab_file=source_vocab_path)
    
    train_target , valid_target, test_target = \
        TRG.load(train=opt.train_trg,
                valid=opt.valid_trg, 
                test = opt.test_trg, 
                mode=1, 
                vocab_file=target_vocab_path)

    SrcDict = SRC.reverse_dict
    TrgDict = TRG.reverse_dict
    src_size = len(SRC.dict)
    trg_size = len(TRG.dict)
    padding_idx = SRC.dict["<pad>"]
    trg_sos_idx = TRG.dict["<sos>"]
    trg_eos_idx = TRG.dict["<eos>"]

    #create batch sampler
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
    
