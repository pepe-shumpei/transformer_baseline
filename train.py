import time
import argparse
import os 
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from cupy.cuda import cudnn
from apex import amp
from nltk.translate.bleu_score import corpus_bleu

from model.Models import Transformer
from preprocessing import Preprocess
from dataset import MyDataset
from trainer import Trainer
from model.Translator import Translator
from utils import *
from lr_scheduler import lr_schedule


# cuDNNを使用しない  
seed = 88
cudnn.deterministic = True  
random.seed(seed)  
torch.manual_seed(seed)  
# cuda でのRNGを初期化  
torch.cuda.manual_seed_all(seed)  

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--max_steps', type=int, default=100000)
    #parser.add_argument('--batch_max_token', type=int, default=10000)
    parser.add_argument('--check_interval', type=int, default=1250)
    parser.add_argument('--batch_size',  type=int, default=50)
    parser.add_argument('--valid_batch_size',  type=int, default=50)
    parser.add_argument('--word_cut',  type=int, default=50000)
    parser.add_argument('--accumulation_steps',  type=int, default=1)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)

    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    #parser.add_argument('--warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--level', type=str, default="O1")
    parser.add_argument('--cuda_n', type=str, default="0")

    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--mode', type=str, default="full")

    parser.add_argument('--train_src', type=str, default=None)
    parser.add_argument('--train_trg', type=str, default=None)
    parser.add_argument('--valid_src', type=str, default=None)
    parser.add_argument('--valid_trg', type=str, default=None)
    parser.add_argument('--test_src', type=str, default=None)
    parser.add_argument('--test_trg', type=str, default=None)

    opt = parser.parse_args()
    
    return opt

def main():
    
    opt = parse()
    model_path = "RESULT/"+ opt.save + "/model"
    vocab_path = "RESULT/" + opt.save + "/vocab"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(vocab_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_n
    device = torch.device("cuda:0")

    opt.log = "RESULT/" + opt.save + "/log"
    opt.save_model = model_path

    # write a setting 
    with open(opt.log, "a") as f:
        f.write("-----setting-----\n")
        f.write("MAX ITERATION : %d \
                \nCHECK INTERVAL : %d \
                \nBATCH SIZE : %d \
                \nACCUMULATION STEPS : %d \
                \nWORD CUT : %d \
                \nD_MODEL : %d \
                \nN_LAYERS : %d \
                \nN_HEAD : %d \
                \nDROPOUT : %.1f \
                \nMODE : %s \
                \nSAVE_MODEL : %s \
                \nLOG_PATH : %s \
                \nGPU NAME: %s \
                \nGPU NUM %s \
                \nDATASET : \n%s\n%s\n%s\n%s\n%s\n%s" \
                    %(opt.max_steps, \
                    opt.check_interval, \
                    opt.batch_size, \
                    opt.accumulation_steps, \
                    opt.word_cut, \
                    opt.d_model, \
                    opt.n_layers, \
                    opt.n_head, \
                    opt.dropout, \
                    opt.mode, \
                    opt.save, \
                    opt.log, \
                    torch.cuda.get_device_name(), \
                    opt.cuda_n, \
                    opt.train_src, \
                    opt.train_trg, \
                    opt.valid_src, \
                    opt.valid_trg, \
                    opt.test_src, \
                    opt.test_trg))

    #gradient accumulation
    opt.batch_size = int(opt.batch_size/opt.accumulation_steps)
    opt.check_interval = int(opt.check_interval * opt.accumulation_steps)
    opt.max_steps = int(opt.max_steps * opt.accumulation_steps)

    #前処理
    source_vocab_path = "RESULT/" + opt.save + "/vocab/source_vocab"
    target_vocab_path = "RESULT/" + opt.save + "/vocab/target_vocab"

    SRC = Preprocess()
    TRG = Preprocess()

    train_source, valid_source, test_source = \
        SRC.load(train=opt.train_src,
                valid=opt.valid_src, 
                test = opt.test_src, 
                mode=1, 
                vocab_file=source_vocab_path)
    
    train_target, valid_target, test_target = \
        TRG.load(train=opt.train_trg,
                valid=opt.valid_trg, 
                test = opt.test_trg, 
                mode=1, 
                vocab_file=target_vocab_path)

    #SrcDict = SRC.reverse_dict
    TrgDict = TRG.reverse_dict
    src_size = len(SRC.dict)
    trg_size = len(TRG.dict)
    pad_idx = SRC.dict["<pad>"]
    trg_sos_idx = TRG.dict["<sos>"]
    trg_eos_idx = TRG.dict["<eos>"]

    #create batch sampler
    train_batch_sampler = create_batch_sampler(train_source, train_target, opt.batch_size)
    valid_batch_sampler = create_batch_sampler(valid_source, valid_target, opt.valid_batch_size)
    
    #create dataset and dataloader
    train_data_set = MyDataset(train_source, train_target)
    valid_data_set = MyDataset(valid_source, valid_target)
    valid_data_loader = DataLoader(valid_data_set, batch_sampler=valid_batch_sampler, collate_fn=valid_data_set.collater)
    test_data_set = MyDataset(test_source, test_target)
    test_data_loader = DataLoader(test_data_set, batch_size=1, collate_fn=test_data_set.collater, shuffle=False)

    model = Transformer(src_size, trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt.level)

    trainer = Trainer(
        model = model,
        loss = None,
        optimizer = optimizer,
        train_data_set = train_data_set,
        train_batch_sampler = train_batch_sampler,
        valid_data_loader = valid_data_loader,
        lr_scheduler = scheduler,
        device = device,
        TrgDict = TrgDict,
        pad_idx = pad_idx
        )
        
    if opt.mode == "full" or opt.mode == "train":
        trainer.train(opt.epoch, opt)

    """
    if opt.mode == "full" or opt.mode == "test":
        load_point = opt.max_step//opt.check_interval
        average_model(load_point, opt)
        test(opt)
    """

if __name__ == "__main__":
    main()