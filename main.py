import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from apex import amp

import time
import argparse
import os 
import random
from nltk.translate.bleu_score import corpus_bleu

from Models import Transformer
from utils import *
from preprocess import preprocess

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-train_b', '--train_batch_size', type=int, default=100)
    parser.add_argument('-test_b', '--test_batch_size', type=int, default=1)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    #parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-level', type=str, default="O1")
    parser.add_argument('-cuda_n', type=str, default="0")

    #parser.add_argument('-log', default=None)
    #parser.add_argument('-save_model', type=str, default=None)
    parser.add_argument('-save', type=str, default=None)

    opt = parser.parse_args()
    
    return opt

def train_epoch(model, optimizer, scheduler, train_data_set, batch_sampler, padding_idx, device):
    model.train()
    epoch_loss = 0
    random.shuffle(batch_sampler) #あとでこれは1epochごとにshuffleするようにする
    train_iterator = DataLoader(train_data_set, batch_sampler=batch_sampler, collate_fn=train_data_set.collater)

    for iters in train_iterator:
        src = iters[0].to(device)
        trg = iters[1].to(device)

        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, device, padding_idx)
        optimizer.zero_grad()
        preds = model(src, trg_input, src_mask, trg_mask, train=True)
        preds = preds.view(-1, preds.size(-1))
        ys = trg[:, 1:].contiguous().view(-1)

        #loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=padding_idx)
        loss = cal_loss(preds, ys, padding_idx, smoothing=True)

        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss/len(train_iterator)


"""
def valid_epoch(model, optimizer, scheduler, valid_iterator, padding_idx, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_iterator):
            src = batch.src.permute(1,0).to(device)
            trg = batch.trg.permute(1,0).to(device)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, device)
            preds = model(src, trg_input, src_mask, trg_mask, train=True)
            preds = preds.view(-1, preds.size(-1))
            ys = trg[:, 1:].contiguous().view(-1)
            #loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=padding_idx)
            loss = cal_loss(preds, ys, padding_idx, smoothing=True)
            epoch_loss += loss.item()
        return epoch_loss/len(valid_iterator)

"""

#lossを計算できるようにする。
def valid_epoch(model, valid_iterator, padding_idx, device, Dict):
    #generator
    model.eval()
    gen_sentence_list = []
    ref_sentence_list = []
    with torch.no_grad():
        for iters in valid_iterator:
            src = iters[0].to(device)
            trg = iters[1].to(device)
            trg_input = trg[:, :-1] 
            src_mask, trg_mask = create_masks(src, trg_input, device, padding_idx)
            seq = model(src, trg_input, src_mask, trg_mask, train=False)
            #
            sentence_to_list(gen_sentence_list, seq, Dict)
            sentence_to_list(ref_sentence_list, trg, Dict, ref=True)
        bleu_score = corpus_bleu(ref_sentence_list, gen_sentence_list)
        return bleu_score


def train(opt):
    #train and validation 
    #print("---training---")

    with open(opt.log, "a") as f:
        f.write("\n-----training-----\n")

    for epoch in range(opt.epoch):

        start_time = time.time()

        train_loss = train_epoch(opt.model, opt.optimizer, opt.scheduler, opt.train_data_set, opt.batch_sampler, opt.padding_idx, opt.device)
        torch.cuda.empty_cache()
        valid_bleu = valid_epoch(opt.model, opt.valid_iterator, opt.padding_idx, opt.device, opt.Dict)
        torch.cuda.empty_cache()

        end_time = time.time()

        with open(opt.log, "a") as f:
            f.write("[Epoch %d] [Train Loss %d] [Valid BLEU %.3f] [TIME %.3f]\n" % (epoch+1, train_loss, valid_bleu*100, end_time - start_time))

        save_model(opt.model, epoch+1, opt.save_model)
        

def load_model(model_num, opt):
    model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(opt.device)
    model.load_state_dict(torch.load(opt.save_model + "/model_save" + str(model_num)))
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

def checkpoint_averaging(opt):

    with open(opt.log, "a") as f:
        f.write("\n-----checkpoint averaging-----\n")

    best_bleu=-1
    for epoch in range(5, opt.epoch+1):
        average_model(epoch, opt)
        valid_bleu = valid_epoch(opt.model, opt.valid_iterator, opt.padding_idx, opt.device, opt.Dict)

        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            #save_model(opt.model, epoch, opt.save_model+ "_ave")
            torch.save(opt.model.state_dict(), opt.save_model+"_ave/best.model")
            
       
        #print("[Epoch %d] [Valid BLEU %.3f]" &(epoch, valid_bleu))
        with open(opt.log, "a") as f:
            f.write("[Epoch %d] [Valid BLEU %.3f]\n" %(epoch, valid_bleu*100))
        

def main():


    opt = parse()
    model_path = "RESULT/"+ opt.save + "/model"
    model_ave_path = "RESULT/" + opt.save + "/model_ave"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(model_ave_path, exist_ok=True)

    torch.cuda.set_device(torch.device('cuda:' + opt.cuda_n))
    opt.device = torch.device("cuda:" + opt.cuda_n)

    opt.log = "RESULT/" + opt.save + "/log"
    opt.save_model = model_path

    preprocess(opt)

    #model, optimizer, scheduler 
    model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(opt.device)
    #model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    opt.scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    opt.model, opt.optimizer = amp.initialize(model, optimizer, opt_level=opt.level)
    
    # write a setting 
    with open(opt.log, "a") as f:
        f.write("-----setting-----\n")
        f.write("N_EPOCH : %d\nD_MODEL : %d\nN_LAYERS : %d\nN_HEAD : %d\nBATCH SIZE : %d\nDROPOUT : %.1f\nSAVE_MODEL : %s\nLOG_PATH : %s\n"
                %(opt.epoch, opt.d_model, opt.n_layers, opt.n_head, opt.train_batch_size, opt.dropout, opt.save, opt.log))
        

    train(opt)
    checkpoint_averaging(opt)

if __name__ == "__main__":
    main()
