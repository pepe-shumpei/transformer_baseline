import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from cupy.cuda import cudnn
from apex import amp

import time
import argparse
import os 
import random
from nltk.translate.bleu_score import corpus_bleu

from Models import Transformer
from utils import *
from preprocess import preprocess
from Translator import Translator

# cuDNNを使用しない  
seed = 88
cudnn.deterministic = True  
random.seed(seed)  
torch.manual_seed(seed)  
# cuda でのRNGを初期化  
torch.cuda.manual_seed_all(seed)  

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=40)
    parser.add_argument('-max_iteration', type=int, default=100000)
    #parser.add_argument('-batch_max_token', type=int, default=10000)
    parser.add_argument('-check_interval', type=int, default=1250)
    parser.add_argument('-batch_size',  type=int, default=50)
    parser.add_argument('-valid_batch_size',  type=int, default=50)
    parser.add_argument('-word_cut',  type=int, default=50000)
    parser.add_argument('-accumulation_steps',  type=int, default=1)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    #parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-level', type=str, default="O1")
    parser.add_argument('-cuda_n', type=str, default="0")

    parser.add_argument('-save', type=str, default=None)
    parser.add_argument('-mode', type=str, default="full")

    parser.add_argument('-train_src', type=str, default=None)
    parser.add_argument('-train_trg', type=str, default=None)
    parser.add_argument('-valid_src', type=str, default=None)
    parser.add_argument('-valid_trg', type=str, default=None)
    parser.add_argument('-test_src', type=str, default=None)
    parser.add_argument('-test_trg', type=str, default=None)

    opt = parser.parse_args()
    
    return opt

#lossを計算できるようにする。
def valid_epoch(model, valid_iterator, padding_idx, device, TrgDict):
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
            max_length = src.size(1) + 50
            seq = model(src, trg_input, src_mask, trg_mask, max_length, train=False)
            #
            sentence_to_list(gen_sentence_list, seq, TrgDict)
            sentence_to_list(ref_sentence_list, trg, TrgDict, ref=True)
        bleu_score = corpus_bleu(ref_sentence_list, gen_sentence_list)
        return bleu_score

def train(opt):
    #train and validation 
    iteration = 0
    train_loss = 0
    start_time = time.time()
    with open(opt.log, "a") as f:
        f.write("\n-----training-----\n")

    for epoch in range(opt.epoch):
        opt.model.train()
        random.shuffle(opt.train_batch_sampler) 
        train_iterator = DataLoader(opt.train_data_set, batch_sampler=opt.train_batch_sampler, collate_fn=opt.train_data_set.collater)

        for iters in train_iterator:

            src = iters[0].to(opt.device)
            trg = iters[1].to(opt.device)

            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt.device, opt.padding_idx)
            
            preds = opt.model(src, trg_input, src_mask, trg_mask, train=True)
            preds = preds.view(-1, preds.size(-1))
            ys = trg[:, 1:].contiguous().view(-1)

            loss = cal_loss(preds, ys, opt.padding_idx, smoothing=True)
            loss = loss/opt.accumulation_steps
            with amp.scale_loss(loss, opt.optimizer) as scaled_loss:
                scaled_loss.backward()
            train_loss += loss.item()

            if (iteration + 1) % opt.accumulation_steps == 0:
                opt.optimizer.step()
                opt.scheduler.step()
                opt.optimizer.zero_grad()
            
            iteration += 1

            #validation
            if iteration % opt.check_interval == 0:
                torch.cuda.empty_cache()
                valid_bleu = valid_epoch(opt.model, opt.valid_iter, opt.padding_idx, opt.device, opt.TrgDict)
                torch.cuda.empty_cache()

                train_loss = train_loss/opt.check_interval
                num_save = iteration // opt.check_interval
                end_time = time.time()
                with open(opt.log, "a") as f:
                    f.write("[Num Epoch %d] [Num Save %d] [Train Loss %d] [Valid BLEU %.3f] [TIME %.3f]\n" \
                            % (epoch+1, num_save, train_loss, valid_bleu*100, end_time - start_time))
                
                if num_save >= opt.max_iteration//opt.check_interval - 5 :
                    #save model
                    torch.save(opt.model.state_dict(), opt.save_model+ "/n_" + str(num_save) + ".model")

                opt.model.train()
                start_time = time.time()
                train_loss = 0

            #終了
            if iteration == opt.max_iteration:
                return

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

def test_epoch_beam(translator, test_iterator, SrcDict, TrgDict, device, load):
    
    PATH = "RESULT/" + load + "/output.txt"
    file = open(PATH , "w", encoding="utf8")
    for iters in test_iterator:
        src = iters[0].to(device)
        max_length = src.size(1) + 50
        if max_length > 400:
            max_length = 400
        sentence = translator.translate_sentence(src, max_length)
        generate_sentence(sentence, TrgDict, file) 
    file.close


def test(opt):
    torch.cuda.empty_cache()
    beamsize = 4
    max_seq_len = 410
    translator = Translator(opt.model, beamsize, max_seq_len, opt.padding_idx, opt.padding_idx, opt.trg_sos_idx, opt.trg_eos_idx)
    test_epoch_beam(translator, opt.test_iterator, opt.SrcDict, opt.TrgDict, opt.device, opt.save)

def main():


    opt = parse()
    model_path = "RESULT/"+ opt.save + "/model"
    model_ave_path = "RESULT/" + opt.save + "/model_ave"
    vocab_path = "RESULT/" + opt.save + "/vocab"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(model_ave_path, exist_ok=True)
    os.makedirs(vocab_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_n
    opt.device = torch.device("cuda:0")

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
                    %(opt.max_iteration, \
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
    opt.max_iteration = int(opt.max_iteration * opt.accumulation_steps)

    preprocess(opt)

    model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    opt.scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    opt.model, opt.optimizer = amp.initialize(model, optimizer, opt_level=opt.level)
        
    if opt.mode == "full" or opt.mode == "train":
        train(opt)

    if opt.mode == "full" or opt.mode == "test":
        load_point = opt.max_iteration//opt.check_interval
        average_model(load_point, opt)
        test(opt)

if __name__ == "__main__":
    main()