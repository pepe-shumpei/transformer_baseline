import time
import argparse
import os 
import random

import torch
from torch.utils.data import DataLoader
from apex import amp
from nltk.translate.bleu_score import corpus_bleu

from model.Models import Transformer
from utils import *
from loss import cal_loss
from model.Translator import Translator


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_data_set,
        train_batch_sampler,
        valid_data_loader,
        lr_scheduler,
        device,
        TrgDict,
        pad_idx
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_data_set = train_data_set
        self.train_batch_sampler = train_batch_sampler
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.TrgDict = TrgDict
        self.pad_idx = pad_idx

    def train(self, n_epoch, opt):
        #train and validation 
        steps = 0
        train_loss = 0
        start_time = time.time()
        with open(opt.log, "a") as f:
            f.write("\n-----training-----\n")

        for epoch in range(n_epoch):
            self.model.train()
            random.shuffle(self.train_batch_sampler)
            train_data_loader = DataLoader(self.train_data_set, batch_sampler=self.train_batch_sampler, collate_fn=self.train_data_set.collater)

            for iters in train_data_loader:

                src = iters[0].to(self.device)
                trg = iters[1].to(self.device)

                trg_input = trg[:, :-1]
                src_mask, trg_mask = create_masks(src, trg_input, self.device, self.pad_idx)
                
                preds = self.model(src, trg_input, src_mask, trg_mask, train=True)
                preds = preds.view(-1, preds.size(-1))
                ys = trg[:, 1:].contiguous().view(-1)

                loss = cal_loss(preds, ys, self.pad_idx, smoothing=True)
                loss = loss/opt.accumulation_steps
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                train_loss += loss.item()

                if (steps + 1) % opt.accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                steps += 1
                #validation
                if steps % opt.check_interval == 0:
                    torch.cuda.empty_cache()
                    valid_bleu = self._valid_epoch()
                    torch.cuda.empty_cache()

                    train_loss = train_loss/opt.check_interval
                    num_save = steps // opt.check_interval
                    end_time = time.time()
                    with open(opt.log, "a") as f:
                        f.write("[Num Epoch %d] [Num Save %d] [Train Loss %.5f] [Valid BLEU %.3f] [TIME %.3f]\n" \
                                % (epoch+1, num_save, train_loss, valid_bleu*100, end_time - start_time))
                    
                    if num_save >= opt.max_steps//opt.check_interval - 5 :
                        #save model
                        torch.save(self.model.state_dict(), opt.save_model+ "/n_" + str(num_save) + ".model")

                    self.model.train()
                    start_time = time.time()
                    train_loss = 0

                #終了
                if steps == opt.max_steps:
                    return

    def _valid_epoch(self):
        self.model.eval()
        gen_sentence_list = []
        ref_sentence_list = []
        with torch.no_grad():
            for iters in self.valid_data_loader:
                src = iters[0].to(self.device)
                trg = iters[1].to(self.device)
                trg_input = trg[:, :-1] 
                src_mask, trg_mask = create_masks(src, trg_input, self.device, self.pad_idx)
                max_length = src.size(1) + 50
                seq = self.model(src, trg_input, src_mask, trg_mask, max_length, train=False)
                #
                sentence_to_list(gen_sentence_list, seq, self.TrgDict)
                sentence_to_list(ref_sentence_list, trg, self.TrgDict, ref=True)
            bleu_score = corpus_bleu(ref_sentence_list, gen_sentence_list)
        return bleu_score