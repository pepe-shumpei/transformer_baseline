import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import time
import argparse
from nltk.translate.bleu_score import corpus_bleu

from Models import Transformer
from utils import *
from Translator import Translator
from preprocess import preprocess

def test_epoch(model, test_iterator, device, Dict, load, pad_idx):
    #generator
    model.eval()
    with torch.no_grad():
        #PATH = "translate_file/" + output_file_path
        PATH = "RESULT/" + load + "/output.txt"
        file = open(PATH , "w", encoding="utf8")
        gen_sentence_list = []
        ref_sentence_list = []
        for i , batch in enumerate(test_iterator):
            print(i)
            src = batch.src.permute(1,0)
            trg = batch.trg.permute(1,0)
            trg_input = trg[:, :-1] 
            src_mask, trg_mask = create_masks(src, trg_input, device, pad_idx)
            seq = model(src, trg_input, src_mask, trg_mask, train=False)
            output_file(seq, Dict, file)
            #
            sentence_to_list(gen_sentence_list, seq, Dict)
            sentence_to_list(ref_sentence_list, trg, Dict, ref=True)
        bleu_score = corpus_bleu(ref_sentence_list, gen_sentence_list)
        print("BLEU SCORE : ", bleu_score* 100)
        file.close
        print()

def test_epoch_beam(translator, test_iterator, Dict):
    
    file = open("output.txt", "w", encoding="utf8")
    for i , batch in enumerate(test_iterator):
        src = batch.src.permute(1,0)
        sentence = translator.translate_sentence(src)
        generate_sentence(sentence, Dict, file) 
        print(i)
    file.close

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-train_b', '--train_batch_size', type=int, default=100)
    parser.add_argument('-test_b', '--test_batch_size', type=int, default=100)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-dropout', type=float, default=0.1)

    #parser.add_argument('-log', default=None)
    #parser.add_argument('-load_model', type=str, default=None)
    #parser.add_argument('-output_file', type=str, default="output.txt")
    parser.add_argument('-load', type=str, default=None)

    opt = parser.parse_args()
    opt.device = torch.device("cuda:0")
    preprocess(opt)

    model = Transformer(opt.src_size, opt.trg_size, opt.d_model, opt.n_layers, opt.n_head, opt.dropout).to(opt.device)

    beam = False

    #PATH = "trained_model/" + opt.load_model
    PATH = "RESULT/" + opt.load + "/model_ave/best.model"
    model.load_state_dict(torch.load(PATH))
    
 
    if not beam:
        test_epoch(model, opt.test_iterator, opt.device, opt.Dict, opt.load, opt.padding_idx)

    else:
        beamsize = 4
        max_seq_len = 50
        translator = Translator(model, beamsize, max_seq_len, opt.padding_idx, opt.padding_idx, opt.trg_sos_idx, opt.trg_eos_idx)
        test_epoch_beam(translator, opt.test_iterator, opt.Dict, opt.padding_idx)

if __name__ == "__main__":
    main()


