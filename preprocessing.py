import os
import torch
import sys
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PADDING = 0
UNK = 1
START_DECODING = 2
STOP_DECODING = 3
import copy
class Preprocess():
    def __init__(self):
        self.init_dict = {"<pad>": PADDING ,"<unk>": UNK, "<sos>": START_DECODING, "<eos>": STOP_DECODING}

    def getVocab(self, vocab_file):
        return self.pushVocab(vocab_file)

    def pushVocab(self, vocab_file):
        vocab_dict = copy.copy(self.init_dict)
        with open(vocab_file) as f:
            for count, vocab in enumerate(f):
                vocab_dict[vocab.strip()] = len(vocab_dict)
                #if len(vocab_dict) >= 50000:
                    #break
        return vocab_dict

    def load(self, data_path, mode, vocab_dict):
        '''
            mode : 0 -> train_source / eval / decode
            mode : 1 -> train_target

        '''
        self.dict = vocab_dict
        with open(data_path) as f:
            tensor_data = [ self.ConvertTensor(doc, mode) for i, doc in enumerate(f)]
        return tensor_data

    def ConvertTensor(self, doc, mode):
        doc = self.replaceWord(doc)
        words = self.DocToWord(doc)
        if mode == 1:
            words = ["<sos>"] + words + ["<eos>"]
        words_id = self.SentenceToDictID(words)
        return words_id

    def replaceWord(self, doc):
        doc = doc.replace("<t>", "")
        doc = doc.replace("</t>", "")
        return doc

    def DocToWord(self, strs):
        return strs.strip().split(' ')

    def SentenceToDictID(self, sentence):
        slist = []
        for word in sentence:
            if word in self.dict:
                slist.append(self.dict[word])
            else:
                slist.append(UNK)
        return torch.tensor(slist)

def GetVocab(file_name, vocab_size=50000):
    vocab = {}
    count_vocab = {}
    with open(file_name) as lines:
        for line in lines:
            words = line.split()
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
                    count_vocab[word] = 1
                else:
                    count_vocab[word] += 1

    #出現回数で単語をカットする
    min_token_num = 1 
    flag = False
    while True:
        min_token_num += 1
        for word in count_vocab:
            if count_vocab[word] < min_token_num :
                if word in vocab:
                    del vocab[word]
            if len(vocab) < vocab_size:
                flag = True
                break
        if flag:
            break

    return vocab

