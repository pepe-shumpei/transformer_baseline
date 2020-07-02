import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext import data


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

def save_model(model, epoch, path):
    PATH = path + "/model_save" + str(epoch)
    #PATH = "./trained_model/model_save" + str(epoch+1)
    torch.save(model.state_dict(), PATH)

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
def preprocess(opt):

    #データの読み込み、前処理
    SRC = torchtext.data.Field(init_token = "<sos>", eos_token="<eos>", lower=True)
    TRG = torchtext.data.Field(init_token = "<sos>", eos_token="<eos>") 
    train_ds, valid_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='~/work/Corpus/ASPEC-JE/corpus100k', train='train.tsv',
        validation='dev.tsv',test='test.tsv', format='tsv',
        fields=[('src', SRC ), ('trg', TRG)])

    SRC.build_vocab(train_ds, min_freq = 2)
    TRG.build_vocab(train_ds, min_freq = 2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_ds, valid_ds, test_ds),
        batch_sizes= (opt.train_batch_size, opt.train_batch_size, opt.test_batch_size),
        sort=False,
        device = opt.device)

    padding_idx = TRG.vocab.stoi["<pad>"]
    trg_sos_idx = TRG.vocab.stoi["<sos>"]
    trg_eos_idx = TRG.vocab.stoi["<eos>"]
    src_size = len(SRC.vocab)
    trg_size = len(TRG.vocab)
    

    opt.train_iterator = train_iterator
    opt.valid_iterator = valid_iterator
    opt.test_iterator = test_iterator
    opt.SRC = SRC
    opt.TRG = TRG
    opt.padding_idx = padding_idx
    opt.trg_sos_idx = trg_sos_idx
    opt.trg_eos_idx = trg_eos_idx
    opt.src_size = src_size
    opt.trg_size = trg_size
"""
