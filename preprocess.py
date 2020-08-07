import torch
from preprocessing import *
from dataset import MyDataset


import random

def create_batch_sampler(src, trg, max_token):
    indices = torch.arange(len(src)).tolist()
    random.shuffle(indices)
    indices = sorted(indices, key=lambda idx: len(src[idx]))
    sorted_indices = sorted(indices, key=lambda idx: len(trg[idx]))

    batch_indices = []
    idx = 0
    while True:
        indices = []
        num_src_token = 0
        num_trg_token = 0
        src_max_length = 0 
        #max_token = 5000
        
        while True:
            indices.append(sorted_indices[idx])
            num_trg_token += len(trg[sorted_indices[idx]])
            num_src_token += len(src[sorted_indices[idx]])
            if src_max_length < len(src[sorted_indices[idx]]):
                src_max_length = len(src[sorted_indices[idx]])

            idx += 1
            if len(src) == idx:
                break
            if num_trg_token > max_token:
                break
            if len(indices)*src_max_length > max_token:
                break
        
        batch_indices.append(indices)

        if len(src) == idx:
            break

    return batch_indices


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

    translate_dict = {}
    for key, value in target_dict.items():
        translate_dict[value] = key

    SrcDict = {}
    for key, value in source_dict.items():
        SrcDict[value] = key

    src_size = len(source_dict)
    trg_size = len(target_dict)

    train_source = pre_data.load(opt.train_src , 1, source_dict)
    train_target = pre_data.load(opt.train_trg , 1, target_dict)
    valid_source = pre_data.load(opt.valid_src , 1, source_dict)
    valid_target = pre_data.load(opt.valid_trg , 1, target_dict)
    test_source = pre_data.load(opt.test_src , 1, source_dict)
    test_target = pre_data.load(opt.test_trg , 1, target_dict)

    train_batch_sampler = create_batch_sampler(train_source, train_target, opt.batch_max_token)
    valid_batch_sampler = create_batch_sampler(valid_source, valid_target, opt.batch_max_token)
    
    #create dataset and dataloader
    batch_size = 100
    train_data_set = MyDataset(train_source, train_target)
    #train_iter = DataLoader(train_data_set, batch_sampler=batch_sampler, collate_fn=train_data_set.collater)
    valid_data_set = MyDataset(valid_source, valid_target)
    #valid_iter = DataLoader(valid_data_set, batch_size=batch_size, collate_fn=valid_data_set.collater, shuffle=False)
    test_data_set = MyDataset(test_source, test_target)
    test_iter = DataLoader(test_data_set, batch_size=1, collate_fn=valid_data_set.collater, shuffle=False)

    padding_idx = source_dict["<pad>"]
    trg_sos_idx = source_dict["<sos>"]
    trg_eos_idx = source_dict["<eos>"]
    

    #opt.train_iterator = train_iter
    #opt.valid_iterator = valid_iter
    opt.test_iterator = test_iter
    opt.Dict = translate_dict
    opt.SrcDict = SrcDict
    opt.train_data_set = train_data_set
    opt.train_batch_sampler = train_batch_sampler
    opt.valid_data_set = valid_data_set
    opt.valid_batch_sampler = valid_batch_sampler
    opt.padding_idx = padding_idx
    opt.trg_sos_idx = trg_sos_idx
    opt.trg_eos_idx = trg_eos_idx
    opt.src_size = src_size
    opt.trg_size = trg_size
