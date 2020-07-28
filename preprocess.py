import torch
from preprocessing import *
from dataset import MyDataset


import random

def create_batch_sampler(src, trg):
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
        max_token = 5000
        while True:
            indices.append(sorted_indices[idx])
            num_trg_token += len(trg[sorted_indices[idx]])
            idx += 1
            if len(src) == idx:
                break
            if num_trg_token > max_token:
                break
        
        batch_indices.append(indices)

        if len(src) == idx:
            break

    return batch_indices


def preprocess(opt):
    
    """
    #ASPEC1.5M EN-JA SUBWORD
    train_src = "../ASPEC1.5M/train.en.16000"
    train_trg = "../ASPEC1.5M/train.ja.16000"
    valid_src = "../ASPEC1.5M/dev.en.16000"
    valid_trg = "../ASPEC1.5M/dev.ja.16000"
    test_src = "../ASPEC1.5M/test.en.16000"
    test_trg = "../ASPEC1.5M/test.ja.16000"
    """

    """
    #ASPEC1.5M EN-JA WORD
    train_src = "../ASPEC1.5M/train.en"
    train_trg = "../ASPEC1.5M/train.ja"
    valid_src = "../ASPEC1.5M/dev.en"
    valid_trg = "../ASPEC1.5M/dev.ja"
    test_src = "../ASPEC1.5M/test.en"
    test_trg = "../ASPEC1.5M/test.ja"
    """

    """
    #ASPEC1.5M JA-EN SUBWORD
    train_src = "../ASPEC1.5M/train.ja.16000"
    train_trg = "../ASPEC1.5M/train.en.16000"
    valid_src = "../ASPEC1.5M/dev.ja.16000"
    valid_trg = "../ASPEC1.5M/dev.en.16000"
    test_src = "../ASPEC1.5M/test.ja.16000"
    test_trg = "../ASPEC1.5M/test.en.16000"
    """

    """
    #ASPEC1.5M JA-EN WORD
    train_src = "../ASPEC1.5M/train.ja"
    train_trg = "../ASPEC1.5M/train.en"
    valid_src = "../ASPEC1.5M/dev.ja"
    valid_trg = "../ASPEC1.5M/dev.en"
    test_src = "../ASPEC1.5M/test.ja"
    test_trg = "../ASPEC1.5M/test.en"
    """

    """
    #ASPEC1.5M EN-JA SUBWORD1k→1k
    train_src = "../ASPEC1.5M/train.en.1000"
    train_trg = "../ASPEC1.5M/train.ja.1000"
    valid_src = "../ASPEC1.5M/dev.en.1000"
    valid_trg = "../ASPEC1.5M/dev.ja.1000"
    test_src = "../ASPEC1.5M/test.en.1000"
    test_trg = "../ASPEC1.5M/test.ja.1000"
    """

    """
    #ASPEC1.5M JA-EN SUBWORD1k→1k
    train_src = "../ASPEC1.5M/train.ja.1000"
    train_trg = "../ASPEC1.5M/train.en.1000"
    valid_src = "../ASPEC1.5M/dev.ja.1000"
    valid_trg = "../ASPEC1.5M/dev.en.1000"
    test_src = "../ASPEC1.5M/test.ja.1000"
    test_trg = "../ASPEC1.5M/test.en.1000"
    """
    
    """
    #ASPEC1.5M EN-JA SUBWORD1k→16k
    train_src = "../ASPEC1.5M/train.en.1000"
    train_trg = "../ASPEC1.5M/train.ja.16000"
    valid_src = "../ASPEC1.5M/dev.en.1000"
    valid_trg = "../ASPEC1.5M/dev.ja.16000"
    test_src = "../ASPEC1.5M/test.en.1000"
    test_trg = "../ASPEC1.5M/test.ja.16000"
    """

    #ASPEC1.5M JA-EN SUBWORD1k→16k
    train_src = "../ASPEC1.5M/train.ja.1000"
    train_trg = "../ASPEC1.5M/train.en.16000"
    valid_src = "../ASPEC1.5M/dev.ja.1000"
    valid_trg = "../ASPEC1.5M/dev.en.16000"
    test_src = "../ASPEC1.5M/test.ja.1000"
    test_trg = "../ASPEC1.5M/test.en.16000"

    source_vocab_path = "RESULT/" + opt.save + "/vocab/source_vocab"
    target_vocab_path = "RESULT/" + opt.save + "/vocab/target_vocab"

    #create vocab
    source_vocab = GetVocab(train_src)
    target_vocab = GetVocab(train_trg)
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

    train_source = pre_data.load(train_src , 1, source_dict)
    train_target = pre_data.load(train_trg , 1, target_dict)
    valid_source = pre_data.load(valid_src , 1, source_dict)
    valid_target = pre_data.load(valid_trg , 1, target_dict)
    test_source = pre_data.load(test_src , 1, source_dict)
    test_target = pre_data.load(test_trg , 1, target_dict)

    batch_sampler = create_batch_sampler(train_source, train_target)
    #random.shuffle(batch_sampler) #あとでこれは1epochごとにshuffleするようにする
    
    #create dataset and dataloader
    batch_size = 100
    train_data_set = MyDataset(train_source, train_target)
    #train_iter = DataLoader(train_data_set, batch_sampler=batch_sampler, collate_fn=train_data_set.collater)
    valid_data_set = MyDataset(valid_source, valid_target)
    valid_iter = DataLoader(valid_data_set, batch_size=batch_size, collate_fn=valid_data_set.collater, shuffle=False)
    test_data_set = MyDataset(test_source, test_target)
    test_iter = DataLoader(test_data_set, batch_size=1, collate_fn=valid_data_set.collater, shuffle=False)

    padding_idx = source_dict["<pad>"]
    trg_sos_idx = source_dict["<sos>"]
    trg_eos_idx = source_dict["<eos>"]
    

    #opt.train_iterator = train_iter
    opt.valid_iterator = valid_iter
    opt.test_iterator = test_iter
    opt.Dict = translate_dict
    opt.SrcDict = SrcDict
    opt.train_data_set = train_data_set
    opt.batch_sampler = batch_sampler
    opt.padding_idx = padding_idx
    opt.trg_sos_idx = trg_sos_idx
    opt.trg_eos_idx = trg_eos_idx
    opt.src_size = src_size
    opt.trg_size = trg_size
