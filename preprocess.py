import torch
from preprocessing import *
from dataset import MyDataset

def preprocess(opt):
    
    train_en = "../train_data/train.en.16000"
    train_ja = "../train_data/train.ja.16000"
    valid_en = "../train_data/dev.en.16000"
    valid_ja = "../train_data/dev.ja.16000"
    test_en = "../train_data/test.en.16000"
    test_ja = "../train_data/test.ja.16000"


    source_vocab = "vocab/source_vocab"
    target_vocab = "vocab/target_vocab"

    pre_data = Preprocess()
    source_dict = pre_data.getVocab(source_vocab)
    target_dict = pre_data.getVocab(target_vocab)

    translate_dict = {}
    for key, value in target_dict.items():
        translate_dict[value] = key

    src_size = len(source_dict)
    trg_size = len(target_dict)

    train_source = pre_data.load(train_en , 1, source_dict)
    train_target = pre_data.load(train_ja , 1, target_dict)
    valid_source = pre_data.load(valid_en , 1, source_dict)
    valid_target = pre_data.load(valid_ja , 1, target_dict)
    test_source = pre_data.load(test_en , 1, source_dict)
    test_target = pre_data.load(test_ja , 1, target_dict)

    
    #create dataset and dataloader
    batch_size = 100
    train_data_set = MyDataset(train_source, train_target)
    train_iter = DataLoader(train_data_set, batch_size=batch_size, collate_fn=train_data_set.collater, shuffle=True)
    valid_data_set = MyDataset(valid_source, valid_target)
    valid_iter = DataLoader(valid_data_set, batch_size=batch_size, collate_fn=valid_data_set.collater, shuffle=False)
    test_data_set = MyDataset(test_source, test_target)
    test_iter = DataLoader(test_data_set, batch_size=1, collate_fn=valid_data_set.collater, shuffle=False)

    padding_idx = source_dict["<pad>"]
    trg_sos_idx = source_dict["<sos>"]
    trg_eos_idx = source_dict["<eos>"]
    

    opt.train_iterator = train_iter
    opt.valid_iterator = valid_iter
    opt.test_iterator = test_iter
    #opt.SRC = SRC
    #opt.TRG = TRG
    opt.Dict = translate_dict
    opt.padding_idx = padding_idx
    opt.trg_sos_idx = trg_sos_idx
    opt.trg_eos_idx = trg_eos_idx
    opt.src_size = src_size
    opt.trg_size = trg_size
