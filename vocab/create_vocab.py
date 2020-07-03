source_path = "../../train_data/train.en.16000"
target_path = "../../train_data/train.ja.16000"


def GetVocab(file_name, min_token_num=0):
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
    for word in count_vocab:
        if count_vocab[word] < min_token_num :
            del vocab[word]

    return vocab

source_vocab = GetVocab(source_path)
target_vocab = GetVocab(target_path)

with open("source_vocab", "w") as f:
    for key in source_vocab.keys():
        f.write(key + "\n")

with open("target_vocab", "w") as f:
    for key in target_vocab.keys():
        f.write(key + "\n")

