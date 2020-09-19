import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.Layers import EncoderLayer, DecoderLayer

import copy
import math

from utils import no_peak_mask

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 300, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        x = x + pe
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab, bias=False)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        #重み共有
        #self.x_logit_scale = (d_model ** -0.5)
        self.out.weight = self.decoder.embed.weight

    def forward(self, src, trg, src_mask, trg_mask, max_length=None, train=True):
        if train:
            e_outputs = self.encoder(src, src_mask)
            d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
            #output = self.out(d_output) * self.x_logit_scale
            output = self.out(d_output) 
            return output

        else:
            e_outputs = self.encoder(src, src_mask)
            out = trg[:,:1]
            for i in range(1, max_length):
                trg_mask = no_peak_mask(i).to(out.device)
                de_out = self.decoder(out , e_outputs, src_mask, trg_mask)          
                de_out = self.out(de_out)
                de_out = torch.argmax(de_out.float(), dim=2)
                out = torch.cat((out, de_out[:,-1].unsqueeze(1)),1)
            return out.squeeze(0)
