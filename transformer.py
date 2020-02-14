import math
from torch import nn
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    
    def __init__(self, n_emb, d_model, n_heads, h_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_emb, d_model, padding_idx=0)
        self.positional = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, h_dim),
                                             n_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_heads, h_dim),
                                             n_layers)
        self.out_linear = nn.Linear(d_model, n_emb)

    @staticmethod
    def get_square_subsequent_mask(size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, max_iter=200, bos_idx=1):
        # Work in progress
        x = self.embedding(x)
        x = self.positional(x)
        memory = self.encoder(x)
        curr_seq = torch.zeros(max_iter, x.shape[1], dtype=torch.long)
        curr_seq[0] = torch.full((x.shape[1], ), bos_idx)
        logits = []
        for i in range(max_iter):
            with torch.no_grad():
                x = self.embedding(curr_seq[:i+1])
            x = self.decoder(x, memory)
            x = self.out_linear(x)
            curr_seq[i+1] = torch.argmax(x[-1], dim=-1) #?
            logits.append(x[-1])
        return x
