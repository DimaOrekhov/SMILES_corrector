from torch import nn
from torch.nn.utils.rnn import *

PADDING_IDX = 0

class RNNVAE(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 h_dim,
                 n_layers,
                 latent_dim,
                 bidir=False,
                 pdrop=0):
        super().__init__()
        self.encoder = Encoder(vocab_size,
                               emb_dim,
                               h_dim,
                               n_layers,
                               latent_dim,
                               bidir,
                               pdrop)
        self.decoder = Decoder(vocab_size,
                               emb_dim,
                               h_dim,
                               n_layers,
                               latent_dim,
                               bidir,
                               pdrop)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, y, xlens, ylens):
        mu, logsigma2 = self.encoder(x, xlens)
        sample = mu + torch.exp(logsigma2 / 2) + torch.randn_like(mu)
        rec = self.decoder(sample, ylens)
        #return sample
        return rec, mu, logsigma2

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, h_dim, n_layers, latent_dim, bidir, pdrop):
        super().__init__()
        self.n_layers = n_layers
        self.bidir = bidir
        self.h_dim = h_dim
        
        self.emb = nn.Embedding(vocab_size, emb_dim, PADDING_IDX)
        # Add dropout if needed
        self.rnn = nn.LSTM(emb_dim,
                           h_dim,
                           n_layers,
                           bidirectional=bidir,
                           dropout=pdrop)
        self.hidden = None
        self.mu = nn.Linear(h_dim * (2 if bidir else 1), latent_dim)
        self.logsigma2 = nn.Linear(h_dim * (2 if bidir else 1), latent_dim)
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.n_layers * (2 if self.bidir else 1),
                                   batch_size,
                                   self.h_dim,
                                   device=self.device),
                       torch.zeros(self.n_layers * (2 if self.bidir else 1),
                                   batch_size,
                                   self.h_dim,
                                   device=self.device))
        
    def forward(self, x, lens):
        x = pad_sequence(x, batch_first=False, padding_value=PADDING_IDX)
        self.init_hidden(x.shape[1])
        x = self.emb(x)
        x = pack_padded_sequence(x, lens, enforce_sorted=True)
        x, (h, c) = self.rnn(x, self.hidden)
        h = h[-(1 + self.bidir):]
        x = torch.cat((h[0], h[1]), -1)
        mu = self.mu(x)
        logsigma2 = self.logsigma2(x)
        return mu, logsigma2

    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, h_dim, n_layers, latent_dim, bidir, pdrop):
        super().__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.bidir = bidir

        self.fc = nn.Linear(latent_dim, h_dim * (2 if bidir else 1))
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(h_dim * (2 if bidir else 1),
                           h_dim,
                           n_layers,
                           bidirectional=bidir,
                           dropout=pdrop)
        self.fc_out = nn.Linear(h_dim * (2 if bidir else 1), vocab_size)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.n_layers * (2 if self.bidir else 1),
                                   batch_size,
                                   self.h_dim,
                                   device=self.device),
                       torch.zeros(self.n_layers * (2 if self.bidir else 1),
                                   batch_size,
                                   self.h_dim,
                                   device=self.device))

    def forward(self, z, ylens):
        self.init_hidden(z.shape[0])
        z = z.unsqueeze(0).repeat(max(ylens), 1, 1)
        z = self.relu(self.fc(z))
        z = pack_padded_sequence(z, ylens, enforce_sorted=False)
        rec, _ = self.rnn(z)
        rec, _ = pad_packed_sequence(rec, batch_first=False)
        rec = self.fc_out(rec)
        return rec