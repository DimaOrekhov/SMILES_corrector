from tqdm import tqdm
from collections import namedtuple

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import *

from model import RNNVAE

# from model import PADDING_IDX
PADDING_IDX = 0


def collate(batch):
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    reactions, cores, rlens, clens = zip(*batch)
    return reactions, cores, rlens, clens


def kl_divergence(mu, log_sigma2):
    KL = 0.5 * torch.sum(torch.exp(log_sigma2) + mu ** 2 - log_sigma2 - 1)
    return KL


class VAETrainer:
    
    def __init__(self, data, logpath, logfreq, savepath, savefreq):
        self.data = data
        self.logpath = logpath
        self.logfreq = logfreq
        self.savepath = savepath # Configure unique filename?
        self.savefreq = savefreq
        
        self.kl_loss = kl_divergence
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
    
    def fit(self, hypers):
        # save hypers so you will know how the model was trained
        self.model = RNNVAE(**hypers.vae)
        loader = DataLoader(self.data,
                            batch_size=hypers.batch_size,
                            shuffle=True,
                            collate_fn=collate)
        self.n_batches = len(loader)
        self.opt = hypers.optim(self.model.parameters())
        kl_weight = 0 # Add scheduling scheme
        for i in tqdm(range(hypers.n_epochs)):
            self.r_ce_loss = 0
            self.r_kl_loss = 0
            self.r_total_loss = 0
            for j, (reacts, cores, rlens, clens) in enumerate(loader):
                rec, mu, log_sigma2 = self.model(reacts, cores, rlens, clens)
                target = pad_sequence(cores,
                                      batch_first=False,
                                      padding_value=PADDING_IDX).flatten()
                
                self.opt.zero_grad()
                # return reacts, cores, rlens, clens, rec
                rec_loss = self.ce_loss(rec.view(-1, rec.shape[-1]),
                                        target)
                kl_loss = self.kl_loss(mu, log_sigma2)
                loss_val = rec_loss +  kl_weight * kl_loss
                loss_val.backward()
                self.opt.step()
                
                self.log(i, j, rec_loss, kl_loss, loss_val)
    
    def log(self, i, j, rec_loss, kl_loss, loss_val):
        self.r_ce_loss += rec_loss.item()
        self.r_kl_loss += kl_loss.item()
        self.r_total_loss += loss_val.item()

        if (i * self.n_batches + j) % self.logfreq == 0:
            # Maybe percentages would be more convenient
            logstr = 'Epoch {}, batch {}\ttotal_loss = {}\tce_loss = {}\tkl_loss = {}\n'.\
                format(i, j, self.r_total_loss, self.r_ce_loss, self.r_kl_loss)
            with open(self.logpath, 'a') as fstream:
                fstream.write(logstr)
            self.r_ce_loss = 0
            self.r_kl_loss = 0
            self.r_total_loss = 0
            
        if (i * self.n_batches + j) % self.savefreq == 0:
            pass # SAVE MODEL HERE


# Add optim_params field later
Hypers = namedtuple('HyperParameters', ['vae', 'optim', 'n_epochs', 'batch_size'])
class HyperParameters(Hypers):
    
    def save(self, savepath):
        pass