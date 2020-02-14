import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import json


class DataSplitter:

    def __init__(self, split_rate=0.8):
        self.split_rate = split_rate
        self.train_ids = None
        self.test_ids = None

    def split(self, infname: str, sep='\t', id_field='reaxys_id'):
        df = pd.read_csv(infname, sep=sep)
        unique_ids = df[id_field].unique()
        n_ids = unique_ids.shape[0]
        self.train_ids = np.random.choice(unique_ids,
                                          replace=False,
                                          size=int(self.split_rate*n_ids))
        self.test_ids = np.array([x for x in unique_ids if x not in self.train_ids])
        train = df[df[id_field].isin(self.train_ids)]
        test = df[df[id_field].isin(self.test_ids)]
        return train, test
        

class SmilesDataset(Dataset):

    def __init__(self, infname: str, sep='\t', fields_to_leave=['original', 'erroneous']):
        df = pd.read_csv(infname, sep=sep)
        df = df[fields_to_leave]
        # Tokenize
        self.token_map = self.get_token_map(df, fields_to_leave)
        self.num_to_token = {v : k for k, v in self.token_map.items()}
        df['erroneous'] = df['erroneous'].apply(lambda x: "BOS " + x + " EOS")
        df['original'] = df['original'].apply(lambda x: "BOS " + x + " EOS")
        self.inputs = [torch.LongTensor([self.token_map[t] for t in row.split()]) for row in df['erroneous']]
        self.labels = [torch.LongTensor([self.token_map[t] for t in row.split()]) for row in df['original']]
        self.input_lens = torch.LongTensor([len(x) for x in self.inputs])
        self.label_lens = torch.LongTensor([len(x) for x in self.labels])
        #max_inp_len = torch.max(self.input_lens).item()
        #max_label_len = torch.max(self.label_lens).item()
        #pad_len = max(max_inp_len, max_label_len)
        #self.inputs = torch.LongTensor([x + [0] * (pad_len - len(x)) for x in self.inputs])
        #self.labels = torch.LongTensor([x + [0] * (pad_len - len(x)) for x in self.labels])
        #assert self.inputs.shape[0] == self.labels.shape[0]
        #assert self.inputs.shape[1] == self.labels.shape[1]
        #assert self.input_lens.shape[0] == self.label_lens.shape[0]
        
    def save_token_map(self, out_fname: str):
        with open(out_fname, "w") as ostream:
            ostream.write(json.dumps(self.token_map))
        
    def get_token_map(self, df, fields_to_leave):
        token_map = {'PAD': 0, 'BOS': 1, 'EOS': 2}
        start_idx = 3
        token_set = set()
        for field in fields_to_leave:
            series = df[field]
            for entry in series:
                curr_set = set(entry.split())
                token_set = token_set.union(curr_set)
        for token in token_set:
            token_map[token] = len(token_map)
        return token_map
    
    @property
    def n_tokens(self):
        return len(self.token_map)

    def as_smiles(self, tokenized: list):
        return "".join([self.num_to_token[x] for x in tokenized])

    def __len__(self):
        #return self.inputs.shape[0]
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.input_lens[idx], self.label_lens[idx]
    
    @staticmethod
    def collate(data):
        inps, labs, in_lens, lab_lens = zip(*data)
        inps = pad_sequence(inps)
        labs = pad_sequence(labs)
        in_lens = torch.stack(in_lens)
        lab_lens = torch.stack(lab_lens)
        return inps, labs, in_lens, lab_lens
