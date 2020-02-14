import torch
import json
from torch.utils.data import Dataset


SPECIALS = ['pad', 'eos', 'bos', 'unk']


def get_reactions(json_path: str)->list:
    reactions = []
    with open(json_path) as fsream:
        for line in fsream:
            reactions.append(json.loads(line))
    return reactions


def get_reaction_str(reaction: dict)->str:
    result = "bos " + reaction['reactants'][0]
    if len(reaction['reactants']) == 2:
        result += " . " + reaction['reactants'][1]
    result += " >> " + reaction['products'][0] + " eos"
    return result


def get_vocab_from_tokenized(reactions: list):
    from collections import Counter
    counter = Counter()
    for react in reactions:
        for char in react['reaction_core'].split():
            counter[char] += 1
        for reag in react['reactants']:
            for char in reag.split():
                counter[char] += 1
        for char in react['products'][0].split():
            counter[char] += 1
    return counter


class ReactionsDataset(Dataset):
    
    def __init__(self, file: str):
        reactions = get_reactions(file)
        self._init_code(reactions)
        self._encode(reactions)

    def _init_code(self, reactions: list):
        vocab = get_vocab_from_tokenized(reactions)
        self.code = {k: (i+len(SPECIALS)) for i, (k, v) in enumerate(vocab.items())}
        for i, c in enumerate(SPECIALS):
            self.code[c] = i
        assert('.' in self.code)
                    
    def _encode(self, reactions: list):
        self.reactions = []
        self.cores = []
        self.rlens = []
        self.clens = []
        for react in reactions:
            tmp_react = [self.code[c] for c in get_reaction_str(react).split()]
            # Возможно к стринге реакшн кора тоже стоит добавить токен eos
            tmp_core = [self.code[c] for c in react['reaction_core'].split()]

            self.reactions.append(torch.LongTensor(tmp_react))
            self.rlens.append(len(tmp_react))
            self.cores.append(torch.LongTensor(tmp_core))
            self.clens.append(len(tmp_core))
        assert(len(self.cores) == len(self.reactions))
        
    def __len__(self):
        return len(self.reactions)
    
    def __getitem__(self, idx: int):
        return (self.reactions[idx],
                self.cores[idx],
                self.rlens[idx],
                self.clens[idx])
    
    @property
    def vocab_size(self):
        # + 1 for padding symbol
        return len(self.code) + 1