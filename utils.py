import json
from tqdm import tqdm_notebook
import re


def get_reactions(json_path: str, n : int = None)->list:
    reactions = []
    with open(json_path) as fsream:
        for i, line in enumerate(fsream):
            reactions.append(json.loads(line))
            if n and i >= n:
                break
    return reactions


def get_vocab_from_tokenized_full_json(reations: list):
    from collections import Counter
    counter = Counter()
    for react in tqdm_notebook(reactions):
        for char in react['reaction_core'].split():
            counter[char] += 1
        for reag in react['reactants']:
            for char in reag.split():
                counter[char] += 1
        for char in react['products'][0].split():
            counter[char] += 1
    return counter

def get_vocab_from_tokenized(in_fname: str):
    from collections import Counter
    exp = re.compile(r"^(reactant|product)_\d+$")
    vocab = Counter()
    with open(in_fname, 'r') as istream:
        for entry in tqdm_notebook(istream):
            entry = json.loads(entry)
            for key in entry.keys():
                if not exp.match(key):
                    continue
                for entity in entry[key]:
                    for ch in entity.split():
                        vocab[ch] += 1 
    return vocab


def get_reaction_str(reaction: dict)->str:
    result = reaction['reactants'][0]
    if len(reaction['reactants']) == 2:
        result += " . " + reaction['reactants'][1]
    result += " >> " + reaction['products'][0]
    return result


def filter_by_tokens(reactions: list, tokens: set)->list:
    result = []
    for react in tqdm_notebook(reactions):
        if get_tokenset(react) <= tokens:
            result.append(react)
    return result


def get_tokenset(reaction: dict)->set:
    result = set()
    for char in reaction['reaction_core'].split():
        result.add(char)
    for reag in reaction['reactants']:
        for char in reag.split():
            result.add(char)
    for char in reaction['products'][0].split():
        result.add(char)
    return result


def vocab_to_tokenset(vocab: dict, threshold: int=0)->set:
    result = set()
    for k, v in vocab.items():
        if v >= threshold:
            result.add(k)
    return result


def to_jsonl(reactions: list, outfile: str):
    with open(outfile, 'w') as fstream:
        for r in reactions:
            json.dump(r, fstream)
            fstream.write('\n')
