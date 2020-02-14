from rdkit import Chem
import random


def get_alternative_smiles(smiles: str, n_copies: int=2, n_tries: int=3) -> list:
    alternatives = set()
    i = 0
    while len(alternatives) < n_copies and i < n_copies * n_tries:
        alt = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True)
        alternatives.add(alt)
        i += 1
    return list(alternatives)


class Enumerator:

    @staticmethod
    def get_alternative_smiles(smi_list : list):
        """
        One or less alternative SMILES string per input string
        """
        alternative_smiles = []
        for smi in smi_list:
            alt = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
            if smi != alt:
                alternative_smiles.append(alt)
        random.shuffle(alternative_smiles)
        return alternative_smiles


    @staticmethod
    def get_alternative_smiles_tries(smi_list : list, n_tries : int = 10):
        alternative_smiles = []
        for smi in smi_list:
            for _ in range(n_tries):
                alt = Chem.MolToSmiles(Chem.MolFromSmile(smi), doRandom=True)
                if smi != alt:
                    alternative_smiles.append(alt)
                    break
        random.shuffle(alternative_smiles)
        return alterative_smiles

    @staticmethod
    def get_n_alternatives(smiles : str, n : int, n_tries : int = 10):
        seqs = set([smiles])
        for _ in range(n):
            pass
