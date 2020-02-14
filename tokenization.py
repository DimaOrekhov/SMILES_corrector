import re


LEVEL1 = set(["C", "H", "O", "N", "c", "h", "o", "n"])
LEVEL2 = set(["Br", "S", "Cl", "B", "F", "br", "s", "cl", "b", "f"])
LEVEL3 = set(["Si", "I", "P", "si", "i", "p"])
LEVEL4 = set(["Sn", "Se", "Mg", "sn", "se", "mg"])


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    
    Что я изменил:
    1. добавил поиск H: "...|H|..."
    2. "...|S|..." -> "...|Si?|..."
    3. "...|>|..." -> "...|>>|...", символ реакции - один токен
    
    * У них есть такой паттерн \[[^\]]+], то есть все, что обернуто в квадратные скобки - один токен,
    видел такое и в других статьях. Не очень понимаю только, почему паттерн не \[[^\]]+\]. Разве не нужно
    экранировать вторую закрывающую скобку тоже? На всякий случай, я экранировал вторую скобку тоже.
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|H|N|O|Si?|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    #assert smi == ''.join(tokens)
    return ' '.join(tokens)
