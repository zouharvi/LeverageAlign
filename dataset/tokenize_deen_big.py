#!/bin/env python3
from mosestokenizer import *
import random

with open('TildeMODEL/de.txt', 'r') as f:
    data_de = f.read().split('\n')
with open('TildeMODEL/en.txt', 'r') as f:
    data_en = f.read().split('\n')

# sample sentences
random.seed(2)
data_small = random.sample(list(zip(data_de, data_en)), 1000000)
data_de = [x[0] for x in data_small]
data_en = [x[1] for x in data_small]

with MosesTokenizer('de') as tokenize:
    data_de_tok = []
    for i, line in enumerate(data_de):
        if i % 10000 == 0:
            print(f'de: {i/len(data_de):.4f}')
        data_de_tok.append(' '.join(tokenize(line)))
with open('TildeMODEL/de.tok', 'w') as f:
    f.write('\n'.join(data_de_tok))

with MosesTokenizer('en') as tokenize:
    data_en_tok = []
    for i, line in enumerate(data_en):
        if i % 10000 == 0:
            print(f'en: {i/len(data_en):.4f}')
        data_en_tok.append(' '.join(tokenize(line))) 
with open('TildeMODEL/en.tok', 'w') as f:
    f.write('\n'.join(data_en_tok))