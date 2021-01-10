#!/bin/python3

"""
Compute all flattened features and save them 
"""


import numpy as np
from utils_align import algn_to_feature
from marian_wrap import attention, MARIAN_CONFIGS
from align_extractor import extract_1, extract_2, extract_3
from data_loader import *
from itertools import product
from sentencepiece_wrap import dataset_encoder
import sys
sys.path += [sys.path[0]+ '/../']
from utils import MODELS

LANGS = ['csen', 'encs']

features = {}
for lang in LANGS:
    features[lang]= load_any(f'{lang}/features.pkl')

for feature in ['tokpos', 'toklen', 'toksub', 'toksubl', 'tokeq', 'toklev']:
    corravg = np.average([
        np.corrcoef(features[lang][feature], features[lang]["align"])[0][1]
        for lang in LANGS]
        )
    print(f'{feature}: {corravg}')