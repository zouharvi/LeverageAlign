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
import argparse
from Levenshtein import distance as levdist
import sys
sys.path += [sys.path[0]+ '/../']
from utils import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--langs', default='ende')
args = parser.parse_args()

langs = args.langs
if langs == 'encs':
    dataset = DatasetLoader('en', 'cs', maxlen=None)
if langs == 'csen':
    dataset = DatasetLoader('cs', 'en', maxlen=None)
elif langs == 'ende':
    dataset = DatasetLoader('en', 'de', suff='_small', nosub=True)

print(langs.upper())

instances = sum([len(s1.split())*len(s2.split()) for s1, s2 in zip(dataset.sents1, dataset.sents2)])
instances_pos = sum([len(algn) for algn in dataset.sure])

print(f'instances {instances:>9}')
print(f'positive  {instances_pos:>9}')
print(f'proportion {instances_pos/instances*100:8.2f}%')

features = {}

print('models')
for model in MODELS:
    model_data = load_scores(langs + '/' + model + '.pkl')
    features[model] = [item for sublist in model_data for item in sublist]
    print(len(features[model]))

for model in ['marian_avg', 'fastalign'] + (['fastalign_small'] if langs == 'ende' else []):
    print(model)
    model_data = load_scores(langs + '/' + model + '.pkl')
    features[model] = [item for sublist in model_data for item in sublist]
    print(len(features[model]))

print('raw')
features['raw'] = []
for sent1, sent2 in zip(dataset.sents1, dataset.sents2):
    sent1 = sent1.split()
    sent2 = sent2.split()
    for t1, t2 in product(sent1, sent2):
        features['raw'].append((t1, t2))
print(len(features['raw']))

features['align'] = algn_to_feature(dataset.sents1, dataset.sents2, dataset.sure)

print('custom features')
features['tokpos'] = []
features['toklen'] = []
features['toksub'] = []
features['toksubl'] = []
features['tokeq'] = []
features['toklev'] = []
vocab = MARIAN_CONFIGS[langs]['vocab']
for sent1, sent2 in zip(dataset.sents1, dataset.sents2):
    sent1 = sent1.split()
    sent2 = sent2.split()
    # sentences are split here, so that every considered token is a "sentence" for SentencePiece
    subwords1 = dataset_encoder(sent1, vocab)
    subwords2 = dataset_encoder(sent2, vocab)
    for (i1, t1), (i2, t2) in product(enumerate(sent1), enumerate(sent2)):
        features['tokpos'].append(abs(i1/len(sent1)-i2/len(sent2)))
        features['toklen'].append(abs(len(t1)-len(t2)))
        subw1 = set(subwords1[i1])
        subw2 = set(subwords2[i2])
        features['toksub'].append(len(subw1 & subw2))
        features['toksubl'].append(abs(len(subw1) - len(subw2)))
        features['tokeq'].append(1*(t1.lower()==t2.lower()))
        features['toklev'].append(levdist(t1.lower(), t2.lower())/max(len(t1), len(t2)))
print(len(features['tokpos']))
print(len(features['toklen']))
print(len(features['toksub']))
print(len(features['toksubl']))
print(len(features['tokeq']))
print(len(features['toklev']))

print('saving')
save_any(features, f'{langs}/features.pkl')