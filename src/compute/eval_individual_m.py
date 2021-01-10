#!/bin/python3

"""
Compute all model scores
"""

from models import *
from fastalign_wrap import fast_align
from align_extractor import extract_1, extract_2, extract_3
from data_loader import *
from evaluator import *

data_ende = DatasetLoader('en', 'de', suff='_small', maxlen=None)
sents1 = data_ende.sents1
sents2 = data_ende.sents2

print('model 1')
scores = model_m1('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m1.pkl')
print('model 2a')
scores = model_m2a('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m2b.pkl')
print('model 2b')
scores = model_m2b('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m2a.pkl')
print('model 3aa')
scores = model_m3aa('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m3aa.pkl')
print('model 3ab')
scores = model_m3ab('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m3ab.pkl')
print('model 3ba')
scores = model_m3ba('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m3ba.pkl')
print('model 3bb')
scores = model_m3bb('en', 'de', sents1, sents2)
save_scores(scores, 'ende/m3bb.pkl')

data_encs = DatasetLoader('en', 'cs', maxlen=None)
sents1 = data_encs.sents1
sents2 = data_encs.sents2

print('model 1')
scores = model_m1('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m1.pkl')
print('model 2a')
scores = model_m2a('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m2b.pkl')
print('model 2b')
scores = model_m2b('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m2a.pkl')
print('model 3aa')
scores = model_m3aa('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m3aa.pkl')
print('model 3ab')
scores = model_m3ab('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m3ab.pkl')
print('model 3ba')
scores = model_m3ba('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m3ba.pkl')
print('model 3bb')
scores = model_m3bb('en', 'cs', sents1, sents2)
save_scores(scores, 'encs/m3bb.pkl')

data_csen = DatasetLoader('cs', 'en', maxlen=None)
sents1 = data_csen.sents1
sents2 = data_csen.sents2

print('model 1')
scores = model_m1('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m1.pkl')
print('model 2a')
scores = model_m2a('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m2b.pkl')
print('model 2b')
scores = model_m2b('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m2a.pkl')
print('model 3aa')
scores = model_m3aa('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m3aa.pkl')
print('model 3ab')
scores = model_m3ab('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m3ab.pkl')
print('model 3ba')
scores = model_m3ba('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m3ba.pkl')
print('model 3bb')
scores = model_m3bb('cs', 'en', sents1, sents2)
save_scores(scores, 'csen/m3bb.pkl')