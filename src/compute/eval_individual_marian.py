#!/bin/python3

"""
Run marian on all data
"""

from marian_wrap import attention
from data_loader import *
from utils_align import algn_to_feature, reverse_algn, feature_to_sent
from evaluator import *
from align_extractor import *
import numpy as np

data_ende_small = DatasetLoader('en', 'de', suff='_small', nosub=True)
data_encs = DatasetLoader('en', 'cs', suff='')
data_deen_small = DatasetLoader('de', 'en', suff='_small', nosub=True)
data_csen = DatasetLoader('cs', 'en', suff='')

alpha=0.99
method = 'avg'
extract_method=extract_3
if method == 'max':
    aggregation_method=max
elif method == 'avg':
    aggregation_method=np.average

print('method', method.upper(), '\talpha', alpha, '\n')

print('ENDE small')
scores = attention(
    'en', 'de',
    data_ende_small.sents1, data_ende_small.sents2,
    aggregation_method=aggregation_method
)
alignment = extract_method(
    data_ende_small.sents1, data_ende_small.sents2,
    scores,
    alpha=alpha
)
evaluate_dataset(alignment, data_ende_small.sure, data_ende_small.poss)
save_scores(
    scores,
    f'ende/marian_{method}.pkl',
)

print('ENCS')
scores = attention(
    'en', 'cs',
    data_encs.sents1, data_encs.sents2,
    aggregation_method=aggregation_method
)
alignment = extract_method(
    data_encs.sents1, data_encs.sents2,
    scores,
    alpha=alpha
)
evaluate_dataset(alignment, data_encs.sure, data_encs.poss)
save_scores(
    scores,
    f'encs/marian_{method}.pkl',
)

print('CSEN')
scores = attention(
    'cs', 'en',
    data_csen.sents1, data_csen.sents2,
    aggregation_method=aggregation_method
)
alignment = extract_method(
    data_csen.sents1, data_csen.sents2,
    scores,
    alpha=alpha
)
evaluate_dataset(alignment, data_csen.sure, data_csen.poss)
save_scores(
    scores,
    f'csen/marian_{method}.pkl',
)
