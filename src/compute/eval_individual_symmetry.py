#!/bin/python3

"""
Run marian on all data
"""

from marian_wrap import attention
from data_loader import *
from utils_align import algn_to_feature, reverse_algn, feature_to_sent, comb_algn_scores, intersect_algn
from evaluator import *
from align_extractor import extract_4, extract_2, extract_3
import numpy as np
import math

data_ende_small = DatasetLoader('en', 'de', suff='_small', nosub=True)
data_encs = DatasetLoader('en', 'cs', suff='')
data_deen_small = DatasetLoader('de', 'en', suff='_small', nosub=True)
data_csen = DatasetLoader('cs', 'en', suff='')

RECIPES = [
    ('m1', lambda a, b: a + b),
    ('m3bb', lambda a, b: a + b),
    ('marian_max', lambda a, b: a * b),
    ('marian_avg', lambda a, b: a * b),
]

print('DEEN marian_avg')
scores_bw = load_scores(f'ende/marian_avg.pkl')
scores_r = reverse_algn(
    data_deen_small.sents1,
    data_deen_small.sents2,
    scores_bw
)
alignment_r = extract_4(
    data_deen_small.sents1,
    data_deen_small.sents2,
    scores_r,
    alpha=0.99
)
evaluate_dataset(alignment_r, data_deen_small.sure, data_deen_small.poss)

print('ENDE M1')
scores_fw = load_scores(f'ende/marian_avg.pkl')
# best ensemble alignment
alignment_a3 = extract_3(
    data_ende_small.sents1,
    data_ende_small.sents2,
    scores_fw,
    alpha=1
)
alignment_a4 = extract_4(
    data_ende_small.sents1,
    data_ende_small.sents2,
    scores_fw,
    alpha=1
)
alignment_a3a4 = intersect_algn(alignment_a3, alignment_a4)
evaluate_dataset(alignment_a3a4, data_ende_small.sure, data_ende_small.poss)

for model, combiner in RECIPES:
    print('\nMODEL', model)
    scores_fw = load_scores(f'encs/{model}.pkl')
    scores_bw = load_scores(f'csen/{model}.pkl')

    print('ENCS')
    scores_r = reverse_algn(
        data_encs.sents1,
        data_encs.sents2,
        scores_bw
    )
    scores_mult = comb_algn_scores(scores_fw, scores_r)

    alignment_fw = extract_4(
        data_encs.sents1,
        data_encs.sents2,
        scores_fw,
        alpha=0.99
    )
    alignment_r = extract_4(
        data_encs.sents1,
        data_encs.sents2,
        scores_r,
        alpha=0.99
    )
    alignment_mult = extract_4(
        data_encs.sents1,
        data_encs.sents2,
        scores_mult,
        alpha=0.99
    )
    alignment_intersect = intersect_algn(alignment_fw, alignment_r)

    # best ensemble alignment
    alignment_a3 = extract_3(
        data_encs.sents1,
        data_encs.sents2,
        scores_fw,
        alpha=1
    )
    alignment_a4 = extract_4(
        data_encs.sents1,
        data_encs.sents2,
        scores_fw,
        alpha=1
    )
    alignment_a3a4 = intersect_algn(alignment_a3, alignment_a4)
    
    print('fw')
    evaluate_dataset(alignment_fw, data_encs.sure, data_encs.poss)
    print('r')
    evaluate_dataset(alignment_r, data_encs.sure, data_encs.poss)
    print('mult')
    evaluate_dataset(alignment_mult, data_encs.sure, data_encs.poss)
    print('intersect')
    evaluate_dataset(alignment_intersect, data_encs.sure, data_encs.poss)
    print('a3a4')
    evaluate_dataset(alignment_a3a4, data_encs.sure, data_encs.poss)

    print('CSEN')
    scores_fw, scores_bw = scores_bw, scores_fw

    scores_r = reverse_algn(
        data_csen.sents1,
        data_csen.sents2,
        scores_bw
    )
    scores_mult = comb_algn_scores(scores_fw, scores_r)

    alignment_fw = extract_4(
        data_csen.sents1,
        data_csen.sents2,
        scores_fw,
        alpha=0.99
    )
    alignment_r = extract_4(
        data_csen.sents1,
        data_csen.sents2,
        scores_r,
        alpha=0.99
    )
    alignment_mult = extract_4(
        data_csen.sents1,
        data_csen.sents2,
        scores_mult,
        alpha=0.99
    )
    alignment_intersect = intersect_algn(alignment_fw, alignment_r)
    
    # best ensemble alignment
    alignment_a3 = extract_3(
        data_csen.sents1,
        data_csen.sents2,
        scores_fw,
        alpha=1
    )
    alignment_a4 = extract_4(
        data_csen.sents1,
        data_csen.sents2,
        scores_fw,
        alpha=1
    )
    alignment_a3a4 = intersect_algn(alignment_a3, alignment_a4)

    print('fw')
    evaluate_dataset(alignment_fw, data_csen.sure, data_csen.poss)
    print('r')
    evaluate_dataset(alignment_r, data_csen.sure, data_csen.poss)
    print('mult')
    evaluate_dataset(alignment_mult, data_csen.sure, data_csen.poss)
    print('intersect')
    evaluate_dataset(alignment_intersect, data_csen.sure, data_csen.poss)
    print('a3a4')
    evaluate_dataset(alignment_a3a4, data_csen.sure, data_csen.poss)