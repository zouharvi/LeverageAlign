#!/bin/python3

"""
Run fast_align on all data
"""

from utils_align import algn_to_feature_sent
from fastalign_wrap import fast_align
from data_loader import *
from evaluator import *
import numpy as np

data_ende_big = DatasetLoader('en', 'de', suff='_big', noalgn=True)
data_ende_small = DatasetLoader('en', 'de', suff='_small', nosub=True)
data_encs = DatasetLoader('en', 'cs', suff='')
data_deen_big = DatasetLoader('de', 'en', suff='_big', noalgn=True)
data_deen_small = DatasetLoader('de', 'en', suff='_small', nosub=True)
data_csen = DatasetLoader('cs', 'en', suff='')

print('ENDE small')
alignment = fast_align(
    data_ende_small.sents1, data_ende_small.sents2, []
)
evaluate_dataset(alignment, data_ende_small.sure, data_ende_small.poss)
# save_scores(
#     algn_to_feature_sent(data_ende_small.sents1, data_ende_small.sents2, alignment),
#     'ende/fastalign_small.pkl',
# )

print('ENDE small + big')
alignment = fast_align(
    data_ende_small.sents1, data_ende_small.sents2,
    zip(data_ende_big.sents1, data_ende_big.sents2)
)
evaluate_dataset(alignment, data_ende_small.sure, data_ende_small.poss)
# save_scores(
#     algn_to_feature_sent(data_ende_small.sents1, data_ende_small.sents2, alignment),
#     'ende/fastalign.pkl',
# )

print('ENCS')
alignment = fast_align(
    data_encs.sents1, data_encs.sents2, []
)
evaluate_dataset(alignment, data_encs.sure, data_encs.poss)
# save_scores(
#     algn_to_feature_sent(data_encs.sents1, data_encs.sents2, alignment),
#     'encs/fastalign.pkl',
# )


print('CSEN')
alignment = fast_align( 
    data_csen.sents1, data_csen.sents2, []
)
evaluate_dataset(alignment, data_csen.sure, data_csen.poss)
# save_scores(
#     algn_to_feature_sent(data_csen.sents1, data_csen.sents2, alignment),
#     'csen/fastalign.pkl',
# )
