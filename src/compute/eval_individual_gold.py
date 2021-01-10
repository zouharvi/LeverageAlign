#!/bin/python3

"""
Run theoretical max from gold on all data
"""

from marian_wrap import attention
from data_loader import *
from utils_align import algn_to_feature_sent, algn_to_feature
from evaluator import *
from align_extractor import extract_1, extract_2
from collections import Counter
import numpy as np

data_ende_small = DatasetLoader('en', 'de', suff='_small', nosub=True)
data_encs = DatasetLoader('en', 'cs', suff='')
data_deen_small = DatasetLoader('de', 'en', suff='_small', nosub=True)
data_csen = DatasetLoader('cs', 'en', suff='')

print('CS<->EN statistics')
unaligned_tokens = 0
total_tokens = 0
freq_tokens = []
for sent1, sent_align in zip(data_csen.sents1+data_encs.sents1, data_csen.sure+data_encs.sure):
    sent1 = sent1.split()
    left_align = {t1 for t1, t2 in sent_align }
    counter_align = Counter(t1 for t1, t2 in sent_align)
    freq_tokens += [v for k, v in counter_align.items() if k != 0]
    end_tok = max(max(left_align)+1, len(sent1))
    unaligned_tokens += end_tok - len(left_align)
    total_tokens += end_tok

print('Proportion of unaligned tokens', unaligned_tokens/total_tokens)
print('Average number of non-zero alignments', np.average(freq_tokens))

raise Exception("Not implemented")

print('ENDE small')
scores = algn_to_feature_sent(data_ende_small.sents1, data_ende_small.sents2, data_ende_small.sure)
alignment = extract_1(data_ende_small.sents1, data_ende_small.sents2, scores)
evaluate_dataset(alignment, data_ende_small.sure)

print('ENCS')
scores = algn_to_feature_sent(data_encs.sents1, data_encs.sents2, data_encs.sure)
alignment = extract_1(data_encs.sents1, data_encs.sents2, scores)
evaluate_dataset(alignment, data_encs.sure)

print('CSEN')
scores = algn_to_feature_sent(data_csen.sents1, data_csen.sents2, data_csen.sure)
alignment = extract_1(data_csen.sents1, data_csen.sents2, scores)
evaluate_dataset(alignment, data_csen.sure)