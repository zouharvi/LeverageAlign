#!/bin/python3

"""
Evaluate all individual models
"""

from models import *
from fastalign_wrap import fast_align
from align_extractor import extract_1, extract_2, extract_3, extract_4
from data_loader import *
from evaluator import *
import numpy as np
from collections import defaultdict
import argparse
import sys
sys.path += [sys.path[0] + '/../']
from utils import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--performance', action='store_true', default=False)
parser.add_argument('-t', '--tokens', action='store_true', default=False)
parser.add_argument('-s', '--sentlen', action='store_true', default=False)
args = parser.parse_args()

PERFORMANCE_METRICS = args.performance
TOKENS_METRICS = args.tokens
SENT_LEN_METRICS = args.sentlen

QUEUE = [('en', 'de', '_small', True),
         ('en', 'cs', '', False), ('cs', 'en', '', False)]

if PERFORMANCE_METRICS:
    # returning (alpha, (precision, recall, F1))
    def metrics(dataset, scores):
        # A1 is skipped, because it is already part of A3
        metrics_data = {}
        metrics_data['A2'] = []
        for alpha in range(-200, 40, 10):
            print(alpha)
            alignment = extract_2(
                dataset.sents1, dataset.sents2,
                scores,
                alpha=alpha
            )
            metrics_data['A2'].append(
                (alpha, evaluate_dataset(alignment, dataset.sure, dataset.poss)))

        metrics_data['A3'] = []
        for alpha in np.arange(0, 1.025, 0.025):
            print(alpha)
            alignment = extract_3(
                dataset.sents1, dataset.sents2,
                scores,
                alpha=alpha
            )
            metrics_data['A3'].append(
                (alpha, evaluate_dataset(alignment, dataset.sure, dataset.poss)))

        metrics_data['A4'] = []
        for alpha in np.arange(0.5, 1.025, 0.025):
            print(alpha)
            alignment = extract_4(
                dataset.sents1, dataset.sents2,
                scores,
                alpha=alpha
            )
            metrics_data['A4'].append(
                (alpha, evaluate_dataset(alignment, dataset.sure, dataset.poss)))
        return metrics_data

    data = {}

    for lang1, lang2, suff, nosub in QUEUE:
        dataset = DatasetLoader(lang1, lang2, suff=suff, nosub=nosub)
        data[lang1+lang2] = {}
        for model in MODELS:
            print(lang1+lang2, model)
            scores = load_scores(f'{lang1}{lang2}/{model}.pkl')
            data[lang1+lang2][model] = metrics(dataset, scores)

    with open('computed/metrics_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if TOKENS_METRICS:
    def aggregate_tokens(alignment, sents1, sents2):
        tokens_data = []
        for sent_algn, sent1, sent2 in zip(alignment, sents1, sents2):
            sent_data = {}
            for i, _ in enumerate(sent1.split()):
                sent_data[i] = 0
            for t1, t2 in sent_algn:
                sent_data[t1] += 1
            tokens_data += sent_data.values()
        return np.average(tokens_data)

    def metrics(dataset, scores):
        # A1 is skipped, because it is already part of A3
        metrics_data = {}
        metrics_data['A2'] = []
        for alpha in range(-200, 40, 10):
            print(alpha)
            alignment = extract_2(
                dataset.sents1, dataset.sents2,
                scores,
                alpha=alpha
            )
            metrics_data['A2'].append((alpha, aggregate_tokens(
                alignment, dataset.sents1, dataset.sents2)))

        metrics_data['A3'] = []
        for alpha in np.arange(0, 1.025, 0.025):
            print(alpha)
            alignment = extract_3(
                dataset.sents1, dataset.sents2,
                scores,
                alpha=alpha
            )
            metrics_data['A3'].append((alpha, aggregate_tokens(
                alignment, dataset.sents1, dataset.sents2)))
        return metrics_data

    data = {}
    for lang1, lang2, suff, nosub in QUEUE:
        dataset = DatasetLoader(lang1, lang2, suff=suff, nosub=nosub)
        data[lang1+lang2] = {}
        for model in MODELS:
            print(lang1+lang2, model)
            scores = load_scores(f'{lang1}{lang2}/{model}.pkl')
            data[lang1+lang2][model] = metrics(dataset, scores)

    with open('computed/metrics_data_tokens.pkl', 'wb') as f:
        pickle.dump(data, f)

if SENT_LEN_METRICS:
    def eval_by_len(alignmentA, alignmentS, alignmentP, sents1, sent_maxlen):
        new_data = [
            (algnA, algnS, algnP)
            for (algnA, algnS, algnP, sent1)
            in zip(alignmentA, alignmentS, alignmentP, sents1)
            if len(sent1.split()) <= sent_maxlen
        ]
        return evaluate_dataset([x[0] for x in new_data], [x[1] for x in new_data], [x[2] for x in new_data])

    def metrics(dataset, scores):
        # A1 is skipped, because it is already part of A3
        metrics_data = {}
        metrics_data['A3'] = []
        alignment = extract_3(
            dataset.sents1, dataset.sents2,
            scores,
            alpha=1
        )
        for sent_maxlen in range(1, 40):
            print(sent_maxlen)
            metrics_data['A3'].append((sent_maxlen, eval_by_len(
                alignment, dataset.sure, dataset.poss, dataset.sents1, sent_maxlen)))
        return metrics_data

    data = {}
    for lang1, lang2, suff, nosub in QUEUE:
        dataset = DatasetLoader(lang1, lang2, suff=suff, nosub=nosub)
        data[lang1+lang2] = {}
        for model in MODELS:
            print(lang1+lang2, model)
            scores = load_scores(f'{lang1}{lang2}/{model}.pkl')
            data[lang1+lang2][model] = metrics(dataset, scores)

    with open('computed/metrics_data_sentlen.pkl', 'wb') as f:
        pickle.dump(data, f)
