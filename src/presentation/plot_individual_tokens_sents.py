#!/bin/python3

"""
Plot performance dependency on predicted token count and sentence length
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path += [sys.path[0]+ '/../']
from utils import MODELS, MODELS_NICE, MODEL_PLOT_STYLE

with open('computed/metrics_data_sentlen.pkl', 'rb') as f:
    data_sentlen = pickle.load(f)
with open('computed/metrics_data_tokens.pkl', 'rb') as f:
    data_tokens = pickle.load(f)

EXTRACTOR = 'A3'
DATASET = 'csen_mix'
GRAYSCALE = False

def plot(metrics_sentlen, metrics_tokens):
    fig, axes = plt.subplots(1,2, figsize=(10, 3.5))
    for model in MODELS:
        alphas = [x[0] for x in metrics_tokens[model][EXTRACTOR]]
        mapped_tokens = [x[1] for x in metrics_tokens[model][EXTRACTOR]]
        sentlens = [x[0] for x in metrics_sentlen[model][EXTRACTOR]]
        AER   = [x[1][2] for x in metrics_sentlen[model][EXTRACTOR]]
        if GRAYSCALE:
            axes[0].plot(sentlens, AER, label=MODELS_NICE[model], color=MODEL_PLOT_STYLE[model][0], linestyle=MODEL_PLOT_STYLE[model][1], alpha=0.8)
            axes[1].plot(alphas, mapped_tokens, color=MODEL_PLOT_STYLE[model][0], linestyle=MODEL_PLOT_STYLE[model][1], alpha=0.8)
        else:
            axes[0].plot(sentlens, AER, label=MODELS_NICE[model], alpha=0.8)
            axes[1].plot(alphas, mapped_tokens, alpha=0.8)    
    axes[0].set_xlabel('Max sentence length\nalpha = 1')
    axes[1].set_xlabel('alpha')
    axes[0].set_ylabel('AER')
    axes[1].set_ylabel('Aligned tokens')
    axes[0].locator_params(nbins=8)
    axes[1].locator_params(nbins=8)
    fig.legend(loc='upper center', bbox_transform=(0.5, 1.1), ncol=len(MODELS))
    plt.tight_layout(rect=(0.14, -0.05, 0.76, 1))
    plt.subplots_adjust(top=0.85)
    plt.show()

matplotlib.rcParams.update({'font.size': 12})

data_sentlen['csen_mix'] = {}
for model in MODELS:
    data_sentlen['csen_mix'][model] = {}
    data_sentlen['csen_mix'][model][EXTRACTOR] = [
        (alpha1, (
            np.average([prec1, prec2]),
            np.average([recl1, recl2]),
            np.average([fone1, fone2]),
        ))
        for (
            (alpha1, (prec1, recl1, fone1)),
            (alpha2, (prec2, recl2, fone2))
        )
        in zip(data_sentlen['csen'][model][EXTRACTOR], data_sentlen['encs'][model][EXTRACTOR])
    ]
data_tokens['csen_mix'] = {}
for model in MODELS:
    data_tokens['csen_mix'][model] = {}
    data_tokens['csen_mix'][model][EXTRACTOR] = [
        (alpha1, np.average([tokens1, tokens2]))
        for (
            (alpha1, tokens1),
            (alpha2, tokens2)
        )
        in zip(data_tokens['csen'][model][EXTRACTOR], data_tokens['encs'][model][EXTRACTOR])
    ]

plot(data_sentlen[DATASET], data_tokens[DATASET])