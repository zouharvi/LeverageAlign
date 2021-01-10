#!/bin/python3

"""
Plot precision, recall and F1 curves
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys
sys.path += [sys.path[0]+ '/../']
from utils import MODELS, MODELS_NICE, MODEL_PLOT_STYLE

with open('computed/metrics_data.pkl', 'rb') as f:
    data = pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--extractor', default='A3')
parser.add_argument('-l', '--langs', default='csen_mix')
parser.add_argument('-g', '--grayscale', action='store_true')
args = parser.parse_args()

EXTRACTOR = args.extractor
DATASET = args.langs
GRAYSCALE = args.grayscale

def plot(metrics):
    fig, axes = plt.subplots(1,3, figsize=(9, 3.5))
    for model in MODELS:
        alphas = [x[0] for x in metrics[model][EXTRACTOR]]
        prec = [x[1][0] for x in metrics[model][EXTRACTOR]]
        recl = [x[1][1] for x in metrics[model][EXTRACTOR]]
        aer   = [x[1][2] for x in metrics[model][EXTRACTOR]]
        if GRAYSCALE:
            axes[0].plot(alphas, prec, color=MODEL_PLOT_STYLE[model][0], linestyle=MODEL_PLOT_STYLE[model][1], alpha=0.8)
            axes[1].plot(alphas, recl, color=MODEL_PLOT_STYLE[model][0], linestyle=MODEL_PLOT_STYLE[model][1], alpha=0.8)
            axes[2].plot(alphas, aer, color=MODEL_PLOT_STYLE[model][0], linestyle=MODEL_PLOT_STYLE[model][1], alpha=0.8, label=MODELS_NICE[model])
        else:
            axes[0].plot(alphas, prec, alpha=0.8)
            axes[1].plot(alphas, recl, alpha=0.8)
            axes[2].plot(alphas, aer, alpha=0.8, label=MODELS_NICE[model])
        print(model, 'max', np.min(aer))
    axes[0].set_xlabel('alpha')
    axes[1].set_xlabel('alpha')
    axes[2].set_xlabel('alpha')
    axes[0].set_ylabel('Precision', labelpad=-0.5)
    axes[1].set_ylabel('Recall', labelpad=-0.5)
    axes[2].set_ylabel('AER', labelpad=-0.5)
    axes[0].locator_params(nbins=8)
    axes[1].locator_params(nbins=8)
    axes[2].locator_params(nbins=8)
    fig.legend(loc='upper center', bbox_transform=(0.5, 1.1), ncol=len(MODELS))
    plt.tight_layout(pad=1, w_pad=0.5)
    plt.subplots_adjust(top=0.85)
    plt.show()

matplotlib.rcParams.update({'font.size': 12})

data['csen_mix'] = {}
for model in MODELS:
    data['csen_mix'][model] = {}
    data['csen_mix'][model][EXTRACTOR] = [
        (alpha1, (
            np.average([prec1, prec2]),
            np.average([recl1, recl2]),
            np.average([fone1, fone2]),
        ))
        for (
            (alpha1, (prec1, recl1, fone1)),
            (alpha2, (prec2, recl2, fone2))
        )
        in zip(data['csen'][model][EXTRACTOR], data['encs'][model][EXTRACTOR])
    ]
plot(data[DATASET])
