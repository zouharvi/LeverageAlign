#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import json
"""
Adapted from hw2 of https://github.com/xutaima/jhu-mt-hw
"""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('gold', help='Gold alignments file')
parser.add_argument('hypothesis', help='Hypothesis alignments file')
parser.add_argument('--gold-index-one', action='store_true', help="Shift hypothesis by one.")
args = parser.parse_args()

aers = []
metric_p = []
metric_r = []
for (i, (algn_gold, algn_hypothesis)) in enumerate(zip(open(args.gold), open(args.hypothesis))):
    if algn_hypothesis == "<title>400 Bad Request</title>\n":
        print("Skipping")
        continue
    algn_hypothesis = json.loads(algn_hypothesis)["alignment"] 
    sure = set([tuple(map(int, x.split("-"))) for x in algn_gold.strip().split() if x.find("-") > -1])
    possible = set([tuple(map(int, x.split("?"))) for x in algn_gold.strip().split() if x.find("?") > -1])
    alignment = set([tuple(map(int, x.split("-"))) for x in algn_hypothesis.strip().split()])
    
    if args.gold_index_one:
        alignment = {(x+1,y+1) for x,y in alignment}

    size_a = len(alignment)
    size_s = len(sure)
    size_a_and_s = len(alignment & sure)
    assert(len(sure & possible) == 0)
    size_a_and_p = len(alignment & possible) + len(alignment & sure)
    aers.append(0 if size_a + size_s == 0 else 1-(size_a_and_s + size_a_and_p) / (size_a + size_s))
    metric_p.append(0 if size_a == 0 else size_a_and_p / size_a)
    metric_r.append(0 if size_s == 0 else size_a_and_s / size_s)

print("AER", np.average(aers))
print("Recall", np.average(metric_r))
print("Precision", np.average(metric_p))
print(len(aers))