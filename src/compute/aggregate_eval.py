#!/bin/python3

from data_loader import *
from evaluator import *
from align_extractor import *
from utils_align import feature_to_sent, reverse_algn, sent_to_feature, intersect_algn
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import argparse
import math
DEVICE = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--langs', default='ende')
parser.add_argument('--model-langs', default=None)
parser.add_argument('-m', '--model-type', default='i')
parser.add_argument('-e', '--epoch', default=0)
args, _ = parser.parse_known_args()

FEATURES_ALL = {}
FEATURES_ALL['h'] = ['tokpos', 'toklen', 'tokeq', 'toklev', 'toksub', 'toksubl']
FEATURES_ALL['i'] = ['m1', 'm2b', 'm3aa', 'm3bb']
FEATURES_ALL['f'] = ['fastalign']
FEATURES_ALL['a'] = ['marian_avg']
FEATURES_ALL['r'] = ['m1r']
FEATURES_ALL['align'] = ['align']

langs = args.langs
if args.model_langs is None:
    model_langs = langs
else:
    model_langs = args.model_langs
model_type = args.model_type
epoch = args.epoch
print(f"{langs}_{model_type}")

if langs == 'encs':
    dataset = DatasetLoader('en', 'cs', maxlen=None)
    trainSentCount = 4000
    validSentCount = 500
if langs == 'csen':
    dataset = DatasetLoader('cs', 'en', maxlen=None)
    trainSentCount = 4000
    validSentCount = 500
elif langs == 'ende':
    dataset = DatasetLoader('en', 'de', suff='_small', nosub=True)
    trainSentCount = 100
    validSentCount = 0

# hotfix exploding values
def data_sanitizer(feature, i):
    if feature in {'m3aa', 'm3bb'}:
        return math.exp(data[feature][i])
    else:
        return data[feature][i]

data = load_scores(f'{langs}/features.pkl')
features = []
for k in args.model_type:
  if k == 'r':
    data_bw = load_scores(f'{langs[2:]}{langs[:2]}/features.pkl')['m1']
    scores_bw = feature_to_sent(dataset.sents2, dataset.sents1, data_bw)
    data['m1r'] = reverse_algn(dataset.sents1, dataset.sents2, scores_bw)
    data['m1r'] = sent_to_feature(data['m1r'])
    print(len(data['m1r']))
    print(len(data['m1']))
  features += FEATURES_ALL[k]

dataY = np.array(data['align'])
dataX = np.array([np.array([data_sanitizer(feature, i) for feature in features]) for i in range(len(dataY))])
del data

sents1train = dataset.sents1[:trainSentCount]
sents2train = dataset.sents2[:trainSentCount]

sents1valid  = dataset.sents1[trainSentCount:trainSentCount+validSentCount]
sents2valid  = dataset.sents2[trainSentCount:trainSentCount+validSentCount]
alignValidS   = dataset.sure[trainSentCount:trainSentCount+validSentCount]
alignValidP   = dataset.poss[trainSentCount:trainSentCount+validSentCount]

sents1test = dataset.sents1[trainSentCount+validSentCount:trainSentCount+2*validSentCount]
sents2test = dataset.sents2[trainSentCount+validSentCount:trainSentCount+2*validSentCount]
alignTestS  = dataset.sure[trainSentCount+validSentCount:trainSentCount+2*validSentCount]
alignTestP  = dataset.poss[trainSentCount+validSentCount:trainSentCount+2*validSentCount]

trainCount = sum([len(s1.split())*len(s2.split()) for s1, s2 in zip(sents1train, sents2train)])
testCount  = sum([len(s1.split())*len(s2.split()) for s1, s2 in zip(sents1test, sents2test)])
validCount = sum([len(s1.split())*len(s2.split()) for s1, s2 in zip(sents1valid, sents2valid)])

dataXbase = torch.tensor(dataX.reshape(-1, len(features))).type(torch.FloatTensor).to(DEVICE)
dataYbase = torch.tensor(np.array(dataY).reshape(-1, 1)).type(torch.FloatTensor).to(DEVICE)
dataX = dataXbase.narrow(0, 0, trainCount)
dataY = dataYbase.narrow(0, 0, trainCount)
dataXvalid = dataXbase.narrow(0, trainCount, validCount)
dataYvalid = dataYbase.narrow(0, trainCount, validCount)
dataXtest = dataXbase.narrow(0, trainCount+validCount, testCount)
dataYtest = dataYbase.narrow(0, trainCount+validCount, testCount)
del dataXbase
del dataYbase

print(dataX.shape)
print(dataY.shape)
print(dataXvalid.shape)
print(dataYvalid.shape)
print(dataXtest.shape)
print(dataYtest.shape)
print(trainCount+testCount+validCount)

# MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(features), 32)
        self.do1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.do2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.do1(x)
        x = self.act(self.fc2(x))
        x = self.do2(x)
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.sig(self.fc5(x))
        return x

model = torch.load(f'computed/{model_langs}_{model_type}_{epoch:0>4}.pt', map_location=torch.device('cpu'))
def evaluate(ALPHA2 = 0.0001, ALPHA3 = 1, ALPHA4 = 1):
    model.eval()
    print('VALID')
    pred = feature_to_sent(sents1valid, sents2valid, model(dataXvalid).cpu().reshape(-1).detach().numpy())
    algn4 = extract_4(sents1valid, sents2valid, pred, alpha=ALPHA4)
    algn3 = extract_3(sents1valid, sents2valid, pred, alpha=ALPHA3)
    algn2 = extract_2(sents1valid, sents2valid, pred, alpha=ALPHA2)
    algnI = intersect_algn(algn2, intersect_algn(algn3, algn4))
    valAER = evaluate_dataset(algnI, alignValidS, alignValidP, verbose=True)[2]
    print('TRAIN')
    pred = feature_to_sent(sents1train, sents2train, model(dataX).cpu().reshape(-1).detach().numpy())
    algn4 = extract_4(sents1train, sents2train, pred, alpha=ALPHA4)
    algn3 = extract_3(sents1train, sents2train, pred, alpha=ALPHA3)
    algn2 = extract_2(sents1train, sents2train, pred, alpha=ALPHA2)
    algnI = intersect_algn(algn2, intersect_algn(algn3, algn4))
    valAER = evaluate_dataset(algnI, dataset.sure[:trainSentCount], dataset.poss[:trainSentCount], verbose=True)[2]
    print('TEST')
    pred = feature_to_sent(sents1test, sents2test, model(dataXtest).cpu().reshape(-1).detach().numpy())
    algn4 = extract_4(sents1test, sents2test, pred, alpha=ALPHA4)
    algn3 = extract_3(sents1test, sents2test, pred, alpha=ALPHA3)
    algn2 = extract_2(sents1test, sents2test, pred, alpha=ALPHA2)
    algnI = intersect_algn(algn2, intersect_algn(algn3, algn4))
    valAER = evaluate_dataset(algnI, alignTestS, alignTestP, verbose=True)[2]

evaluate()