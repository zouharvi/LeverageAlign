import os
from pathlib import Path
from utils_align import process_algn_line
import pickle
import numpy as np

"""
Load alignment corpora + score  save utils
"""

class DatasetLoader():
    def _load_data(self, lang1, lang2, suff, maxlen, noalgn, nosub):
        # load sentences
        with open(f'dataset/data_{lang1}{lang2}{suff}.sent', 'r') as f:
            self.sent = [
                tuple(line.split(' ||| ')) for line in f.read().replace('\ufeff', '').split('\n')
            ]

        if not noalgn:
            # load sure alignments
            with open(f'dataset/data_{lang1}{lang2}{suff}.algn', 'r') as f:
                self.sure = [
                    process_algn_line(line.split(' ||| ')[0]) for line in f.read().strip('\n').split('\n')
                ]

            # load possible alignments
            with open(f'dataset/data_{lang1}{lang2}{suff}.algn', 'r') as f:
                self.poss = [
                    process_algn_line(line.split(' ||| ')[1]) for line in f.read().strip('\n').split('\n')
                ]

            data = [
                (sent, sure, poss)
                for (sent, sure, poss) in zip(self.sent, self.sure, self.poss)
                if (sure or poss) and (maxlen is None or len(sent[0].split()) <= maxlen)
            ]
            def substract_one(alignment_line):
                if nosub:
                    return alignment_line
                return [(x[0]-1, x[1]-1) for x in alignment_line]

            self.sent = [sent for (sent, _, _) in data]
            self.sure = [substract_one(sure) for (_, sure, _) in data]
            self.poss = [substract_one(poss) for (_, _, poss) in data]

    def reverse(self, noalgn):
        self.sent = [(s[1], s[0]) for s in self.sent if len(s) == 2]
        if not noalgn:
            self.sure = [[(a2, a1) for (a1, a2) in alignment] for alignment in self.sure]
            self.poss = [[(a2, a1) for (a1, a2) in alignment] for alignment in self.poss]

    def __init__(self, lang1, lang2, suff='', maxlen=None, noalgn=False, nosub=False):
        if os.path.isfile(f'dataset/data_{lang1}{lang2}{suff}.sent'):
            # original languages
            self._load_data(lang1, lang2, suff, maxlen, noalgn, nosub)
        elif os.path.isfile(f'dataset/data_{lang2}{lang1}{suff}.sent'):
            # switch languages
            self._load_data(lang2, lang1, suff, maxlen, noalgn, nosub)
            self.reverse(noalgn)
        else:
            raise Exception(
                'Neither ' +
                f'"dataset/data_{lang1}{lang2}{suff}.sent" nor ' +
                f'"dataset/data_{lang2}{lang1}{suff}.sent" exist'
            )
        self.sent = [x for x in self.sent if len(x) == 2]
        self.sents1 = [s1 for (s1, s2) in self.sent]
        self.sents2 = [s2 for (s1, s2) in self.sent]
        del self.sent

        if not noalgn:
            self.poss = [ set(p) | set(s) for p, s in zip(self.poss, self.sure) ]
            self.sure = [ set(s) for s in self.sure ]

def save_any(scores, path, prefix='computed/'):
    # save
    Path(os.path.split(prefix+path)[0]).mkdir(parents=True, exist_ok=True)
    with open(prefix+path, 'wb') as f:
        pickle.dump(scores, f)

def save_features(scores, path, prefix='computed/'):
    # assert the object is valid
    assert(type(scores) is list)
    for val in scores:
        assert(type(val) is float)

    save_any(scores, path, prefix)

def save_scores(scores, path, prefix='computed/'):
    # assert the object is valid
    assert(type(scores) is list)
    for sent_scores in scores:
        assert(type(sent_scores) is list)
        for val in sent_scores:
            assert(type(val) is float or type(val) is np.float64)

    save_any(scores, path, prefix)

def load_scores(path, prefix='computed/'):
    with open(prefix+path, 'rb') as f:
        return pickle.load(f)

def load_any(path, prefix='computed/'):
    with open(prefix+path, 'rb') as f:
        return pickle.load(f)