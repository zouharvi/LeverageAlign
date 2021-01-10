from itertools import product
import numpy as np

"""
Extracts hard alignment from scores
"""


def extract(sents1, sents2, scores_data, agregation):
    data = []
    for scores, sent1, sent2 in zip(scores_data, sents1, sents2):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sent_buffer = []
        for tok1_i, _tok1 in enumerate(sent1):
            tok1_scores = scores[:len(sent2)]
            scores = scores[len(sent2):]

            tok1_alignments = agregation(tok1_scores)
            # prefix with current token index
            tok1_alignments = [(tok1_i, i) for i in tok1_alignments]

            sent_buffer += tok1_alignments
        data.append(sent_buffer)
    return data


def extract_1(*args):
    return extract_3(*args, alpha=1)


def extract_2(*args, alpha):
    def agregation(tok_scores):
        return [i for i, score in enumerate(tok_scores) if score >= alpha]

    return extract(*args, agregation=agregation)

def extract_3(*args, alpha):
    def agregation(tok_scores):
        boundary = np.max(tok_scores)
        boundary = min(boundary*alpha, -np.inf if alpha == 0 else boundary/alpha)
        return [i for i, score in enumerate(tok_scores) if score >= boundary]

    return extract(*args, agregation=agregation)

def extract_4(*args, alpha):
    def agregation(tok_scores):
        boundary = np.max(tok_scores)
        boundary = min(boundary*alpha, -np.inf if alpha == 0 else boundary/alpha)
        return [i for i, score in enumerate(tok_scores) if score >= boundary]

    return extract_rev(*args, agregation=agregation)


def extract_rev(sents1, sents2, scores_data, agregation):
    data = []
    for scores, sent1, sent2 in zip(scores_data, sents1, sents2):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sent_buffer = []
        for tok2_i, _tok2 in enumerate(sent2):
            tok2_scores = [scores[i*len(sent2)+tok2_i] for i in range(len(sent1))]

            tok2_alignments = agregation(tok2_scores)
            # suffix with current token index
            tok2_alignments = [(i, tok2_i) for i in tok2_alignments]

            sent_buffer += tok2_alignments
        data.append(sent_buffer)
    return data