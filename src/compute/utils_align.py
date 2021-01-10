from itertools import product

"""
Processing alignment and feature data
"""

def process_algn_line(line):
    if len(line) == 0:
        return []
    return [
        tuple(
            int(tokens)
            for tokens in alignment_tokens.split('-'))
        for alignment_tokens in line.split(' ')
    ]

def process_algn_line_mapper(line, mapper1, mapper2):
    algn = process_algn_line(line)
    return [(mapper1[t1], mapper2[t2]) for t1, t2 in algn]

def algn_to_feature(sents1, sents2, algns):
    data = []
    for sent1, sent2, algn in zip(sents1, sents2, algns):
        sent1 = sent1.split()
        sent2 = sent2.split()
        features = [0.0 for _ in range(len(sent1)*len(sent2))]
        for t1, t2 in algn:
            if len(sent2)*t1 + t2 >= len(sent1)*len(sent2):
                pass
                # raise Exception('Alignment out of bounds')
            else:
                features[len(sent2)*t1 + t2] = 1.0
        data += features
    return data

def sent_to_feature(algn):
    return [item for sublist in algn for item in sublist]

def feature_to_sent(sents1, sents2, features):
    data = []
    for sent1, sent2 in zip(sents1, sents2):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sent_buffer = features[:len(sent1)*len(sent2)]
        features = features[len(sent1)*len(sent2):]
        data.append(sent_buffer)
    return data

def algn_to_feature_sent(sents1, sents2, algns):
    features = algn_to_feature(sents1, sents2, algns)
    sents = feature_to_sent(sents1, sents2, features)
    return sents

def reverse_algn(sents1, sents2, algns):
    """
    Reverse sentence structured alignment scores
    """
    data = []
    for sent1, sent2, algn in zip(sents1, sents2, algns):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sent_buffer = []
        for (t1_i, t1), (t2_i, t2) in product(enumerate(sent1), enumerate(sent2)):
            sent_buffer.append(algn[len(sent1)*t2_i + t1_i])
        assert(len(sent_buffer) == len(algn))
        data.append(sent_buffer)
    return data

def comb_algn_scores(scores1, scores2, comb_method=lambda a, b: a+b):
    data = []
    for sent_scores_1, sent_scores_2 in zip(scores1, scores2):
        data.append([s1 + s2 for s1, s2 in zip(sent_scores_1, sent_scores_2)])
    return data

def intersect_algn(algns1, algns2):
    return [list(set(algn1) & set(algn2)) for algn1, algn2 in zip(algns1, algns2)]