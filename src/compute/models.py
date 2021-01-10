from itertools import product
from marian_wrap import scorer as marian_scorer
from marian_wrap import UNKNOWN_TOKEN

"""
Each model's relevance result is a list of sentence lists
with tokens, as produced by the itertools.product(sent1, sent2)
"""


def model_m1(lang1, lang2, sents1, sents2):
    compute_buffer1 = []
    compute_buffer2 = []
    for sent1, sent2 in zip(sents1, sents2):
        for tok1, tok2 in product(sent1.split(), sent2.split()):
            compute_buffer1.append(tok1)
            compute_buffer2.append(tok2)

    computed = marian_scorer(lang1, lang2, compute_buffer1, compute_buffer2)
    # take only sentence scores
    computed = [sent for sent, word in computed]

    data = []
    for sent1, sent2 in zip(sents1, sents2):
        sent1 = sent1.split()
        sent2 = sent2.split()
        data.append(computed[:len(sent1)*len(sent2)])
        computed = computed[len(sent1)*len(sent2):]
    return data


def _sent_modifier_delete(index, sent):
    return [token for i, token in enumerate(sent) if i != index]


def _sent_modifier_substitute(index, sent):
    return [token if i != index else UNKNOWN_TOKEN for i, token in enumerate(sent)]


def model_m2a(*args):
    return model_m2(*args, sent_func=_sent_modifier_delete)


def model_m2b(*args):
    return model_m2(*args, sent_func=_sent_modifier_substitute)


def model_m2(lang1, lang2, sents1, sents2, sent_func):
    # only take token prob outputs
    data_sent = [
        word
        for sent, word in marian_scorer(lang1, lang2, sents1, sents2)
    ]
    compute_buffer1 = []
    compute_buffer2 = []
    for sent_i, (sent1, sent2) in enumerate(zip(sents1, sents2)):
        sent1 = sent1.split()
        for (tok1_i, tok1) in enumerate(sent1):
            # construct new target sentence
            new_sent1 = ' '.join(sent_func(tok1_i, sent1))
            compute_buffer1.append(new_sent1)
            compute_buffer2.append(sent2)

    computed = marian_scorer(lang1, lang2, compute_buffer1, compute_buffer2)

    data = []
    for sent_i, (sent1, sent2) in enumerate(zip(sents1, sents2)):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sentence_buffer = []
        for (tok1_i, tok1), (tok2_i, tok2) in product(enumerate(sent1), enumerate(sent2)):
            # take target token prob
            # compute tok1 - tok2 score
            score = data_sent[sent_i][tok2_i] - computed[tok1_i][1][tok2_i]
            sentence_buffer.append(score)
        # discard used data
        computed = computed[len(sent1):]
        data.append(sentence_buffer)
    return data


def model_m3aa(*args):
    return model_m3(
        *args,
        sent1_func=_sent_modifier_delete,
        sent2_func=_sent_modifier_delete,
    )


def model_m3ab(*args):
    return model_m3(
        *args,
        sent1_func=_sent_modifier_delete,
        sent2_func=_sent_modifier_substitute,
    )


def model_m3ba(*args):
    return model_m3(
        *args,
        sent1_func=_sent_modifier_substitute,
        sent2_func=_sent_modifier_delete,
    )


def model_m3bb(*args):
    return model_m3(
        *args,
        sent1_func=_sent_modifier_substitute,
        sent2_func=_sent_modifier_substitute,
    )


def model_m3(lang1, lang2, sents1, sents2, sent1_func, sent2_func):
    compute_buffer1 = []
    compute_buffer2 = []
    for sent_i, (sent1, sent2) in enumerate(zip(sents1, sents2)):
        sent1 = sent1.split()
        sent2 = sent2.split()
        for (tok1_i, tok1), (tok2_i, tok2) in product(enumerate(sent1), enumerate(sent2)):
            # construct new target sentence
            new_sent1 = ' '.join(sent1_func(tok1_i, sent1))
            new_sent2 = ' '.join(sent2_func(tok2_i, sent2))
            compute_buffer1.append(new_sent1)
            compute_buffer2.append(new_sent2)

    computed = marian_scorer(lang1, lang2, compute_buffer1, compute_buffer2)

    data = []
    for sent_i, (sent1, sent2) in enumerate(zip(sents1, sents2)):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sentence_buffer = []
        for (tok1_i, tok1), (tok2_i, tok2) in product(enumerate(sent1), enumerate(sent2)):
            # take sentence prob as score
            sentence_buffer.append(computed.pop(0)[0])
        data.append(sentence_buffer)
    return data
