import sentencepiece as spm
from os.path import expanduser

"""
Invokes SentencePiece
"""

def sent_mapper(sent, sent_encoded):
    sent = sent.split()
    mapping = [0]*(len(sent_encoded) + len([x for x in sent_encoded if x == '▁']) +1)
    srcptr = -1
    for encptr, tok_encoded in enumerate(sent_encoded):
        if tok_encoded[0] == '▁' and len(tok_encoded) > 0:
            # new word
            srcptr += 1
        mapping[encptr] = srcptr
    return mapping


def dataset_mapper(sents, vocab):
    # expanduser, because sentence piece does not know how to deal with ~
    vocabmodel = spm.SentencePieceProcessor(model_file=expanduser(vocab))
    sents_encoded = vocabmodel.encode(sents, out_type=str)
    return [sent_mapper(sent, sent_encoded) for sent, sent_encoded in zip(sents, sents_encoded)]

def dataset_encoder(sents, vocab):
    # expanduser, because sentence piece does not know how to deal with ~
    vocabmodel = spm.SentencePieceProcessor(model_file=expanduser(vocab))
    sents_encoded = vocabmodel.encode(sents, out_type=str)
    return sents_encoded