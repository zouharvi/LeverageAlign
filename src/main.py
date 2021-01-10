#!/bin/python3

"""
Temporary playground
"""

from models import *
from marian_wrap import attention
from align_extractor import extract_1, extract_2, extract_3
from data_loader import *
from evaluator import *

data_encs = DatasetLoader('en', 'cs', maxlen=None)
sents1 = data_encs.sents1
sents2 = data_encs.sents2

alignment = attention(
    'en', 'de',
    ['Hello there', 'Hallo da'], ['Hello', 'HalloHalloHalloHallo'],
    threshold='1'
)
for algn in alignment:
    print(algn)