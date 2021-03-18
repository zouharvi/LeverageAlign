#!/bin/python3
from scipy.stats import ttest_1samp
import numpy as np

ENCS = [0.141, 0.144, 0.144, 0.143, 0.144, 0.144, 0.149, 0.142, 0.142, 0.148]
CSEN = [0.146, 0.146, 0.147, 0.146, 0.146, 0.142, 0.143, 0.147, 0.145, 0.146]

tscore, pvalue = ttest_1samp(ENCS+CSEN, popmean=(0.25-0.1), alternative="less")
print(f'{pvalue:.10f}')