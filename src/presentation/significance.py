#!/bin/python3
from scipy.stats import ttest_1samp
import numpy as np

ENCS = [0.141, 0.144, 0.144, 0.143, 0.144, 0.144, 0.149, 0.142, 0.142, 0.148]
CSEN = [0.146, 0.146, 0.147, 0.146, 0.146, 0.142, 0.143, 0.147, 0.145, 0.146]

avgs = [-(x+y)/2 for x, y in zip(CSEN, ENCS)]

print(np.average(avgs))

tscore, pvalue = ttest_1samp(avgs, popmean=-(0.25-0.1))
# divide by two because we are doing one tailed
print(f'{pvalue/2:.10f}')