import numpy as np

"""
Compute alignment evaluation metrics
"""


def evaluate_sent_prec(algnA, algnP):
    return len(set(algnA) & set(algnP))


def evaluate_sent_recl(algnA, algnS):
    return len(set(algnA) & set(algnS))


def evaluate_dataset(algnsA, algnsS, algnsP, verbose=True):
    precs = []
    recls = []
    aers = []
    for algnA, algnS, algnP in zip(algnsA, algnsS, algnsP):
        prec_top = evaluate_sent_prec(algnA, algnS)
        rec_top = evaluate_sent_recl(algnA, algnS)
        precs.append(0 if len(algnA) == 0 else prec_top/len(algnA))
        recls.append(rec_top / len(algnS))
        aers.append(1-(prec_top + rec_top)/(len(algnA) + len(algnS)))

    if verbose:
        print(f'precision: {np.average(precs):.3f}')
        print(f'recall:    {np.average(recls):.3f}')
        print(f'AER:       {np.average(aers):.3f}')
    return np.average(precs), np.average(recls), np.average(aers)
