import subprocess
from utils_align import process_algn_line_mapper
from sentencepiece_wrap import dataset_mapper, dataset_encoder

"""
Invoke MarianNMT scoring and alignment
"""

MARIAN_CONFIGS = {
    'csen': {
        'model': '~/bergamot-students/csen/csen.student.tiny11/model.bin',
        'vocab': '~/bergamot-students/csen/csen.student.tiny11/vocab.spm',
    },
    'encs': {
        'model': '~/bergamot-students/csen/encs.student.tiny11/model.bin',
        'vocab': '~/bergamot-students/csen/encs.student.tiny11/vocab.spm',
    },
    'ende': {
        'model': '~/bergamot-students/deen/ende.student.tiny11/model.npz',
        'vocab': '~/bergamot-students/deen/ende.student.tiny11/vocab.deen.spm',
    },
}

UNKNOWN_TOKEN = '<unk>'
MARIAN_SCORER_PATH = '~/bin/marian-dev/build/marian-scorer'


def scorer(lang1, lang2, sents1, sents2, tmp_prefix='/tmp/text.'):
    """
    Returns [(sent prob, [target token prob])] log probabilities. Textual input should 
    be lists of strings 
    """
    assert(len(sents1) == len(sents2))

    config = MARIAN_CONFIGS[(lang1+lang2).lower()]
    filelang1 = tmp_prefix+lang1
    filelang2 = tmp_prefix+lang2

    with open(filelang1, 'w') as f:
        f.write('\n'.join(sents1)+'\n')
    with open(filelang2, 'w') as f:
        f.write('\n'.join(sents2)+'\n')

    command = [
        MARIAN_SCORER_PATH,
        '-m', config['model'],
        '-v', config['vocab'], config['vocab'],
        '-t', filelang1, filelang2,
        '--word-scores',
        # '--cpu-threads', '4',
        '--devices', '0 1 2 3',
    ]

    marianProcess = subprocess.Popen(' '.join(
        command), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = marianProcess.communicate()

    scorerData = stdout.rstrip('\n').split('\n')
    assert(len(scorerData) == len(sents1))
    scorerData = [line.split(' ||| WordScores= ') for line in scorerData]
    scorerData = [(float(sentp), [float(wordprob) for wordprob in wordprobs.split()])
                  for sentp, wordprobs in scorerData]
    return scorerData


def attention(lang1, lang2, sents1, sents2, aggregation_method=max, tmp_prefix='/tmp/text.'):
    """
    Returns [(sent prob, [target token prob])] log probabilities. Textual input should 
    be lists of strings 
    """
    assert(len(sents1) == len(sents2))

    config = MARIAN_CONFIGS[(lang1+lang2).lower()]
    filelang1 = tmp_prefix+lang1
    filelang2 = tmp_prefix+lang2

    with open(filelang1, 'w') as f:
        f.write('\n'.join(sents1)+'\n')
    with open(filelang2, 'w') as f:
        f.write('\n'.join(sents2)+'\n')

    command = [
        MARIAN_SCORER_PATH,
        '-m', config['model'],
        '-v', config['vocab'], config['vocab'],
        '-t', filelang1, filelang2,
        # '--cpu-threads', '4',
        '--devices', '0 1 2 3',
        '--alignment', 'soft',
    ]

    marianProcess = subprocess.Popen(' '.join(
        command), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = marianProcess.communicate()

    sents1mapper = dataset_mapper(sents1, config['vocab'])
    sents2mapper = dataset_mapper(sents2, config['vocab'])
    sents1encoder = dataset_encoder(sents1, config['vocab'])
    sents2encoder = dataset_encoder(sents2, config['vocab'])
    sent_scores_raw = [line.split(' ||| ')[1]
                       for line in stdout.rstrip('\n').split('\n')]

    alignmentData = []
    for sent_i, (sent1, sent2, line) in enumerate(zip(sents1, sents2, sent_scores_raw)):
        sent1 = sent1.split()
        sent2 = sent2.split()
        sent1mapper = sents1mapper[sent_i]
        sent2mapper = sents2mapper[sent_i]
        scores = [[] for _ in range(len(sent1)*len(sent2))]
        for t2_i, t1t2_data in list(enumerate(line.split(' ')))[:-1]:
            for t1_i, t1t2_raw in list(enumerate(t1t2_data.split(',')))[:-1]:
                scores[sent1mapper[t1_i] *
                       len(sent2) + sent2mapper[t2_i]].append(float(t1t2_raw))
        scores = [aggregation_method(x) for x in scores]
        alignmentData.append(scores)

    assert(len(alignmentData) == len(sents1))
    return alignmentData


def aligner_stock(lang1, lang2, sents1, sents2, threshold='1', tmp_prefix='/tmp/text.'):
    """
    Returns [(sent prob, [target token prob])] log probabilities. Textual input should 
    be lists of strings 
    """
    assert(len(sents1) == len(sents2))

    config = MARIAN_CONFIGS[(lang1+lang2).lower()]
    filelang1 = tmp_prefix+lang1
    filelang2 = tmp_prefix+lang2

    with open(filelang1, 'w') as f:
        f.write('\n'.join(sents1)+'\n')
    with open(filelang2, 'w') as f:
        f.write('\n'.join(sents2)+'\n')

    command = [
        MARIAN_SCORER_PATH,
        '-m', config['model'],
        '-v', config['vocab'], config['vocab'],
        '-t', filelang1, filelang2,
        # '--cpu-threads', '4',
        '--devices', '0 1 2 3',
        '--alignment', threshold,
    ]

    marianProcess = subprocess.Popen(' '.join(
        command), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = marianProcess.communicate()

    sents1mapper = dataset_mapper(sents1, config['vocab'])
    sents2mapper = dataset_mapper(sents2, config['vocab'])

    alignmentData = stdout.rstrip('\n').split('\n')
    assert(len(alignmentData) == len(sents1))
    alignmentData = [line.split(' ||| ')[1] for line in alignmentData]
    alignmentData = [process_algn_line_mapper(
        line, sents1mapper[i], sents2mapper[i]) for i, line in enumerate(alignmentData)]

    return alignmentData
