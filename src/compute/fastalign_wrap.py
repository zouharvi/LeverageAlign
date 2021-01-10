import subprocess
from utils_align import  process_algn_line

"""
Invokes fast_align
"""


FAST_ALIGN_PATH = '~/bin/fast_align/build/fast_align'


def fast_align(sents1, sents2, sentsTrain, tmp_prefix='/tmp/text_parallel'):
    """
    Performs word alignment on sourceText to targetText using fast_align.
    The output is in the Pharaoh format.
    """

    with open(tmp_prefix, 'w') as f:
        # write test sentences
        out = [s1 + ' ||| ' + s2 for s1, s2 in zip(sents1, sents2)]
        f.write('\n'.join(out))
        # write train sentences
        out = [s1 + ' ||| ' + s2 for s1, s2 in sentsTrain]
        f.write('\n'.join(out))
        
    command = [
        FAST_ALIGN_PATH,
        '-d', '-o', '-v',
        '-i', tmp_prefix,
    ]

    alignProcess = subprocess.Popen(' '.join(
        command), encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = alignProcess.communicate()

    data = [
        process_algn_line(line)
        for line in stdout.split('\n')[:len(sents1)] if len(line) > 0
    ]
    return data
