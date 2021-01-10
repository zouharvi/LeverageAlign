# Leveraging Neural MT Output for Word Alignment

Word alignment tool for the purposes of a small test experiment in which the output probabilities of a trained MT system are used.

## Prerequisites

- [MarianNMT](https://marian-nmt.github.io) scorer binary, path in `marian_wrap`
- [fast_align](https://github.com/clab/fast_align) binary, path in `fastalign_wrap`
- MarianNMT based [models](http://statmt.org/bergamot/models/). The article uses models from the [Bergamot project](https://github.com/browsermt/students) (not affiliated with this experiment).