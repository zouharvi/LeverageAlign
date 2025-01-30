# Leverage Align

[![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://ufal.mff.cuni.cz/pbml/116/art-zouhar-pylypenko.pdf)

Repository for [Leveraging Neural Machine Translation for Word Alignment](https://ufal.mff.cuni.cz/pbml/116/art-zouhar-pylypenko.pdf), which was published in Prague Bulleting of Mathematical Linguistics.

> The most common tools for word-alignment rely on a large amount of parallel sentences, which are then usually processed according to one of the IBM model algorithms. The training data is, however, the same as for machine translation (MT) systems, especially for neural MT (NMT), which itself is able to produce word-alignments using the trained attention heads. This is convenient because word-alignment is theoretically a viable byproduct of any attention-based NMT, which is also able to provide decoder scores for a translated sentence pair.
> 
> We summarize different approaches on how word-alignment can be extracted from alignment scores and then explore ways in which scores can be extracted from NMT, focusing on inferring the word-alignment scores based on output sentence and token probabilities. We compare this to the extraction of alignment scores from attention. We conclude with aggregating all of the sources of alignment scores into a simple feed-forward network which achieves the best results when combined alignment extractors are used. 

Cite as:

```
@article{zouhar2021leveraging,
  title={Leveraging Neural Machine Translation for Word Alignment},
  author={Zouhar, Vil{\'e}m and Pylypenko, Daria},
  journal={The Prague Bulletin of Mathematical Linguistics},
  url={https://ufal.mff.cuni.cz/pbml/116/art-zouhar-pylypenko.pdf},
  number={116},
  pages={43--61},
  year={2021}
}
```

## Prerequisites

- [MarianNMT](https://marian-nmt.github.io) scorer binary, path in `marian_wrap`
- [fast_align](https://github.com/clab/fast_align) binary, path in `fastalign_wrap`
- MarianNMT based [models](http://statmt.org/bergamot/models/). The article uses models from the [Bergamot project](https://github.com/browsermt/students) (not affiliated with this experiment).
