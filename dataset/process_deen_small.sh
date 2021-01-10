#!/bin/env bash

# generate two files (data_deen_small.algn and data_deen_small.sent) out of German-English_WordAlignment dataset folder

paste German-English_WordAlignment/test.de German-English_WordAlignment/test.en | sed -e 's/\t/ ||| /g' > data_deen_small.sent
cat German-English_WordAlignment/de-en.wa | sed -e "s/$/ ||| /" > data_deen_small.algn

echo -n "Sentences: "
cat German-English_WordAlignment/test.de | wc -l

echo -n "Tokens DE: "
cat German-English_WordAlignment/test.de | tr " " "\n" | wc -l

echo -n "Tokens EN: "
cat German-English_WordAlignment/test.en | tr " " "\n" | wc -l