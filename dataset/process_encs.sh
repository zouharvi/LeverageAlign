#!/bin/env bash

# generate two files (data_csen.algn and data_csen.sent) out of CzEnAli_1.0 dataset folder

cat CzEnAli_1.0/data/**/*.wa | grep -e '<english>'  | sed -e 's/\s*<[^>]*>//g' > /tmp/text.en
cat CzEnAli_1.0/data/**/*.wa | grep -e '<czech>'    | sed -e 's/\s*<[^>]*>//g' > /tmp/text.cs
cat CzEnAli_1.0/data/**/*.wa | grep -e '<sure>'     | sed -e 's/\s*<[^>]*>//g' > /tmp/text.sure
cat CzEnAli_1.0/data/**/*.wa | grep -e '<possible>' | sed -e 's/\s*<[^>]*>//g' > /tmp/text.poss

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

paste /tmp/text.en   /tmp/text.cs   | sed -e 's/\t/ ||| /g' > tmp1
paste /tmp/text.sure /tmp/text.poss | sed -e 's/\t/ ||| /g' > tmp2

shuf --random-source=<(get_seeded_random 64) tmp1 > data_encs.sent &
shuf --random-source=<(get_seeded_random 64) tmp2 > data_encs.algn &
wait

rm tmp1 tmp2

echo -n "Sentences: "
cat /tmp/text.en | wc -l

echo -n "Tokens CS: "
cat /tmp/text.cs | tr " " "\n" | wc -l

echo -n "Tokens EN: "
cat /tmp/text.en | tr " " "\n" | wc -l