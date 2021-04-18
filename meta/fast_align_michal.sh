#!/bin/bash

OUT_FILE="fast_align_encs.a"
json_escape () {
    printf '%s' "$1" 
}

while read line; do
   line=$(echo $line | sed "s/|||/|/g")
   l1=$(echo $line | cut -d'|' -f1 | head -n 1 | sed "s/\\\"/'/g")
   l2=$(echo $line | cut -d'|' -f2 | head -n 1 | sed "s/\\\"/'/g")
   # echo $l2
   curl 'https://quest.ms.mff.cuni.cz/ptakopet-mt380/align/en-cs' -H 'Content-Type: application/json' --data "{\"src_text\":\"$l1\",\"trg_text\":\"$l2\"}"
done < data_encs.sent > $OUT_FILE;

# shrink bad requests to just one line
cat $OUT_FILE | grep -v "<h1>Bad Request</h1>" | grep -v "<p>The browser (or proxy) sent a request that this server could not understand.</p>" | grep -v '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">' > tmp_algn
mv tmp_algn $OUT_FILE 

python3 compute_aer.py data_encs.algn $OUT_FILE --gold-index-one