#!/bin/bash

OUT_FILE="computed/fast_align_csen.a"

while read line; do
   line=$(echo $line | sed "s/|||/|/g")
   l1=$(echo $line | cut -d'|' -f1 | head -n 1)
   l2=$(echo $line | cut -d'|' -f2 | head -n 1)
   curl 'https://quest.ms.mff.cuni.cz/ptakopet-mt380/align/en-cs' -H 'Content-Type: application/json' --data "{\"src_text\":\"$l2\",\"trg_text\":\"$l1\"}"
done < dataset/data_encs.sent > $OUT_FILE;

cat $OUT_FILE | grep -v "<h1>Bad Request</h1>" | grep -v "<p>The browser (or proxy) sent a request that this server could not understand.</p>" | grep -v '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">' > /tmp/algn
mv /tmp/algn $OUT_FILE 