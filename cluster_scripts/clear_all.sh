#!/bin/bash

for EXP in i h ih ihf ihfa ihfar; do
for LANG in csen encs; do
	echo $LANG\_$EXP
	rm -f cluster_logs/$LANG\_$EXP.*log
	rm -f cluster_logs/$LANG\_$EXP.*vlog
	rm -f computed/$LANG\_$EXP\_*.pt
done
done
