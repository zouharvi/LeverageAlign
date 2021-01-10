#!/bin/bash

mkdir -p cluster_logs
mkdir -p tmp

EXPS=$1
for EXP in $EXPS; do
for LANG in csen encs; do
	echo -en "\nSubmitting $LANG""_""$EXP"
	echo -e "#!/bin/bash\npython3 src/compute/aggregate_train.py -l $LANG -m $EXP" > tmp/$LANG\_$EXP.sh
	read _
	qsub -q 'gpu*' -cwd -o cluster_logs/$LANG\_$EXP.log -j y -pe smp 4 -l gpu=1,gpu_ram=4G tmp/$LANG\_$EXP.sh
done
done