#!/usr/bin/env bash
STARTS=(0 10 20 30 40 50 60 70 80 90)
ENDS=(10 20 30 40 50 60 70 80 90 100)

set -x #echo on

for i in `seq 0 9`;
 do
    START=${STARTS[$i]};
    END=${ENDS[$i]};
    BID=$(($i + 1))
    python rev2/gs/gs_acid_attack.py resnet ~/Data/intattack/rev2/resnet_gs/fold_1.npz ~/Data/intattack/rev2/resnet_acid_gs_v2/fold1_b$BID.npz --begin $START --end $END;
done;