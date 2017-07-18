#!/bin/bash

dirlist=`ls ../data/`
echo $dirlist
for dirpath in $dirlist;do
    if [ -d '../data/'$dirpath ];then
        datapath="../data/$dirpath/mix_60000.txt"
        echo $datapath
        python model.py $datapath > ../log/"$dirpath.log"
        if [[ $? -ne 0 ]]; then
            exit 1
        fi
    fi
done
