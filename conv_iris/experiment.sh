#!/bin/bash
i=1
while [ "$i" -le 50 ] 
do
    python make_names_list.py 
    ./create_data.sh 2>/dev/null 
    ./train.sh 2> result.txt 
    cat result.txt | grep "Test score #0" | awk '{print($8)}'
    i=$((i + 1))
done