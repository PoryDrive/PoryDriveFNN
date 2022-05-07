#!/bin/bash
layers=0
units=384
batches=32
for i in {0..4}; do

    optim="adam"
    xterm -e "python3 train.py $layers $units $batches $optim 1" &

    optim="nesterov"
    xterm -e "python3 train.py $layers $units $batches $optim 1" &

    echo "$layers $units $batches"
    layers=$((layers+1))
    read -p "Press any key for the next round of $layers layers"

done
