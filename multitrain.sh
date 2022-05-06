#!/bin/bash
layers=0
units=128
batches=32
for i in {1..4}; do

    optim="adam"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    optim="nadam"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    optim="sgd"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    optim="nesterov"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    optim="momentum"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    optim="rmsprop"
    xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    xterm -e "python3 train.py $layers 768 $batches $optim 1" &

    echo "$layers $units $batches"
    layers=$((layers+1))
    read -p "Press any key for the next round of $layers layers"


    # optim="adam"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

    # optim="nadam"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

    # optim="sgd"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

    # optim="nesterov"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

    # optim="momentum"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

    # optim="rmsprop"
    # xterm -e "python3 train.py $layers 128 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 256 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 384 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 512 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 640 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 768 $batches $optim 1" &
    # xterm -e "python3 train.py $layers 896 $batches $optim 1" &

done
