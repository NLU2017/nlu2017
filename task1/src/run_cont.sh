#!/usr/bin/env bash

python3 train.py --task="C" --is_training=False --is_eval=False --is_cont=True --lstm_size=1024 --learning_rate=0.01 --log_dir=../runs/ --model_name=1493922093
#python3 train.py --task "C" --num_epochs 1 --lstm_size 1024 --allow_batchnorm True --dropout_rate 0.5 --model_name "C__BN_drop"
