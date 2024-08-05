#!/bin/bash


python main.py --p 0.8 --q 0.8 --n_layer 2 --n_head 2 --n_embd 16 --sequence_length 200 --order 2 --data_in_ram --iterations 20000 --batch_size 8 --acc_steps 1 --transition random
mv /data/nived/Markov/Markov-LLM-k/src/val-loss-dump.pickle /data/nived/Markov/Markov-LLM-k/src/interpolation_2L_2H/interp_100_iter_$n.rm -r /data/nived/Markov/Markov-LLM-k/src/exps/markov/base

