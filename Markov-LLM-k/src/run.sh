#!/bin/bash

for i in $(seq 2 4);
do
	for j in $(seq 2 4);
	do
		python main.py --p 0.8 --q 0.8 --n_layer $i --n_head 3 --n_embd 30 --sequence_length 200 --order $j --max_dict_size 2 --data_in_ram --iterations 3000 --batch_size 8 --acc_steps 1 --transition  random --wandb;
		rm -rf ./exps/markov/base/base_lr0.002_bs8x1_1nodes_seed=0
		cp ~/projects/Markov/Markov-LLM-k/src/val-loss-dump.pickle ~/projects/Markov/Markov-LLM-k/src/exps/order-dynamics/;
		mv ~/projects/Markov/Markov-LLM-k/src/exps/order-dynamics/val-loss-dump.pickle ~/projects/Markov/Markov-LLM-k/src/exps/order-dynamics/layer_$i-order_$j.pickle;
	done
done

