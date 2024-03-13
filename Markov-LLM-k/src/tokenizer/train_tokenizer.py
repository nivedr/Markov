import numpy as np
import itertools
import collections
import math

import Tokenizer
import BPE
# import LZW
# import SplitBPE
import utils_tokenizer
import pickle
import yaml
from pathlib import Path

# Read config
# with open("config.yml", 'r') as stream:
#     config = yaml.safe_load(stream)

# train = config['train']['tokenizer']['train']
# if not train:
# 	print("No tokenizer was trained.")
# 	exit()
# models = config['default']['models']
# save = config['default']['save']
# delta = config['default']['delta']
# n_sample_train = int(config['train']['tokenizer']['n_sample_train'])
# n_trial_train_tok = int(config['train']['tokenizer']['n_trial_train'])
# max_dict_size_list = [int(a) for a in config['default']['max_dict_size_list']]

# kappa = int(config['train']['tokenizer']['kappa'])
# l_split = int(config['train']['tokenizer']['l_split'])
# epsilon = float(config['train']['tokenizer']['epsilon'])

# print(f"Tokenizer models to be trained: {models}")
# print(f"Save model: {config['default']['save']}")
# print(f"Number of training samples for the tokenizer: {n_sample_train}")
# print(f"Number of training trial runs: {n_trial_train_tok}")
# print(f"List of maximum dictionary sizes: {max_dict_size_list}")

# if 'BPE' in models or 'SplitBPE' in models:
# 	assert kappa, "Minimum merge frequency, kappa, not instantiated"
# 	print(f"BPE/SplitBPE parameters:")
# 	print(f"\tMinimum merge frequency: {kappa}")
# 	if 'SplitBPE' in models:
# 		assert l_split, "Chunk length for SplitBPE, l_split, not instantiated"
# 		assert epsilon, "greedy threshold for SplitBPE, epsilon, not instantiated"
# 		print(f"SplitBPE parameters:")
# 		print(f"\tChunk length: {l_split}")
# 		print(f"\tGreedy threshold: {epsilon}")


# Instantiate tokenizer model
def train_tokenizer(tokenizer, max_dict_size, p, q, order, dataset_size):
	print(f'Training tokenizer: {tokenizer}')
	dataset, _ = utils.get_batch(p, q, order, seq_length=1, batch_size=dataset_size, generator=generator, extra_args=extra_args, device='cpu')
	if model == 'Character':
		tokenizer = Tokenizer.Tokenizer()
		tokenizer.learn_dict()
	elif model == 'BPE':
		tokenizer = BPE.BPE(kappa=kappa, max_dict_size=max_dict_size, dataset=dataset)
		tokenizer.learn_dict()
	# elif model == 'LZW':
	# 	tokenizer = LZW.LZW(max_dict_size=max_dict_size, dataset=dataset)
	# 	tokenizer.learn_dict()
	# elif model == 'SplitBPE':
	# 	tokenizer = SplitBPE.SplitBPE(kappa=kappa, l_split=l_split, epsilon=epsilon,
	# 		max_dict_size=max_dict_size, dataset=dataset)
	# 	tokenizer.learn_dict()
	else:
		raise ValueError('model must be either Character, LZW, BPE or SplitBPE')
	return tokenizer


# Commit tokenizer to file

# for trial_tok in range(n_trial_train_tok):
# 	print(f'Trial number: {trial_tok}')
# 	for model in models:
# 		for max_dict_size in max_dict_size_list:
# 			print(f'{model} and {max_dict_size}')
# 			tokenizer = train_tokenizer(model, max_dict_size)
# 			if save:
# 				folder_path = Path.cwd() / f'./models/{model}/n_sample_train={n_sample_train}/max_dict_size={max_dict_size}/trial_tok={trial_tok}'
# 				folder_path.mkdir(exist_ok=True, parents=True)
# 				file_name = f'tokenizer.pickle'
# 				with (folder_path / file_name).open('wb') as file:
# 					pickle.dump(tokenizer, file)
# 					file.close()


