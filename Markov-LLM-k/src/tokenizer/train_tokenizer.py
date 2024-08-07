import numpy as np
import itertools
import collections
import math

import Tokenizer
import BPE
import LZW
# import SplitBPE
import sys
sys.path.append('optim')
import utils

import pickle
import yaml
from pathlib import Path

# Instantiate tokenizer
def train_tokenizer(tokenizer, max_dict_size, P, order, vocab_size, generator, dataset_size, extra_args):
    print(f'Training tokenizer: {tokenizer}')
    dataset, _ = utils.get_batch(P, order, vocab_size, seq_length=dataset_size, batch_size=1, generator=generator, extra_args=extra_args, device='cpu')
    print(dataset)
    
    if tokenizer == 'Character':
        tokenizer = Tokenizer.Tokenizer()
        tokenizer.learn_dict()
    elif tokenizer == 'BPE':
        tokenizer = BPE.BPE(kappa=10, max_dict_size=max_dict_size)
        tokenizer.learn_dict(dset=dataset[0])
    elif tokenizer == 'LZW':
        tokenizer = LZW.LZW(max_dict_size=max_dict_size)
        tokenizer.learn_dict(dset=dataset[0])
    else:
        raise ValueError('Tokenizer must be either Character, LZW or BPE')
    return tokenizer
