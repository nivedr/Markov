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
def train_tokenizer(tokenizer, max_dict_size, p, q, order, generator, dataset_size, extra_args, r=0, s=0):
    print(f'Training tokenizer: {tokenizer}')
    dataset, _ = utils.get_batch(p, q, order, seq_length=1, batch_size=dataset_size, generator=generator, extra_args=extra_args, device='cpu', r=r, s=s)
    if tokenizer == 'Character':
        tokenizer = Tokenizer.Tokenizer()
        tokenizer.learn_dict()
    elif tokenizer == 'BPE':
        tokenizer = BPE.BPE(kappa=10, max_dict_size=max_dict_size)
        tokenizer.learn_dict(dset=dataset)
    elif tokenizer == 'LZW':
        tokenizer = LZW.LZW(max_dict_size=max_dict_size)
        tokenizer.learn_dict(dset=dataset)
    else:
        raise ValueError('Tokenizer must be either Character, LZW or BPE')
    return tokenizer