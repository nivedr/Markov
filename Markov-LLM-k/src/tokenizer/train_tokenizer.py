import numpy as np
import itertools
import collections
import math

import Tokenizer
import BPE
# import LZW
# import SplitBPE
import sys
sys.path.append('optim')
import utils

import pickle
import yaml
from pathlib import Path

# Instantiate tokenizer model
def train_tokenizer(tokenizer, max_dict_size, p, q, order, generator, dataset_size, extra_args):
    print(f'Training tokenizer: {tokenizer}')
    dataset, _ = utils.get_batch(p, q, order, seq_length=1, batch_size=dataset_size, generator=generator, extra_args=extra_args, device=extra_args.device)
    if model == 'Character':
        tokenizer = Tokenizer.Tokenizer()
        tokenizer.learn_dict()
    elif model == 'BPE':
        tokenizer = BPE.BPE(kappa=kappa, max_dict_size=max_dict_size, dataset=dataset)
        tokenizer.learn_dict()
    else:
        raise ValueError('model must be either Character, LZW, BPE or SplitBPE')
    return tokenizer