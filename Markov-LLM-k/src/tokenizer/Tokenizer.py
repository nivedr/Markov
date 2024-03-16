import numpy as np
import itertools
import collections
import math


class Tokenizer(object):
    _dset_tok = []

    DS = []
    max_dict_size = -1
    dict_size = 0
    dictionary = []


    def __init__(self, max_dict_size=-1):
        self.max_dict_size = max_dict_size

    def learn_dict(self):
        self.dictionary = range(2)

    def encode(self, string):
        "Encode string"
        return string

    def decode(self, string):
        "Decode string"
        return string

    def count(self, string, tok):
        return collections.Counter(string)[tok]

    def encode_batch(self, batch):
        return batch

    def all_counts(self, string):
        return collections.Counter(string)