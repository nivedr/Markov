import numpy as np
import itertools
import collections
import math
import Tokenizer
import torch
from collections import deque


class BPE(Tokenizer.Tokenizer):

    kappa = 10
	

    def __init__(self, kappa, max_dict_size):
        self.kappa = kappa
        super().__init__(max_dict_size=max_dict_size)


    def apply(self, string, tok):
        t1,t2 = self.DS[tok-2]
        print(f't1 = {t1}')
        print(f't2 = {t2}')
        mask = string[:-1].copy()

        if t1 != t2:
            mask = (string[:-1] == t1) and (string[1:] == t2)
        else:
            for j in range(len(string)-1):
                mask[j] = (string[j] == string[j+1]) and (j==0 or mask[j-1] != True)

        string[:-1][mask] = tok
    
        mask0 = deque(mask)
        mask0.appendleft(0)
        mask = np.array(list(mask0))

        return string[(mask[:-1] == False)]


    def unapply(self, string, tok):
        t1,t2 = self.DS[tok-2]

        i=0
        string_new = torch.empty(0)
        while i < len(string):
            if string[i] == tok:
                string_new.append(t1)
                string_new.append(t2)
            else:
                string_new.append(string[i])
            i+=1
        self._dset_tok = torch.tensor(string_new)
        return string_new


    def pairwise_count(self, string, t1, t2):
        i=val=0
        while i < len(string)-1:
            if self._dset_tok[i] == t1 and self._dset_tok[i+1] == t2:
                val += 1
                i += 1 if t1 != t2 else 2
            else:
                i += 1
        return val

    def all_pairwise_counts(self, string):
        count_mat = np.zeros([self.dict_size,self.dict_size])
        repeat = False

        i=0
        while i < len(string)-1:
            t1 = string[i]
            t2 = string[i+1]

            if t1==t2 and not repeat:
                count_mat[t1,t2] += 1
                repeat = True
            if t1==t2 and repeat:
                repeat = False
            else:
                repeat = False
                count_mat[t1,t2] += 1
             
            i += 1

        return count_mat

    def add_token(self, dset):
        apc = self.all_pairwise_counts(dset)
        max_bigram = np.max(apc)
        argmax_bigram = list(np.unravel_index(np.argmax(apc, axis=None), apc.shape))
	
        if max_bigram >= self.kappa:
            self.DS.append(argmax_bigram)
            # return True if a token was added
            return True
        return False

    def learn_dict(self, dset=None):
        print("Learning dictionary using BPE...")
        if dset is not None:
            self._dset_tok = np.array(dset.tolist())
        else:
            return print("No dictionary was learned since dataset was None")
        self.DS = []

        while len(self._dset_tok) > 1:
            self.dict_size = len(self.DS) + 2
            self.dictionary = range(self.dict_size)
            if self.dict_size < self.max_dict_size and self.add_token(self._dset_tok):
                new_tok = len(self.DS)+1
                self._dset_tok = self.apply(self._dset_tok, new_tok)
            else:
                break

        print(f"Dictionary learnt. Size of dictionary: {self.dict_size}")


    def encode(self, string):
        assert set(string).issubset(set(range(2))), "String contains elements outside alphabet"
        self._dset_tok = string

        for tok_i in range(len(self.DS)):
            self._dset_tok = self.apply(self._dset_tok, tok_i+2)
        return torch.tensor(self._dset_tok)


    def decode(self, string):
        self._dset_tok = string
        assert set(self._dset_tok).issubset(set(self.dictionary)), "String contains elements outside dictionary"

        for tok_i in reversed(range(len(self.DS))):
            self.unapply(self._dset_tok, tok_i+2)
        return self._dset_tok
	
    def encode_batch(self, batch):
        batch_size = batch.size(dim=0)
        enc = []
        for i in range(batch_size):
            enc.append(torch.tensor(np.array(self.encode(batch[i,:].tolist()))))
        nt = torch.nested.nested_tensor(enc)
        nt_padded = torch.nested.to_padded_tensor(nt, 0)

        return nt_padded