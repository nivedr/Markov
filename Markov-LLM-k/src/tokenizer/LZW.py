import numpy as np
import itertools
import collections
import math
import Tokenizer
import utils_tokenizer


class LZW(Tokenizer.Tokenizer):

	def __init__(self, max_dict_size):
		super().__init__(max_dict_size=max_dict_size)


	def learn_dict(self, dset=None):
		print("Learning dictionary using LZW...")
		if dset is not None:
			self._dset_tok = dset
		else:
			return print("No dictionary was learned since dataset was None")
		self.DS = []

		start = utils_tokenizer.Node(-1)
		substr = start
		self.DS = [start]
		for i in range(len(self._dset_tok)):	
			longer_exists = [C for C in substr.children if self._dset_tok[i]==C.data]
			if longer_exists:
				substr = longer_exists[0]
			else:
				if self.dict_size==self.max_dict_size:
					break
				new_tok = utils_tokenizer.Node(self._dset_tok[i])
				substr.insert(new_tok)
				self.DS.append(new_tok)
				substr = start
		
			self.dict_size=len(self.DS)-1
		self.dictionary=range(self.dict_size)
		print(f"Dictionary learnt. Size of dictionary: {self.dict_size}")


	def encode(self, string):
		enc = []

		assert set(string.tolist()).issubset(set(range(2))), "String contains elements outside alphabet"
		substr = self.DS[0]
		for i in range(len(string)):
			longer_exists = [C for C in substr.children if string[i]==C.data]
			if longer_exists:
				substr = longer_exists[0]
				i+=1
			else:
				enc.append(self.DS.index(substr)-1)
				substr = self.DS[0]
		return enc
	
	def encode_batch(self, batch):
		return [self.encode(string) for string in batch]


	def decode(self, string):
		assert set(string).issubset(set(self.dictionary)), "String contains elements outside dictionary"
		self._dset_tok = []

		for tok in string:
			if type(string[0]) == utils_tokenizer.Node:
				tok_obj = tok
			else:
				tok_obj = self.DS[tok+1]
			
			dec_str=[]
			while tok_obj.data != -1:
				dec_str.append(tok_obj.data)
				tok_obj = tok_obj.parent
			self._dset_tok.extend(reversed(dec_str))
		return self._dset_tok


