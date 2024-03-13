import numpy as np
import itertools
import collections
import math


def entropy(dist, tol=1e-7):
	return np.sum([-x*math.log(x+tol) for x in dist])

def cross_entropy(empirical_counts, unigram_model, tol=1e-7):
	return np.sum([-empirical_counts[k]*math.log(unigram_model.get(k,0)+tol) for k in empirical_counts.keys()])

def isprefix(prefix, string):
	if len(prefix) > len(string):
		return False
	for i in range(len(prefix)):
		if prefix[i] != string[i]:
			return False
	return True

class Node:

	def __init__(self, data=-1):
		self.data = data
		self.children = []
		self.parent = None

	def insert(self, C):
		if C in self.children:
			C.parent = self
			return False
		else:
			self.children.append(C)
			return True