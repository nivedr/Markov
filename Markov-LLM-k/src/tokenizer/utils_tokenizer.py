import numpy as np
import itertools
import collections
import math

def get_batch(p, q, order, seq_length, batch_size, generator, extra_args, device='cpu'):
    data = torch.zeros(batch_size, seq_length+1, device=device)
    if extra_args.initial == 'steady':
        alpha = q / (p+q)
    elif extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    # Generate first k bits
    for k in range(order):
        data[:,k] = torch.bernoulli(alpha*torch.ones((batch_size,), device=device), generator=generator)
    for i in range(order, seq_length):
        data[:,i] = get_next_symbols(p, q, data[:,i-order])
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    #if "cuda" in torch.device(device).type:
    #    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #    x = x.pin_memory().to(device, non_blocking=True)
    #    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def get_next_symbols(p, q, data):
    P = torch.Tensor([[1-p, p],[q, 1-q]]).to(data.device)
    M = P[data.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s

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