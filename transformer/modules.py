from former import util
from util import mask_

import torch
from torch import nn
import torch.nn.functional as F

import random, math

class SelfAttentionWide(nn.Module):
	def __init__(self, emb, heads=8, mask=False):
	"""
		:param emb: embedding dimension
		:param heads: number of heads
		:param mask:
		"""
		super().__init()__
		
		self.emb = emb
		self.heads = heads
		self.mask - mask
		self.toqueries = nn.Linear(emb, emb*heads, bias=False)
		self.tokeys = nn.Linear(emb, emb*heads, bias=False)
		self.tovalues = nn.Linear(emb, emb*heads, bias=False)
		
		self.unifyheads = nn.Linear(heads*emb, emb)
	def forward(self, x):
		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})
		keys = self.tokeys(x).view(b, t, h, e)
		queries = self.toqueries(x).view(b, t, h, e)
		values = self.tovalues(x).view(b, t, h, e)
		keys = keys.transpose(1, 2).contiguous().view(b*h, t, e)
		queries = queries.transpose(1, 2).contiguous().view(b*h, t, e)
		values = values.transpose(1, 2).contiguous().view(b*h, t, e)
		queries = queries/(e**(1/4))
		keys = keys/(e**(1/4))
		
		dot = torch.bmm(queries, keys.transpose(1, 2))
		
		assert dot.size() == (b*h, t, t)
		
		if self.mask:
			mask_(dot, maskval = float('-inf'), mask_diagonal=False)
		dot = F.softmax(dot, dim = 2)
		
		out = torch.bmm(dot, values).view(b, h, t, e)
		
		out = out.transpose(1, 2).contiguous().view(b, t, h*e)
		
		return self.unifyheads(out)
	
class SelfAttentionNarrow(nn.Module):
	def __init__(self, emb, heads=8, mask=False):
		"""
		
		:param embedding dimension:
		:param number of heads for selfattention:
		:param mask:
		"""
		super().__init__()
		
		assert emb%heads==0, f'Embedding dimension ({emb}) should be divisible by no of heads ({heads})'
		
		self.emb = emb
		self.heads = heads
		self.mask = mask
		
		s = emb//heads
		
		self.tokeys = nn.Linear(s, s, bias=False)
		self.toqueries = nn.Linear(s, s, bias=False)
		self.tovalues = nn.Linear(s, s, bias = False)
		
		self.unifyheads = nn.Linear(heads*s, emb)
	
	def forward(self, x):
		b, t, e = x.size()
		h = self.heads
		assert e==self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
		
		s = e//h
		x = x.view(b, t, h, s)
		
		keys = self.tokeys(x)
		queries = self.toqueries(x)
		values = self.tovalues(x)
		
		assert keys.size() == (b, t, h, s)
		assert queries.size() == (b, t, h, s)
		assert values.size() == (b, t, h, s)
		
		keys = keys.transpose(1, 2).contiguous().view(b*h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b*h, t, s)
		values = values.transpose(1, 2).contiguous().view(b*h, t, s)
		keys = keys/(s**(1/4))
		queries = queries/(s**(1/4))
		
		dot = torch.bmm(queries, keys.transpose(1, 2))
		
		assert dot.size() == (b*h, t, t)
		
		if self.mask:
			mask_(dot, masval=float('-inf'), mask_diagonal=False)
		dot = F.softmax(dot, dim=2)
		
		out = torch.bmm(dot, values).view(b, h, t, s)
		out = out.transform(1, 2).contiguous().view(b, t, s*h)
		return self.unifyheads(out)
		
class TransformerBlock(nn.Module):
	def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()
		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.mask = mask
		self.norm1 = nn.LayerNorm(emb)
		self.norm2 = nn.LayerNorm(emb)
		
		self.ff = nn.Sequential(
			nn.Linear(emb, ff_hidden_mult*emb),
			nn.ReLU(),
			nn.Linear(ff_hidden_mult*emb, emb)
		)
		self.do = nn.Dropout(droupout)
	def forward(self, x):
		attended = self.attention(x)
		x = self.norm1(attended+x)
		x = self.do(x)
		fedforward = self.ff(x)
		x = self.norm2(fedforward + x)
		x = self.do(x)
		return x
		
		
		
