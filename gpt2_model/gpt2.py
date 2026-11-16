#!/home/subinoy/Softwares/anaconda3/envs/tch/bin/python


import torch
import torch.nn as nn
from torch.nn import functional as F

#-----Single Attention Head-----#

class Head(nn.Module):
	
	def __init__(self, n_embd, head_size, block_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
		
	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)
		q = self.query(x)
		v = self.value(x)
		
		scores = q @ k.transpose(-2, -1)/ (k.shape[-1]**0.5)
		scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
		weights = F.softmax(scores, dim=-1)
		
		out = weights @ v
		return out

#-----Multiple heads in parallell and then combined-----#
		
class MultiHeadAttention(nn.Module):

	def __init__(self, n_embd, n_head, block_size, dropout=0.0):
		super().__init__()
		head_size = n_embd // n_head
		self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		out= torch.cat([h(x) for h in self.heads], dim=-1)
		return self.dropout(self.proj(out))
		
#-----FFNN-----#

class FFNN(nn.Module):
	
	def __init__(self, n_embd, dropout=0.0):
		super().__init__()
		self.net = nn.Sequential(
				nn.Linear(n_embd, 4*n_embd),
				nn.ReLU(),
				nn.Linear(4*n_embd, n_embd),
				nn.Dropout(dropout))
	
	def forward(self, x):
		return self.net(x)
	
#-----One transformer block-----#	

class Block(nn.Module):
	
	def __init__(self, n_embd, n_head, block_size):
		super().__init__()
		self.ln1 = nn.LayerNorm(n_embd)
		self.attn = MultiHeadAttention(n_embd, n_head, block_size)
		self.ln2 = nn.LayerNorm(n_embd)
		self.ff = FFNN(n_embd)
		
	def forward(self, x):
		x = x + self.attn(self.ln1(x))
		x = x + self.ff(self.ln2(x))
		return x
		
#-----Graph Pretrained Transformer-----#
		
class GPT(nn.Module):

	def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=32, dropout=0.0, device='cpu'):
		super().__init__()
		self.block_size = block_size
		
		self.token_embedding = nn.Embedding(vocab_size, n_embd)
		self.position_embedding = nn.Embedding(block_size, n_embd)
		
		self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.lm_head = nn.Linear(n_embd, vocab_size)
		
	def forward(self, idx, targets=None):
		B, T = idx.shape
		
		tok_emb = self.token_embedding(idx)
		pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
		x = tok_emb + pos_emb
		
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)
		
		loss = None
		if targets is not None:
			B, T, V = logits.shape
			true_value = logits.view(B*T, V)
			predicted_value = targets.view(B*T)
			loss = F.cross_entropy(true_value, predicted_value)
			
		return logits, loss
		
	@torch.no_grad()
	def generate(self, idx, max_new_tokens):
		"""AUtoregressively generate new tokens one at a time"""
		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.block_size:]
			logits, _ = self(idx_cond)
			logits = logits[:,-1, :]
			probs = F.softmax(logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			idx = torch.cat([idx, next_token], dim=1)
		return idx
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

		

