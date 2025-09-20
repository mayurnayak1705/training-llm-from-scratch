import torch
import torch.nn as nn
import math
import regex as re
import json
from copy import deepcopy

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        batch, vocab = outputs.size()
        max_x, _ = outputs.max(dim=-1, keepdim=True)
        outputs = outputs-max_x
        sum_z = torch.log(torch.exp(outputs).sum(-1, keepdim=True)) 
        sum_z = sum_z.view(batch,1)  
        zi = outputs - sum_z

        loss = 0
        for elements in range(batch):
            loss = loss + (zi[elements][targets[elements].item()])
        loss = -loss/batch
        return loss



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  
                    state["exp_avg_sq"] = torch.zeros_like(p.data) 

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss




def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    if t < Tw: 
        return alpha_max * t / Tw
    
    elif t < Tc:  
        progress = (t - Tw) / (Tc - Tw)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + (alpha_max - alpha_min) * cosine
    
    else:  
        return alpha_min


def gradient_clipping(parameters, M):
    grads = [p.grad for p in parameters if p.grad is not None]
    flat_grads = torch.cat([g.view(-1) for g in grads])
    l2_norm = torch.norm(flat_grads, p=2)
    if l2_norm > M:
        scale = M / (l2_norm + 1e-6)
        for g in grads:
            g.mul_(scale)


import numpy as np


def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(x)
    ix = np.random.randint(0, n - context_length, size=batch_size)
    inputs = []
    labels = []

    for i in ix:
        inputs.append(x[i : i + context_length])
        labels.append(x[i + 1 : i + context_length + 1])
    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    labels = torch.tensor(np.array(labels), dtype=torch.long, device=device)
    return inputs, labels


import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer=None):
    checkpoint = torch.load(src, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration




class BPE_Tokenizer_Regex():
  def __init__(self):
    self.merges = {}
    self.vocab = {}
    self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

  @staticmethod
  def most_freq_bytes_pairs(tokens):
    hash_map = {}
    for i in range(len(tokens)-1):
      pairs = tuple(tokens[i:i+2])
      hash_map[pairs] = hash_map.get(pairs,0) + 1
    return hash_map

  @staticmethod
  def merge(id, pairs, idx):
    new_list = []
    i = 0
    while i < len(id):
      if i < len(id) -1 and pairs[0] == id[i] and pairs[1] == id[i+1]:
        new_list.append(idx)
        i = i + 2
      else:
        new_list.append(id[i])
        i = i +1
    return new_list

  def train(self,corpus, vocab_size, verbose=False):
    self.corpus = corpus
    self.vocab_size = vocab_size

    self.tokens = list(corpus.encode("utf-8"))
    self.tokens = list(map(int, self.tokens))

    target_vocab_size = self.vocab_size
    current_vocab_size = 256
    self.num_merges = target_vocab_size - current_vocab_size
    ids = self.tokens
    self.merges = {}


    for i in range(self.num_merges):
      counts = self.most_freq_bytes_pairs(ids)
      max_pair = max(counts, key=counts.get)
      idx = 256 + i
      ids = self.merge(ids, max_pair, idx)
      if verbose == True:
        print(f"merging {max_pair} to {idx}")
      self.merges[max_pair] = idx

    self.vocab = {idx:bytes([idx]) for idx in range(256)}
    for (p0,p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

  def encode(self, text):
    encodings = []
    regex_split =  re.findall(self.GPT4_SPLIT_PATTERN, text)
    #OPTIMISED CODE
    for item in regex_split:
      tokens = list(item.encode("utf-8"))
      for pair in self.merges:               # apply merges in learned order
          tokens = self.merge(tokens, pair, self.merges[pair])
      encodings.extend(tokens)
    return encodings

  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    string = tokens.decode("utf-8", errors='replace')
    return string

  
  def save(self, file_path):
    data = {
        "merges": {str(k): v for k, v in self.merges.items()},
        "vocab": {str(k): v.decode("latin1") for k, v in self.vocab.items()},
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

  def load(self, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    self.merges = {
        tuple(map(int, k.strip("()").split(","))): v
        for k, v in data["merges"].items()
    }
    self.vocab = {
        int(k): v.encode("latin1") for k, v in data["vocab"].items()
    }