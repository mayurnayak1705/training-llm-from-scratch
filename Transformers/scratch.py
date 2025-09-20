import torch
import torch.nn as nn
import math


class Linear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype= None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.device = device
    self.dtype = dtype
    self.W = nn.Parameter(torch.rand(self.out_features, self.in_features, dtype=self.dtype, device=self.device))
    nn.init.trunc_normal_(self.W, std=0.02)

  def forward(self, x):
    output = x @ self.W.T
    return output


class Embedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.device = device
    self.dtype = dtype
    self.W = nn.Parameter(torch.rand(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device))
    nn.init.trunc_normal_(self.W, std=0.02)

  def forward(self, x):
    return self.W[x]



class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.device = device
    self.dtype = dtype
    self.W = nn.Parameter(torch.rand([self.d_model], device=self.device, dtype=self.dtype))

  def forward(self, x):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x / rms) * self.W


class PositionwiseFeedForward(nn.Module):
  def __init__(self,d_model, dff, device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    self.dff = dff
    self.w1_weight = nn.Parameter(torch.rand(self.dff, d_model,device=device, dtype=dtype))
    self.w3_weight = nn.Parameter(torch.rand(self.dff, d_model,device=device, dtype=dtype))
    self.w2_weight = nn.Parameter(torch.rand(self.d_model, self.dff,device=device, dtype=dtype))

  def forward(self, x):
    x1 = x @ self.w1_weight.T
    sigx = torch.sigmoid(x1)
    swish = x1 * sigx
    swish2 = x @ self.w3_weight.T
    swish3 = swish * swish2
    output = swish3 @ self.w2_weight.T
    return output


class Softmax(nn.Module):
  def __init__(self, d_model,device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    
  def forward(self, x):
    x = x - torch.max(x, dim= self.d_model, keepdim=True).values
    num = torch.exp(x)
    den = torch.sum(num,dim= self.d_model,keepdim=True)
    return num/den


# class RotaryPositionalEmbedding(nn.Module):
#   def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
#     super().__init__()
#     self.theta = theta
#     self.d_k = d_k
#     self.max_seq_len = max_seq_len

#     d = d_k //2
#     dens = torch.arange(0, d, dtype=torch.float32)
#     den = ((2 * dens) / d_k)
#     den = 1.0 / ((theta ** den).view(1, d))
#     pos = torch.arange(0, max_seq_len).view(max_seq_len,1)
#     angle = pos * den
#     cos = torch.cos(angle)
#     sin = torch.sin(angle)
#     self.register_buffer('sin', sin, persistent=False)
#     self.register_buffer('cos', cos, persistent=False)

#   def forward(self, x, token_positions):
#     batch, seq_len, _ = x.shape
#     cos = self.cos[:seq_len,:]
#     sin = self.sin[:seq_len,:]
#     cos = cos.unsqueeze(0)
#     sin = sin.unsqueeze(0)
#     x_even = x[..., 0::2] 
#     x_odd  = x[..., 1::2]
#     x_even_1 = x_even * cos - x_odd* sin
#     x_odd_1 = x_even*sin + x_odd * cos
#     x1 = torch.stack([x_even_1,x_odd_1], dim=-1)
#     x_out = x1.flatten(-2) 
#     return x_out

class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
      super().__init__()
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len

      d = d_k // 2
      dens = torch.arange(0, d, dtype=torch.float32, device=device)
      inv_freq = 1.0 / (theta ** (2 * dens / d_k))  
      pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)  
      angles = pos[:, None] * inv_freq[None, :]     
      
      self.register_buffer('cos', torch.cos(angles), persistent=False)  
      self.register_buffer('sin', torch.sin(angles), persistent=False) 

  def forward(self, x, token_positions=None):
      original_shape = x.shape
      *prefix, seq_len, d_k = x.shape
      cos = self.cos[:seq_len, :]  
      sin = self.sin[:seq_len, :]  
      for _ in prefix:
          cos = cos.unsqueeze(0)
          sin = sin.unsqueeze(0)
      cos = cos.expand(*prefix, seq_len, d_k // 2)
      sin = sin.expand(*prefix, seq_len, d_k // 2)
      x_even = x[..., 0::2]
      x_odd = x[..., 1::2]
      x_rotated_even = x_even * cos - x_odd * sin
      x_rotated_odd = x_even * sin + x_odd * cos
      x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

      return x_out


class ScaledDotProduct(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, Q, K, V, mask = None):
    dk = Q.size(-1)
    attention = Q @ K.transpose(-2, -1) / math.sqrt(dk) 
    softmax = Softmax(-1)
    if mask is not None:
      attention = attention.masked_fill(mask == False, float('-inf'))
    softmax_attention = softmax(attention)
    return softmax_attention @ V

class CasualMultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, max_seq_len, theta,token_positions=None):
    super().__init__()
    dk = d_model // num_heads
    self.head_dim = d_model // num_heads
    self.token_positions = token_positions
    self.d_model = d_model
    self.num_heads = num_heads
    self.max_seq_len = max_seq_len
    self.theta = theta
    self.scaleddot = ScaledDotProduct()
    self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len)
    self.q_proj_weight = nn.Parameter(torch.rand(d_model, d_model))
    self.k_proj_weight = nn.Parameter(torch.rand(d_model, d_model))
    self.v_proj_weight = nn.Parameter(torch.rand(d_model, d_model))
    self.o_proj_weight = nn.Parameter(torch.rand(d_model, d_model))

  def forward(self, x):
    batch, seq_len, _ = x.size()
    Q = x @ self.q_proj_weight.T
    K = x @ self.k_proj_weight.T
    V = x @ self.v_proj_weight.T
    Q = Q.reshape(batch, seq_len, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
    K = K.reshape(batch, seq_len, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
    V = V.reshape(batch, seq_len, self.num_heads, self.d_model//self.num_heads).transpose(1,2)
    Q = self.rope(Q, token_positions=self.token_positions)
    K = self.rope(K, token_positions=self.token_positions)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, self.num_heads, -1, -1)
    attention = self.scaleddot(Q,K,V, mask)
    attention = attention.transpose(1, 2).reshape(batch, seq_len, self.d_model)
    output = attention @ self.o_proj_weight.T
    return output


class TransformerBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_ff = d_ff
    self.max_seq_len = max_seq_len
    self.theta =theta

    self.rms_norm_1 = RMSNorm(self.d_model)
    self.rms_norm_2 = RMSNorm(self.d_model)
    self.mha = CasualMultiHeadAttention(d_model, num_heads, max_seq_len, theta)
    self.ffn = PositionwiseFeedForward(d_model, d_ff)

  def forward(self, x):
    x_norm = self.rms_norm_1(x)
    x_attention = self.mha(x_norm)
    x = x + x_attention
    x_norm_2 = self.rms_norm_2(x)
    x_ffn = self.ffn(x_norm_2)
    x = x + x_ffn
    return x

class Transformer_LM(nn.Module):
  def __init__(self,vocab_size, context_length,d_model,num_layers,num_heads, d_ff,rope_theta):
    super().__init__()
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.d_ff = d_ff
    self.rope_theta = rope_theta

    self.embeddings = Embedding(vocab_size, d_model)
    self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
    self.rms_norm_end = RMSNorm(self.d_model)
    self.linear = Linear(d_model, vocab_size)
    self.softmax = Softmax(-1)


  def forward(self, x):
    x = self.embeddings(x)
    for layers in self.transformer_layers:
      x = layers(x)
    x = self.rms_norm_end(x)
    x = self.linear(x)
    return x


