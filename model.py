"""
A modernised "Let's build GPT": a decoder-only Transformer written from scratch,
updated with the components used by current Llama-style models.

    nanoGPT / GPT-2                 ->  this model
    LayerNorm                       ->  RMSNorm            (Zhang & Sennrich, 2019)
    learned positional embeddings   ->  RoPE               (Su et al., 2021)
    MLP with GELU                   ->  SwiGLU             (Shazeer, 2020)
    linear layers with bias         ->  no bias

Attention, the causal mask and the generation loop are implemented by hand
(no torch.nn.Transformer / nn.MultiheadAttention / scaled_dot_product_attention).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256      # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    multiple_of: int = 64      # SwiGLU hidden size is rounded up to a multiple of this


# ---------------------------------------------------------------------------
# RMSNorm  (Zhang & Sennrich 2019, Section 4, Eq. 4)
#   a_bar_i = a_i / RMS(a) * g_i,   RMS(a) = sqrt( 1/n * sum_i a_i^2 )
# Unlike LayerNorm there is no mean subtraction and no bias - only rescaling.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gain g

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# RoPE  (Su et al. 2021, Section 3.2, Eq. 14-16; efficient form: Section 3.4.2, Eq. 34)
#   theta_i = 10000^(-2(i-1)/d),  i = 1..d/2
#   every pair (x_{2i}, x_{2i+1}) of a query/key vector at position m is rotated by m*theta_i,
#   so q_m . k_n depends only on the relative offset (m - n).
# ---------------------------------------------------------------------------
def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    """Returns cos and sin tables of shape (max_seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dimension"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)  # (T, head_dim/2): m * theta_i
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotates pairs of coordinates. x: (B, n_head, T, head_dim)."""
    T = x.size(-2)
    cos, sin = cos[:T], sin[:T]               # (T, head_dim/2), broadcast over B and heads
    x1, x2 = x[..., 0::2], x[..., 1::2]       # (x_1, x_3, ...) and (x_2, x_4, ...)
    rotated = torch.stack(
        (x1 * cos - x2 * sin,                 # 2D rotation of each pair
         x1 * sin + x2 * cos),
        dim=-1,
    )
    return rotated.flatten(-2)                # interleave back: (x_1', x_2', x_3', x_4', ...)


# ---------------------------------------------------------------------------
# Multi-head causal self-attention  (Phuong & Hutter 2022, Section 5, Algorithms 4-5)
#   Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + causal mask ) V
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # lower-triangular mask: token t may attend only to tokens 0..t
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # position information enters only here: rotate queries and keys
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                                   # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)              # concatenate heads
        return self.resid_dropout(self.proj(y))


# ---------------------------------------------------------------------------
# SwiGLU feed-forward  (Shazeer 2020, Section 2, Eq. 5-6)
#   FFN_SwiGLU(x) = (Swish(x W) * (x V)) W2
# Three matrices instead of two, so the hidden size is reduced by 2/3
# (8/3 * d instead of 4 * d) to keep the parameter count comparable.
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(2 * 4 * cfg.n_embd / 3)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)  # W  (passed through Swish)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)    # V  (linear branch)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)  # W2 (projection back)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))  # SiLU = Swish_1


# ---------------------------------------------------------------------------
# Transformer block: pre-norm residual connections
#   x = x + Attention(RMSNorm(x));  x = x + SwiGLU(RMSNorm(x))
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # no positional embedding table - positions are encoded by RoPE inside attention
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        x = self.dropout(self.tok_emb(idx))                  # (B, T, C)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))                  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling (Phuong & Hutter 2022, Algorithm 14).
        Without a KV cache: the whole (cropped) context is recomputed at every step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
