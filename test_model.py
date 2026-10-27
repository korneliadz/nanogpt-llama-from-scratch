"""
Correctness checks for the building blocks.   Run:  python -m pytest -q
"""

import torch

from model import ModelConfig, RMSNorm, Transformer, apply_rope, precompute_rope

torch.manual_seed(0)
SMALL = ModelConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


def test_forward_shapes_and_loss():
    model = Transformer(SMALL)
    idx = torch.randint(0, SMALL.vocab_size, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, SMALL.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causal_mask_future_tokens_do_not_leak():
    """Changing token t must not change the outputs at positions < t."""
    model = Transformer(SMALL).eval()
    idx = torch.randint(0, SMALL.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (changed[0, 10] + 1) % SMALL.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_rope_depends_only_on_relative_position():
    """RoPE property (Su et al., Eq. 16): <R_m q, R_n k> depends only on m - n."""
    head_dim, T = 16, 64
    cos, sin = precompute_rope(head_dim, T)
    q, k = torch.randn(head_dim), torch.randn(head_dim)
    q_rot = apply_rope(q.expand(1, 1, T, head_dim), cos, sin)[0, 0]  # q placed at every position
    k_rot = apply_rope(k.expand(1, 1, T, head_dim), cos, sin)[0, 0]
    for m, n, shift in [(5, 2, 10), (20, 30, 7), (3, 3, 40)]:
        a = q_rot[m] @ k_rot[n]
        b = q_rot[m + shift] @ k_rot[n + shift]
        assert torch.allclose(a, b, atol=1e-4)


def test_rope_preserves_norm_and_is_identity_at_position_zero():
    head_dim, T = 16, 8
    cos, sin = precompute_rope(head_dim, T)
    x = torch.randn(1, 1, T, head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)  # rotation keeps length
    assert torch.allclose(x[..., 0, :], y[..., 0, :])                   # angle 0 at position 0


def test_rmsnorm_matches_formula():
    norm = RMSNorm(8, eps=0.0)
    x = torch.randn(4, 8)
    expected = x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_generate_appends_tokens_beyond_block_size():
    model = Transformer(SMALL).eval()
    start = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=SMALL.block_size + 5)   # longer than the context
    assert out.shape == (2, 1 + SMALL.block_size + 5)
