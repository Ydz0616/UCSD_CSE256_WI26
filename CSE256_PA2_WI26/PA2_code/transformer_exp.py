"""
Experimental Transformer Decoder variants for Part 3 architecture ablation.
Implements: RoPE, GQA, and SwiGLU as drop-in replacements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# RoPE: Rotary Positional Encoding
# ============================================================

class RotaryPositionalEncoding(nn.Module):
    """
    Precomputes sin/cos frequencies for rotary position embeddings.
    Applied to Q and K inside attention, NOT added at the embedding level.
    """
    def __init__(self, head_size, block_size, base=10000.0):
        super().__init__()
        # Compute inverse frequencies: θ_i = 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos for all positions up to block_size
        t = torch.arange(block_size).float()
        freqs = torch.outer(t, inv_freq)  # (block_size, head_size/2)
        self.register_buffer("cos_cached", freqs.cos())  # (block_size, head_size/2)
        self.register_buffer("sin_cached", freqs.sin())  # (block_size, head_size/2)

    def forward(self, T):
        """Returns (cos, sin) each of shape (T, head_size/2)"""
        return self.cos_cached[:T], self.sin_cached[:T]


def apply_rope(x, cos, sin):
    """
    Apply rotary embeddings to x of shape (B, n_head, T, head_size).
    Splits head_size into pairs and rotates each pair by the position angle.
    """
    # Split into even and odd indices along last dim
    x_even = x[..., 0::2]  # (B, n_head, T, head_size/2)
    x_odd  = x[..., 1::2]  # (B, n_head, T, head_size/2)

    # cos/sin shape: (T, head_size/2) -> broadcast to (1, 1, T, head_size/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotate: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd  = x_odd  * cos + x_even * sin

    # Interleave back: stack along last dim and flatten
    # Shape: (B, n_head, T, head_size/2, 2) -> (B, n_head, T, head_size)
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.flatten(-2)

    return x_rotated


# ============================================================
# SwiGLU: Swish-Gated Linear Unit FFN
# ============================================================

class SwiGLU(nn.Module):
    """
    SwiGLU FFN: uses a gated activation instead of simple ReLU.
    gate = Swish(W_gate @ x)
    value = W_value @ x
    output = W_out @ (gate * value)
    """
    def __init__(self, n_embd, hidden_dim, dropout=0.1):
        super().__init__()
        self.w_gate  = nn.Linear(n_embd, hidden_dim)
        self.w_value = nn.Linear(n_embd, hidden_dim)
        self.w_out   = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate  = F.silu(self.w_gate(x))   # Swish activation
        value = self.w_value(x)
        x = gate * value                 # Element-wise gating
        x = self.dropout(self.w_out(x))
        return x


# ============================================================
# Experimental Causal Self-Attention (supports RoPE + GQA flags)
# ============================================================

class ExpCausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with optional RoPE and GQA support.

    Args:
        n_embd: embedding dimension
        n_head: number of query heads
        block_size: max sequence length
        dropout: dropout rate
        use_rope: if True, apply rotary positional encoding to Q and K
        n_kv_head: number of KV heads (if None, equals n_head = standard MHA)
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1,
                 use_rope=False, n_kv_head=None):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_size = n_embd // n_head
        self.use_rope = use_rope

        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = n_head // self.n_kv_head  # how many Q heads share each KV head

        # Q projection: full n_head
        self.query = nn.Linear(n_embd, n_head * self.head_size, bias=False)
        # K, V projections: n_kv_head (fewer if GQA)
        self.key   = nn.Linear(n_embd, self.n_kv_head * self.head_size, bias=False)
        self.value = nn.Linear(n_embd, self.n_kv_head * self.head_size, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

        # RoPE (only if enabled)
        if use_rope:
            self.rope = RotaryPositionalEncoding(self.head_size, block_size)

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V
        q = self.query(x)  # (B, T, n_head * head_size)
        k = self.key(x)    # (B, T, n_kv_head * head_size)
        v = self.value(x)  # (B, T, n_kv_head * head_size)

        # Reshape into heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, n_head, T, hs)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)    # (B, n_kv_head, T, hs)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)    # (B, n_kv_head, T, hs)

        # Apply RoPE to Q and K (if enabled)
        if self.use_rope:
            cos, sin = self.rope(T)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # GQA: expand KV heads to match Q heads by repeating
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_head, T, hs)
            v = v.repeat_interleave(self.n_rep, dim=1)  # (B, n_head, T, hs)

        # Attention scores
        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        # Causal mask
        wei = wei.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        att_weights = wei
        wei = self.attn_dropout(wei)

        # Weighted sum
        y = wei @ v  # (B, n_head, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return y, att_weights


# ============================================================
# Experimental Decoder Block (supports SwiGLU flag)
# ============================================================

class ExpDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1,
                 use_rope=False, n_kv_head=None, use_swiglu=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = ExpCausalSelfAttention(
            n_embd, n_head, block_size, dropout,
            use_rope=use_rope, n_kv_head=n_kv_head
        )
        self.ln_2 = nn.LayerNorm(n_embd)

        if use_swiglu:
            self.mlp = SwiGLU(n_embd, 100, dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 100),
                nn.ReLU(),
                nn.Linear(100, n_embd),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        att_out, att_map = self.attn(self.ln_1(x))
        x = x + att_out
        x = x + self.mlp(self.ln_2(x))
        return x, att_map


# ============================================================
# Experimental Transformer Decoder (configurable)
# ============================================================

class ExpTransformerDecoder(nn.Module):
    """
    Configurable Transformer Decoder.

    Args:
        use_rope: Replace learned positional embeddings with RoPE
        n_kv_head: Number of KV heads for GQA (None = standard MHA)
        use_swiglu: Replace ReLU FFN with SwiGLU
    """
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1,
                 use_rope=False, n_kv_head=None, use_swiglu=False):
        super().__init__()
        self.block_size = block_size
        self.use_rope = use_rope

        # Token embeddings (always needed)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embeddings (only when NOT using RoPE)
        if not use_rope:
            self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            ExpDecoderBlock(n_embd, n_head, block_size, dropout,
                            use_rope=use_rope, n_kv_head=n_kv_head, use_swiglu=use_swiglu)
            for _ in range(n_layer)
        ])

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        if self.use_rope:
            x = tok_emb  # RoPE is applied inside attention, no positional embedding here
        else:
            pos_emb = self.position_embedding_table(
                torch.arange(T, dtype=torch.long, device=idx.device)
            )
            x = tok_emb + pos_emb

        attn_maps = []
        for block in self.layers:
            x, att_map = block(x)
            attn_maps.append(att_map)

        x = self.ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
            return loss

        return logits, attn_maps
