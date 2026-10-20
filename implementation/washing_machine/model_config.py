"""model_config.py — Washing Machine SLM Architecture & Configuration
======================================================================
Defines the exact PLE Transformer architecture, parameter budgets, and
candidate model configurations for ESP32-S3 deployment.

Maintains 100% compatibility with `esp32-ai-main/src/model.py` and
`esp32-ai-main/runtime/llm.h`.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """Model architecture configuration for Washing Machine SLM."""
    arm: str = "ple"
    vocab_size: int = 1024
    out_vocab_size: Optional[int] = None
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 256
    seq_len: int = 128
    ple_dim: int = 64
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def uses_per_layer(self) -> bool:
        return self.arm in ("ple", "ple_notable")

    @property
    def resolved_out_vocab_size(self) -> int:
        """Rows in the output head. Defaults to the read vocabulary."""
        return self.vocab_size if self.out_vocab_size is None else self.out_vocab_size

    @property
    def output_ids_are_input_ids(self) -> bool:
        """Whether a sampled index can be fed straight back in as a token."""
        return self.resolved_out_vocab_size == self.vocab_size

    @property
    def head_is_tied(self) -> bool:
        """Whether the head shares storage with tok_emb."""
        return self.arm != "fatembed" and self.output_ids_are_input_ids

    @property
    def table_width(self) -> int:
        return self.n_layers * self.ple_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(seq_len: int, head_dim: int, theta: float, device: torch.device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh)
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        if cfg.uses_per_layer:
            self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
            self.ple_proj = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
            self.ple_norm = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ple: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        if ple is not None:
            g = F.gelu(self.ple_gate(x))
            x = x + self.ple_norm(self.ple_proj(g * ple))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        out_vocab = cfg.resolved_out_vocab_size
        if cfg.arm == "fatembed":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.table_width)
            self.emb_down = nn.Linear(cfg.table_width, cfg.d_model, bias=False)
            self.head = nn.Linear(cfg.d_model, out_vocab, bias=False)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.head = nn.Linear(cfg.d_model, out_vocab, bias=False)
            if cfg.head_is_tied:
                self.head.weight = self.tok_emb.weight  # tied

        if cfg.uses_per_layer:
            self.ple_model_proj = nn.Linear(cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False)
            self.ple_proj_norm = RMSNorm(cfg.ple_dim)
        if cfg.arm == "ple":
            self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if cfg.arm == "fatembed":
            nn.init.normal_(self.emb_down.weight, std=cfg.table_width**-0.5)
        for block in self.blocks:
            if cfg.uses_per_layer:
                nn.init.zeros_(block.ple_norm.weight)

        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        cfg = self.cfg
        device = idx.device
        if self.cos.device != device:
            cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta, device)
        else:
            cos, sin = self.cos, self.sin

        x = self.tok_emb(idx)
        if cfg.arm == "fatembed":
            x = self.emb_down(x)

        ple = None
        if cfg.uses_per_layer:
            B, T = idx.shape
            ple = self.ple_model_proj(x) * (cfg.d_model**-0.5)
            ple = self.ple_proj_norm(ple.view(B, T, cfg.n_layers, cfg.ple_dim))
            if cfg.arm == "ple":
                table = self.ple_table(idx).view(B, T, cfg.n_layers, cfg.ple_dim)
                ple = (ple + table * (cfg.ple_dim**0.5)) * (2**-0.5)

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if ple is None else ple[:, :, i])

        x = self.out_norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Masked CrossEntropyLoss for Target-only loss computation
            vocab_dim = cfg.resolved_out_vocab_size
            flat_logits = logits.view(-1, vocab_dim)
            flat_targets = targets.view(-1)
            
            if mask is not None:
                flat_mask = mask.view(-1).float()
                raw_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none", ignore_index=-1)
                loss = (raw_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
            else:
                loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
                
        return logits, loss

    def param_budget(self) -> Dict[str, int]:
        table = 0
        if self.cfg.arm == "ple":
            table += self.ple_table.weight.numel()
        if not self.cfg.head_is_tied:
            table += self.tok_emb.weight.numel()
        stream = self.head.weight.numel()
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return {
            "core": total - table - stream,
            "stream": stream,
            "table": table,
            "total": total,
        }


def get_recommended_config() -> Config:
    """Returns the recommended Candidate C configuration for Washing Machine SLM."""
    return Config(
        arm="ple",
        vocab_size=1024,
        out_vocab_size=1024,
        d_model=128,
        n_layers=6,
        n_heads=4,
        ffn_hidden=256,
        seq_len=128,  # Selected ctx=128 for 99.99% coverage
        ple_dim=64,
        rope_theta=10000.0,
    )


if __name__ == "__main__":
    print("=== Model Config Verification & Candidate Breakdown ===")
    cfg_rec = get_recommended_config()
    model = TinyLM(cfg_rec)
    budget = model.param_budget()
    print(f"Recommended Config: {cfg_rec}")
    print(f"Param Budget: {budget}")
