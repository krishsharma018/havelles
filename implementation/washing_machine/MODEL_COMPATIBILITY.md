# MODEL_COMPATIBILITY.md — Existing PLE Implementation & ESP32-S3 Architecture

## 1. Overview of Existing PLE Implementation

The existing working Per-Layer Embedding (PLE) Small Language Model (SLM) implementation is located in `esp32-ai-main/src/model.py` and deployed via `esp32-ai-main/runtime/llm.h`. 

The architecture is a **decoder-only Transformer** with **Gemma-style Per-Layer Embeddings (PLE)** adapted for ultra-resource-constrained microcontrollers (ESP32-S3).

---

## 2. Core Architecture Parameters

| Parameter | PyTorch Config (`Config`) | C Runtime (`Cfg` in `llm.h`) | Description |
|---|---|---|---|
| `d_model` | 128 | `dim` | Hidden dimension of the transformer core |
| `n_layers` | 6 (or 4) | `n_layers` | Number of transformer decoder layers |
| `n_heads` | 4 | `n_heads` | Number of multi-query/multi-head attention heads |
| `head_dim` | `d_model // n_heads` = 32 | `Dh = dim / n_heads` | Dimension per head |
| `ffn_hidden` | 256 | `ffn` | Hidden width of SwiGLU FFN (`gate`, `up`, `down`) |
| `ple_dim` | 64 | `ple_dim` | Dimension of the per-layer embedding lookup table |
| `seq_len` | 512 (default) / 96 / 128 | `seq_len` | Maximum sequence context window |
| `vocab_size` | 4096 (Barista) / 1024 (Washer) | `vocab` | Input embedding vocabulary size |
| `out_vocab_size` | None / 1024 | `out_vocab` | Output head vocabulary size (`out_head`) |
| `rope_theta` | 10000.0 | `rope_theta` | Rotary Position Embedding base frequency |
| `group` | 64 (in quantize) | `group` | Quantization group size for INT4 symmetric weights |

---

## 3. Mathematical Operations & Layer Formulation

### 3.1 Input Embedding & Per-Layer Conditioning
For token ID $t \in [0, V-1]$ at position $p$:
1. **Token Embedding**:
   $$x_0 = \text{Embedding}(t) \in \mathbb{R}^{D}$$
2. **Model Projection of Input Embedding**:
   $$\text{tmpP} = \text{RMSNorm}\left(\frac{\text{Linear}_{\text{ple\_model\_proj}}(x_0)}{\sqrt{D}}\right) \in \mathbb{R}^{L \times P}$$
3. **Per-Layer Table Lookup**:
   $$\text{table} = \text{Embedding}_{\text{ple\_table}}(t) \in \mathbb{R}^{L \times P}$$
4. **Combined Per-Layer Embedding**:
   $$\text{ple}_l = \frac{\text{tmpP}_l + \text{table}_l \cdot \sqrt{P}}{\sqrt{2}} \in \mathbb{R}^P \quad \text{for } l = 0, \dots, L-1$$

### 3.2 Transformer Block ($l = 0 \dots L-1$)
Each block consists of:
1. **Multi-Head Self-Attention with Split-Half RoPE**:
   $$h_1 = \text{RMSNorm}(x)$$
   $$Q, K, V = \text{Linear}_{\text{qkv}}(h_1) \quad (\text{split into } 3D)$$
   $$Q', K' = \text{ApplyRoPE}(Q, K, \text{pos}=p, \theta=10000)$$
   $$\text{Cache } K', V' \text{ in KV-cache}$$
   $$\text{Attn} = \text{Softmax}\left(\frac{Q' {K'}^T}{\sqrt{D_h}}\right) V'$$
   $$x = x + \text{Linear}_{\text{attn\_proj}}(\text{Attn})$$

2. **SwiGLU Feed-Forward Network**:
   $$h_2 = \text{RMSNorm}(x)$$
   $$\text{FFN}(h_2) = \text{Linear}_{\text{down}}\left(\text{SiLU}(\text{Linear}_{\text{gate}}(h_2)) \cdot \text{Linear}_{\text{up}}(h_2)\right)$$
   $$x = x + \text{FFN}(h_2)$$

3. **Per-Layer Gating Sublayer (PLE Block)**:
   $$g = \text{GELU}(\text{Linear}_{\text{ple\_gate}}(x)) \in \mathbb{R}^P$$
   $$h_3 = \text{Linear}_{\text{ple\_proj}}(g \odot \text{ple}_l) \in \mathbb{R}^D$$
   $$x = x + \text{RMSNorm}(h_3)$$

4. **Final Output Head**:
   $$y = \text{RMSNorm}(x)$$
   $$\text{logits} = \text{Linear}_{\text{out\_head}}(y) \in \mathbb{R}^{V_{\text{out}}}$$

---

## 4. Parameter Classification & Parameter Budget Formula

The PLE architecture separates parameters into **3 distinct memory tiers** by access pattern:

$$\text{Total Parameters} = N_{\text{core}} + N_{\text{table}} + N_{\text{stream}}$$

### 4.1 Core Parameters ($N_{\text{core}}$) — Every-Token Dense Computation
Weights accessed on **every single token** during forward pass:
- `tok_emb` (if non-tied) or `emb_down`
- `ple_model_proj`: $L \cdot P \cdot D$
- `ple_proj_norm`: $P$
- Per-layer ($L$ blocks):
  - `attn_norm`: $D$
  - `qkv`: $3D \cdot D = 3 D^2$
  - `attn_proj`: $D^2$
  - `ffn_norm`: $D$
  - `gate`, `up`: $2 \cdot (F \cdot D)$
  - `down`: $D \cdot F$
  - `ple_gate`: $P \cdot D$
  - `ple_proj`: $D \cdot P$
  - `ple_norm`: $D$
- `out_norm`: $D$

For $D=128, L=6, F=256, P=64$:
$$N_{\text{core}} = L \cdot (4 D^2 + 3 F D + 2 P D + 3 D) + L \cdot P \cdot D + \dots \approx 1,132,992 \text{ params}$$

### 4.2 Table Parameters ($N_{\text{table}}$) — Sparse Memory-Mapped Lookup
Weights accessed via **single-row lookup** per token:
- `ple_table`: $V \cdot L \cdot P$
For $V=1024, L=6, P=64$: $1024 \times 384 = 393,216$ params.

### 4.3 Stream Parameters ($N_{\text{stream}}$) — Output Head Scan
Weights read as a **single matrix-vector multiplication** at the output:
- `out_head`: $V_{\text{out}} \cdot D$ (when tied with `tok_emb`, shares storage with input embedding).
For $V=1024, D=128$: $1024 \times 128 = 131,072$ params.

---

## 5. ESP32-S3 Hardware Memory Tier Allocation

The ESP32-S3 has three distinct memory tiers:

```
+-------------------------------------------------------------------------+
|                              FLASH MEMORY                               |
| - Mapped into memory bus via MMU (pages read on demand)                  |
| - Holds raw INT4 Quantized Model Image (weights + fp16 group scales)    |
| - Holds PLE Lookup Table (ple_table) accessed via sparse row lookups    |
+-------------------------------------------------------------------------+
                                    |
                                    v (Staged on boot)
+-------------------------------------------------------------------------+
|                             PSRAM (8 MB)                                |
| - High-capacity external RAM (SPI bus)                                  |
| - Holds Staged Core Dense Weights (int8 unpacked format for fast GEMM)  |
| - Holds Key-Value Cache (kcache, vcache) for context [2 x L x S x D]   |
| - Holds intermediate execution buffers & model image copy if needed     |
+-------------------------------------------------------------------------+
                                    |
                                    v (Per-token execution)
+-------------------------------------------------------------------------+
|                           INTERNAL SRAM (512 KB)                        |
| - Ultra-fast on-chip RAM                                                |
| - Holds active token activation vector x [D] (128 floats = 512 bytes)   |
| - Holds RMSNorm gains, GELU/SiLU intermediate activation vectors        |
| - Holds int8 activation quantization buffer xq [128 bytes]              |
| - Holds RoPE cos/sin values for current position pos                    |
+-------------------------------------------------------------------------+
```

### 5.1 Memory Breakdown for Washer SLM (Candidate C, ctx=128)

| Memory Tier | Content | Size (Candidate C) | ESP32-S3 Budget | Status |
|---|---|---|---|---|
| **FLASH** | Mapped INT4 Quantized Model Image | **~859.8 KB** | 8 MB / 16 MB | ✅ < 11% of Flash |
| **PSRAM** | Staged Core Dense Weights (INT8) | **1,106.4 KB** | 8 MB (8,192 KB) | ✅ < 14% of PSRAM |
| **PSRAM** | KV-Cache ($2 \times 6 \times 128 \times 128 \times 4$ B) | **768.0 KB** | 8 MB | ✅ < 10% of PSRAM |
| **PSRAM** | Total PSRAM Runtime Footprint | **~1.87 MB** | 8 MB | ✅ < 24% of PSRAM |
| **SRAM** | Hot Activation & Scratch Buffers | **~10.0 KB** | 512 KB | ✅ < 2% of SRAM |

---

## 6. Compatibility & Verification Verdict

The washing machine domain task is **100% compatible** with the existing ESP32-S3 PLE architecture and `llm.h` C-runtime. No changes to the underlying C kernel or memory layout are required.
