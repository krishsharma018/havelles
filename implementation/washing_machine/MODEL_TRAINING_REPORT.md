# Phase 4 Report: Washing Machine SLM Model Architecture & Training Preparation

## 1. Executive Summary

Phase 4 completes the **model architecture definition, memory layout profiling, training pipeline implementation, and sanity verification** for the Washing Machine Small Language Model (SLM).

The model strictly adheres to the **Per-Layer Embedding (PLE) architecture** deployed in `esp32-ai-main/src/model.py` and `esp32-ai-main/runtime/llm.h`.

---

## 2. Selected Model Architecture (Candidate C)

We evaluated 3 candidate configurations adhering to the ESP32-S3 memory hierarchy. **Candidate C (Large S3-Safe PLE)** is recommended for the first training experiment.

### Candidate Comparison Table

| Metric | Candidate A (Smallest) | Candidate B (Medium) | **Candidate C (Recommended)** |
|---|---|---|---|
| `d_model` | 64 | 128 | **128** |
| `n_layers` | 4 | 4 | **6** |
| `n_heads` | 2 | 4 | **4** |
| `ffn_hidden` | 128 | 256 | **256** |
| `ple_dim` | 32 | 64 | **64** |
| `vocab_size` | 1024 | 1024 | **1024** |
| `seq_len` | 128 | 128 | **128** |
| **Total Parameters** | 385,888 | 1,148,608 | **1,657,280 (~1.66M)** |
| **Core Params ($N_{\text{core}}$)** | 189,280 | 755,392 | **1,132,992** |
| **Table Params ($N_{\text{table}}$)** | 131,072 | 262,144 | **393,216** |
| **Stream Head Params ($N_{\text{stream}}$)** | 65,536 | 131,072 | **131,072** |
| **INT4 Flash Image Size** | 200.2 KB | 595.9 KB | **859.8 KB** |
| **Staged Core INT8 (PSRAM)** | 184.8 KB | 737.7 KB | **1,106.4 KB** |
| **KV Cache at ctx=128 (PSRAM)** | 256.0 KB | 512.0 KB | **768.0 KB** |
| **SRAM Hot Working Buffer** | ~5.0 KB | ~10.0 KB | **~10.0 KB** |

### Rationale for Recommending Candidate C:
1. **Model Depth**: 6 Transformer layers provide the depth required to accurately capture multi-slot washing machine dependencies, constraint interactions, and field choices without risk of underfitting.
2. **Hardware Feasibility**: Total PSRAM usage (staged int8 core + KV cache) is **1.87 MB**, utilizing less than 24% of the 8 MB PSRAM available on ESP32-S3.
3. **Flash Footprint**: The packed INT4 model image is **859.8 KB**, occupying less than 11% of standard 8 MB SPI flash.

---

## 3. Context Length Decision: `ctx=96` vs `ctx=128` (Task 4)

Phase 3 reported that `ctx=96` covered 98.44% of training examples, leaving 591 examples (1.56%) exceeding the 96-token boundary.

### Overflow Measurements & Impact:
- **Total training examples**: 37,950
- **Examples exceeding `ctx=96`**: 591 (1.56%)
- **Max sequence length in train set**: 134 tokens
- **Overflow cause**: Driven primarily by multi-slot input descriptions (>30 tokens) combined with complex multi-field JSON targets (>60 tokens).
- **Coverage at `ctx=128`**: 37,948 out of 37,950 training examples fit **without truncation** (**99.99% coverage**). Only 2 examples exceed 128 tokens.

### Memory Impact Comparison (Candidate C):

| Metric | Context = 96 | Context = 128 | Delta |
|---|---|---|---|
| Sequence Coverage | 98.44% | **99.99%** | +1.55% |
| KV-Cache Size (PSRAM) | 576.0 KB | **768.0 KB** | +192.0 KB |
| Total PSRAM Usage | 1.68 MB | **1.87 MB** | +192.0 KB |
| PSRAM Capacity Used (8MB) | 21.0% | **23.4%** | +2.4% |

### Recommendation: **Use Context Length = 128**
- **Decision**: Set `seq_len = 128`.
- **Justification**: Moving from 96 to 128 eliminates 99.66% of sequence truncation while requiring only +192 KB of PSRAM. This guarantees near-100% complete sequence preservation and optimal training correctness.

---

## 4. Training Objective & Loss Masking (Task 5)

Each training record is structured as a complete instruction-to-target sequence:

$$\text{Sequence} = [\text{BOS}] + \text{natural\_language\_input} + [\text{SEP}] + \text{minimal\_JSON\_target} + [\text{EOS}]$$

### Target-Only Loss Masking Design
To ensure the model learns `natural language → JSON` generation without being dominated by input syntax:
- **Loss Mask = 0.0**: Applied to `[BOS]`, `natural_language_input` tokens, and `[SEP]`.
- **Loss Mask = 1.0**: Applied to `minimal_JSON_target` tokens and `[EOS]`.
- **Loss Formula**:
  $$\mathcal{L} = \frac{\sum_{i=1}^{T-1} m_i \cdot \text{CE}(z_i, y_i)}{\sum_{i=1}^{T-1} m_i + \epsilon}$$
  where $m_i \in \{0, 1\}$ is the loss mask.

---

## 5. Output Vocabulary Compatibility (Task 6)

The input vocabulary size is `1024`.
- All minimal JSON syntax characters (`{`, `}`, `:`, `,`, `"`, `[`, `]`), schema field keys (`"intent"`, `"program"`, `"temperature"`, etc.), and canonical values (`"START_CYCLE"`, `"COTTON"`, `40`, etc.) are contained in the 1024 ByteLevel-BPE vocabulary.
- **Verdict**: The output head uses tied input/output embeddings (`out_vocab_size = 1024`, `head_is_tied = True`).
- No modifications to the vocabulary or output projection layout are required.

---

## 6. JSON Decoding Plan (Task 7)

Created `implementation/washing_machine/JSON_DECODING_PLAN.md` defining:
1. Canonical field ordering for minimal schema keys.
2. Allowed enum value sets for each field.
3. Finite State Machine (FSM) transition rules for on-device constrained decoding.
4. Bitmasking mechanism for logit masking in the ESP32 C-runtime.

---

## 7. Sanity Run Verification Results (Task 10)

Executed `python -m implementation.washing_machine.train --sanity-check` to verify end-to-end functionality.

### Verification Checklist:

| Check | Result | Detail |
|---|---|---|
| 1. Model Instantiation | ✅ PASSED | 1.66M parameter PLE model initialized |
| 2. Tokenizer & Hash Loading | ✅ PASSED | SHA256 verified (`fb80b3ca...`) |
| 3. Sequence Dataset Loading | ✅ PASSED | 37,950 train / 3,000 val records loaded |
| 4. Target Loss Masking | ✅ PASSED | Mask applied; loss calculated on target tokens only |
| 5. Forward Pass | ✅ PASSED | Logits shape `[32, 127, 1024]` |
| 6. Backward Pass | ✅ PASSED | Gradients calculated without NaN/Inf |
| 7. Optimizer Step | ✅ PASSED | Loss decreased from `6.9484` → `6.4036` |
| 8. Checkpoint Save | ✅ PASSED | Saved to `checkpoints/sanity_checkpoint.pt` |
| 9. Checkpoint Reload | ✅ PASSED | Metadata and weights reloaded and verified |

---

## 8. Summary of Created Files

| File | Description |
|---|---|
| [MODEL_COMPATIBILITY.md](file:///E:/havells/implementation/washing_machine/MODEL_COMPATIBILITY.md) | Task 1: Existing PLE model analysis & ESP32-S3 memory allocation |
| [model_config.py](file:///E:/havells/implementation/washing_machine/model_config.py) | Tasks 2 & 3: Model config, architecture definition, & candidate solver |
| [JSON_DECODING_PLAN.md](file:///E:/havells/implementation/washing_machine/JSON_DECODING_PLAN.md) | Task 7: On-device constrained JSON decoder specification |
| [train.py](file:///E:/havells/implementation/washing_machine/train.py) | Tasks 8, 9, 10: Training script, loss masking, metadata, & sanity runner |
| `checkpoints/sanity_checkpoint.pt` | Task 10: Verified sanity check checkpoint file |
| [MODEL_TRAINING_REPORT.md](file:///E:/havells/implementation/washing_machine/MODEL_TRAINING_REPORT.md) | Task 12: This report |

---

## 9. Command for FULL TRAINING

When approved by the Project Lead, execute the following command from `E:\havells`:

```bash
python -m implementation.washing_machine.train --steps 4000 --batch-size 32 --seq-len 128 --lr 0.001
```

---

**PHASE 4 PREPARATION COMPLETE. STOP.**

Awaiting Project Lead approval before starting the full training run.
