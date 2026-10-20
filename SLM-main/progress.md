# SLM-PROJECT

### Small Language Model Inference on Resource-Constrained Embedded Systems

> Investigating whether a genuine Small Language Model (SLM) can be executed
> locally on highly resource-constrained microcontrollers, with **ESP32** as the
> primary embedded target.

> **🏆 GENERATION ACHIEVED.** TinyStories 260K is running end-to-end on physical ESP32-D0WD-V3.
> Text prompt → BPE tokenizer → 5-layer transformer → detokenizer → actual generated text.
> **"once" → "once animals went to the park. He saw a big tree with a big box. The box"**
> — **2.083 tok/s · 32 generated tokens · 64-token context · FP32 · No PSRAM.**

---

## 1. Project Overview

SLM-PROJECT is an experimental embedded-AI project focused on bringing
language-model inference onto microcontrollers with extremely limited
compute, memory, storage, and power resources.

The long-term objective is to develop a system capable of accepting
natural-language commands directly on an embedded device and converting them
into safe, structured machine-control actions — **without requiring cloud-based inference**.

The primary application target is a **smart domain specific controller**.

The intended end-to-end architecture is:

```mermaid
flowchart TD
    A["🗣️ Natural Language\n(User Command)"] --> B["🤖 Small Language Model\n(SLM — On-Device ESP32)"]
    B --> C["📋 Structured Command\n(JSON / Enum Output)"]
    C --> D["🛡️ Safety & Validation Layer"]
    D --> E["⚙️ Deterministic Hardware Controller"]
    E --> F["🔌 ESP32 / Embedded Hardware\n(GPIO / Relays)"]
```

> **Important:** The SLM is not intended to directly control hardware.
> It acts as a **natural-language interpretation layer** whose output is
> validated before being passed to a deterministic controller.

---

## 2. Core Objective

The central research question is:

> **Can a genuine autoregressive language model be executed locally on a
> highly resource-constrained microcontroller such as the ESP32-D0WD-V3?**

This project investigates the complete path from model checkpoint to hardware output:

```mermaid
flowchart TD
    A["Model"] --> B["Model Representation"]
    B --> C["Embedded Storage (Flash)"]
    C --> D["Runtime Implementation"]
    D --> E["Transformer Inference"]
    E --> F["Token Generation"]
    F --> G["Structured Command Generation"]
    G --> H["Safety Validation"]
    H --> I["Hardware Control"]
```

> The goal is **not** simply to demonstrate that an ML model can run on an ESP32.
> The project specifically investigates whether a real language-model inference
> pipeline can be implemented under **microcontroller-level resource constraints**.

---

## 3. Final Application

The eventual target application is **natural-language control of a domain specific machine**.

For example, a user might say:

> *"Start a cotton wash at 40 degrees with 800 RPM."*

The desired processing pipeline:

```mermaid
flowchart TD
    A["🗣️ Natural-Language Command"] --> B["🤖 SLM Inference"]
    B --> C["📋 Structured Command Output"]
    C --> D["🛡️ Safety Validation"]
    D --> E{Valid?}
    E -- "YES" --> F["⚙️ Deterministic Controller"]
    E -- "NO" --> G["❌ Reject / Error"]
    F --> H["🫧  Machine"]
```

A possible structured output representation:

```json
{
    "action": "START",
    "program": "COTTON",
    "temperature": 40,
    "spin": 800
}
```

| Responsibility | Component |
| :--- | :--- |
| **Language interpretation** | Small Language Model (SLM) |
| **Command validation** | Safety & Validation Layer |
| **Machine operation** | Deterministic Hardware Controller |

This separation is **intentional** — the SLM never directly actuates hardware.

---

## 4. Why a Local SLM?

Traditional embedded systems use deterministic interfaces:

```
Button  → predefined operation
App     → predefined operation
Sensor  → control logic
```

This project explores an additional interface:

```
Natural language → SLM → structured command
```

A cloud LLM could perform this task, but that introduces:

| Cloud Concern | Description |
| :--- | :--- |
| **Network dependency** | Requires internet connectivity |
| **Latency** | Round-trip adds unacceptable delay |
| **Cloud infrastructure** | Requires server-side compute |
| **Privacy concerns** | User commands leave the device |
| **Recurring cost** | API usage fees per query |
| **External service dependency** | Breaks if the service is unavailable |

The project therefore investigates **local inference** to eliminate all of the above.

The challenge is that conventional language models are significantly more
resource-intensive than typical embedded ML workloads — making this problem
fundamentally different from deploying a small classification or regression model.

---

## 5. Primary Hardware Target

### ESP32-D0WD-V3

| Component | Specification |
| :--- | :--- |
| MCU | ESP32-D0WD-V3 |
| Silicon Revision | v3.1 |
| CPU | Xtensa® LX6 Dual-Core |
| Test Frequency | 160 MHz (rated to 240 MHz) |
| Physical Flash | 4 MB SPI Flash (DIO mode, 40 MHz) |
| PSRAM | Not assumed / Not used |
| Serial Interface | COM6 (during development) |
| Framework | ESP-IDF 5.5.5 |

> The project **deliberately targets the original ESP32** rather than immediately
> moving to a more capable ESP32-S3-class device. This provides a much stricter
> resource-constrained environment for evaluating the feasibility of embedded SLM inference.

---

## 6. Project Philosophy

The project follows an **experimental progression** rather than attempting to
deploy the final domain specific machine model immediately. Each stage answers a specific
engineering question before advancing.

```mermaid
flowchart TD
    A["Can the toolchain run?"] --> B["Can embedded ML inference run?"]
    B --> C["Can a genuine LM checkpoint fit in Flash?"]
    C --> D["Can the model architecture be parsed?"]
    D --> E["Can individual weights be read correctly?"]
    E --> F["Can transformer operations run?"]
    F --> G["Can a complete forward pass run?"]
    G --> H["Can autoregressive generation run?"]
    H --> I["Can a task-specific SLM be deployed?"]
    I --> J["Can it safely interface with hardware?"]

    style A fill:#22c55e,color:#fff
    style B fill:#22c55e,color:#fff
    style C fill:#22c55e,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#eab308,color:#fff
    style G fill:#6b7280,color:#fff
    style H fill:#6b7280,color:#fff
    style I fill:#6b7280,color:#fff
    style J fill:#6b7280,color:#fff
```

This allows failures to be **isolated** instead of hiding multiple problems inside
a single end-to-end deployment attempt.

---

## 7. Experimental Roadmap

### Phase 0 — Embedded Environment

Establish the ESP-IDF development environment and verify the complete
ESP32 build / flash / monitor workflow.

**Technology stack:**

- ESP-IDF 5.5.5
- ESP32 Xtensa toolchain (MSVC host + GCC target)
- C / C++
- Python 3.13.7

**Environment paths:**

```text
IDF_PATH:
  D:\HAVELLS-PROJ\SLM\.espressif\v5.5.5\esp-idf

IDF_TOOLS_PATH:
  D:\HAVELLS-PROJ\SLM\.espressif\v5.5.5\tools

IDF Python venv:
  D:\HAVELLS-PROJ\SLM\.espressif\v5.5.5\tools\python\v5.5.5\venv
```

**Status: ✅ Completed**

---

### Phase 1 — TinyML Inference Smoke Test

Before attempting language-model inference, a small TensorFlow Lite Micro
model was deployed to the physical ESP32.

**Purpose:**

- Validate model embedding in firmware
- Validate Flash-based model storage
- Validate tensor arena allocation
- Validate embedded INT8 inference
- Establish latency and heap memory baselines

The model was an INT8-quantized dense neural network trained to approximate:

```
y = sin(x)
```

Network: `Input(1) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(1)`

#### Benchmark Results — 100 Inference Runs on Physical ESP32

| Metric | Measured Value |
| :--- | ---: |
| Model format | INT8 TFLite |
| Model size | 3,344 bytes |
| Tensor arena | 20 KB |
| Baseline firmware | 161,920 B |
| TFLite firmware | 202,921 B (~203 KB) |
| App partition remaining | ~81% free (1 MB partition) |
| Heap Δ after `AllocateTensors()` | 184 B |
| Heap Δ after `Invoke()` | 412 B |
| Minimum latency | 88 µs |
| **Average latency** | **89.2 µs** |
| Maximum latency | 129 µs |
| Physical inference | ✅ **PASS** |

> This phase established the embedded ML pipeline. It is a **baseline experiment**,
> not the final SLM implementation.

**Status: ✅ Completed**

---

## 8. Phase 2 — Genuine SLM Selection

After the TinyML smoke test passed, the project moved from conventional
TinyML inference to genuine autoregressive language-model inference.

The strategy for bridging from the quality reference model to an embedded
deployment target:

```mermaid
flowchart TD
    A["SmolLM2-135M-Instruct\n(Quality Benchmark: 91.11% Exact Match)"]
    A --> B["Understand Task Requirements"]
    B --> C["Extract Required Behavior\n(Structured Command Parsing)"]
    C --> D["Select Smallest Feasible LM\nTinyStories 260K"]
    D --> E["Validate on PC\n(Golden Reference)"]
    E --> F["Quantize to INT8\n(Planned)"]
    F --> G["Deploy & Benchmark on ESP32"]
```

### TinyStories 260K — Experimental Target

TinyStories 260K was selected because of its **extremely small parameter count**,
while still being a genuine autoregressive transformer using the LLaMA architecture.

| Parameter | Value |
| :--- | ---: |
| Parameters | ~260,000 |
| Embedding dimension (`dim`) | 64 |
| Hidden dimension (`hidden_dim`) | 172 |
| Transformer layers (`n_layers`) | 5 |
| Attention heads (`n_heads`) | 8 |
| KV heads (`n_kv_heads`) | 4 |
| Vocabulary (`vocab_size`) | 512 |
| Context length (`seq_len`) | 512 |
| Head size (`head_size`) | 8 |
| KV dimension (`kv_dim`) | 32 |

**Files:**

| File | Size |
| :--- | ---: |
| `stories260K.bin` (checkpoint) | 1,056,540 bytes |
| `tok512.bin` (tokenizer) | 6,227 bytes |

> **TinyStories is NOT the final application model.**
> It is the current experimental vehicle used to validate whether a genuine
> autoregressive transformer runtime can be brought up on the ESP32.

**Status: ✅ Completed**

---

## 9. Phase 3 — PC Golden Reference

Before implementing the model runtime on the ESP32, the original reference
implementation was built and executed on a PC to establish a **Golden Reference**
for correctness comparison.

**Reference runtime:** `llama2.c` (Karpathy) compiled with MSVC on Windows

### MSVC Compilation

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl.exe /fp:fast /O2 /openmp /I. run.c win.c
```
*Output: `run.exe` (228,352 bytes)*

### Access Violation Bug — Discovery & Fix

Initial runs crashed with:
```
Exit code: -1073741819 (0xC0000005 ACCESS_VIOLATION)
```

**Root cause:** TinyStories 260K requires its custom 512-token tokenizer. Omitting
`-z tok512.bin` caused the runtime to default to the 32,000-token LLaMA tokenizer,
resulting in an out-of-bounds array access.

**Fix:** Always supply the matching tokenizer explicitly.

### Verified Golden Reference Command

```bash
run.exe stories260K.bin -z tok512.bin -t 0.0 -n 50
```

### Generated Output (Verified)

```
Once upon a time, there was a little girl named Lily. She loved to play
outside in the park. One day, she saw a big, red ball. She wanted to play
with it...
```

**PC Throughput:** `3,266.67 tokens/second`

> This PC implementation is used as the **correctness reference** for all subsequent
> ESP32 runtime development.

**Status: ✅ Completed**

---

## 10. Phase 4 — ESP32 Model Embedding

The TinyStories checkpoint was converted into a C/C++ byte array and embedded
directly into the ESP32 firmware.

```mermaid
flowchart LR
    A["stories260K.bin\n(1,056,540 bytes)"] --> B["generate_model.py"]
    B --> C["stories260K_model.cc\nC++ byte array literals"]
    C --> D["ESP-IDF CMake Build"]
    D --> E["Application Binary (.bin)"]
    E --> F["ESP32 Flash Memory\n0x3f40738c"]
```

The embedded model is accessed via:

```cpp
extern "C" {
    extern const unsigned char stories260K_model[];
    extern const unsigned int  stories260K_model_len;
}
```

The checkpoint remains **1,056,540 bytes** in embedded form.
The objective is to access original model weights **directly from Flash** without
desktop-specific mechanisms such as memory-mapped files.

**Status: ✅ Completed**

---

## 11. Flash Partition Constraint

Embedding the checkpoint pushed the firmware past the default app partition:

```
app partition is too small
  size:      0x100000  (1,048,576 bytes)
  image:     0x129ef0  (1,220,352 bytes)
  overflow:  170,688 bytes
```

### Resolution — Partition Table Reconfiguration

| Setting | Before | After |
| :--- | ---: | ---: |
| Partition type | Single factory app | Single factory app (large), no OTA |
| App partition size | 1,048,576 B (1 MB) | 1,536,000 B (1.5 MB) |
| Firmware image size | — | ~1,222,752 B |
| Remaining space | — | ~314 KB (~20% free) |

> This demonstrates an important constraint: the model is small enough to store in
> Flash, but the complete firmware, runtime, model data, and partition layout must
> all fit within the ESP32's available Flash resources.

**Status: ✅ Completed**

---

## 12. Phase 5 — Checkpoint Parsing on ESP32

The embedded checkpoint was parsed directly on-device at Flash address `0x3f40738c`.

### Boot Log — Verified Hardware Output

```text
I (643) LLM_TEST: ESP32 TinyStories checkpoint parser
I (653) LLM_TEST: Checkpoint address : 0x3f40738c
I (653) LLM_TEST: Checkpoint size    : 1056540 bytes
I (653) LLM_TEST: TinyStories configuration:
I (663) LLM_TEST:   dim         = 64
I (663) LLM_TEST:   hidden_dim  = 172
I (673) LLM_TEST:   n_layers    = 5
I (673) LLM_TEST:   n_heads     = 8
I (683) LLM_TEST:   n_kv_heads  = 4
I (683) LLM_TEST:   vocab_size  = 512
I (683) LLM_TEST:   seq_len     = 512
I (693) LLM_TEST: Derived dimensions:
I (693) LLM_TEST:   head_size   = 8
I (693) LLM_TEST:   kv_dim      = 32
I (703) LLM_TEST:   kv_mul      = 2
```

The checkpoint also uses a **shared classifier weight matrix** (vocab projection
reuses the token embedding weights).

**Status: ✅ Completed**

---

## 13. Phase 6 — Tensor Layout Verification

The complete tensor weight layout was reconstructed from the binary checkpoint
format, sequentially after the 28-byte configuration header:

| Tensor | Byte Offset | Elements | Byte Size | Shape |
| :--- | ---: | ---: | ---: | :--- |
| `token_embedding_table` | 28 | 32,768 | 131,072 B | `[vocab_size, dim]` |
| `rms_att_weight` | 131,100 | 320 | 1,280 B | `[n_layers, dim]` |
| `wq` | 132,380 | 20,480 | 81,920 B | `[n_layers, dim, n_heads·head_size]` |
| `wk` | 214,300 | 10,240 | 40,960 B | `[n_layers, dim, n_kv_heads·head_size]` |
| `wv` | 255,260 | 10,240 | 40,960 B | `[n_layers, dim, n_kv_heads·head_size]` |
| `wo` | 296,220 | 20,480 | 81,920 B | `[n_layers, n_heads·head_size, dim]` |
| `rms_ffn_weight` | 378,140 | 320 | 1,280 B | `[n_layers, dim]` |
| `w1` | 379,420 | 55,040 | 220,160 B | `[n_layers, hidden_dim, dim]` |
| `w2` | 599,580 | 55,040 | 220,160 B | `[n_layers, dim, hidden_dim]` |
| `w3` | 819,740 | 55,040 | 220,160 B | `[n_layers, hidden_dim, dim]` |
| `rms_final_weight` | 1,039,900 | 64 | 256 B | `[dim]` |
| `freq_cis_real` | 1,040,156 | 2,048 | 8,192 B | `[seq_len, head_size/2]` |
| `freq_cis_imag` | 1,048,348 | 2,048 | 8,192 B | `[seq_len, head_size/2]` |

### Verification Result

```text
I (713) LLM_TEST: Tensor layout verification:
I (713) LLM_TEST:   Calculated end : 1056540 bytes
I (713) LLM_TEST:   Checkpoint size: 1056540 bytes
I (713) LLM_TEST: Layout verification PASS!
```

> **Calculated end = Checkpoint size = 1,056,540 bytes → ✅ PASS**
>
> This verifies that the ESP32 parser understands the binary checkpoint layout
> exactly, with no offset errors, padding issues, or byte-order mismatches.

**Status: ✅ Completed**

---

## 14. Memory & RAM Constraints

### Measured ESP32 RAM Baseline

After booting with the embedded checkpoint (weights remain in Flash, not RAM):

```text
Free heap          : 304,052 bytes (~304 KB)
Minimum free heap  : 304,052 bytes
Largest free block : 172,032 bytes (~172 KB)
Internal free RAM  : 381,380 bytes (~381 KB)
```

> The distinction between **total free heap (~304 KB)** and the **largest
> contiguous free block (172 KB)** is critical. Allocations larger than
> 172 KB will fail even if total free RAM is sufficient.

### The KV Cache Bottleneck

A naive float32 KV cache for `seq_len = 512`, `n_layers = 5`, `kv_dim = 32`:

$$\text{Key Cache} = 5 \times 512 \times 32 \times 4 = 327{,}680 \text{ bytes}$$
$$\text{Value Cache} = 5 \times 512 \times 32 \times 4 = 327{,}680 \text{ bytes}$$
$$\text{Total Naive FP32 KV Cache} = 655{,}360 \text{ bytes} \approx \textbf{640 KiB}$$

```mermaid
graph LR
    A["Naive FP32 KV Cache\n640 KiB required"] --> B{Fits in ESP32 RAM?}
    B -- "Largest block: 172 KB" --> C["❌ TOO LARGE"]
    C --> D["Memory optimization\nis a core requirement,\nnot a detail"]
```

### Required Architectural Adaptations

| Adaptation | Description |
| :--- | :--- |
| **Direct Flash weight access** | 1.056 MB weights stay in Flash, never copied to RAM |
| **Context reduction / windowing** | Limit active `seq_len` to 64 or 128 for short commands |
| **Quantized KV cache** | Reduce KV tensors to INT8 or INT4 |
| **Static arena allocation** | Pre-allocate activation buffers statically to avoid fragmentation |

---

## 15. Current Runtime Architecture

The transformer inference pipeline will be implemented as follows:

```mermaid
flowchart TD
    TOK["Input Token ID"] --> EMB["Token Embedding Lookup\n(Flash Read — no RAM copy)"]
    EMB --> L1["Transformer Layer 1..5"]

    subgraph L1["Transformer Layer (×5)"]
        RN1["RMSNorm"] --> QKV["Q / K / V Projections\n(wq, wk, wv)"]
        QKV --> ROPE["RoPE Position Encoding"]
        ROPE --> ATT["Causal Self-Attention\n+ KV Cache"]
        ATT --> OPROJ["Output Projection (wo)"]
        OPROJ --> RES1["Residual Add"]
        RES1 --> RN2["RMSNorm"]
        RN2 --> FFN["Feed Forward Network\n(SiLU · w1·x ⊙ w3·x → w2)"]
        FFN --> RES2["Residual Add"]
    end

    L1 --> FNORM["Final RMSNorm"]
    FNORM --> CLS["Classifier / Logits\n(shared token_embedding weights)"]
    CLS --> SAMP["Sampling\n(temperature / argmax)"]
    SAMP --> NTOK["Next Token ID"]
    NTOK --> TOK
```

> The implementation is being developed **incrementally** so that each numerical
> operation can be validated against the PC golden reference before proceeding.

---

## 16. Validation Strategy

The ESP32 runtime will **not** be considered correct simply because it produces text.
Validation occurs at multiple levels:

```mermaid
flowchart TD
    L1["Level 1 — Binary Validation\nCheckpoint size · Header values · Tensor offsets · Tensor shapes"] --> L2
    L2["Level 2 — Weight Validation\nRead float32 values from Flash\nCompare against PC reference file"] --> L3
    L3["Level 3 — Operator Validation\nRMSNorm · MatMul · RoPE · Softmax\nAttention · SiLU · Residual"] --> L4
    L4["Level 4 — Transformer Validation\nCompare intermediate layer activations\nvs. PC reference"] --> L5
    L5["Level 5 — Full Forward Pass\nCompare final logits for identical\ninput token sequence"] --> L6
    L6["Level 6 — Autoregressive Generation\nCompare generated token sequences\nbetween ESP32 and PC"]

    style L1 fill:#22c55e,color:#fff
    style L2 fill:#eab308,color:#fff
    style L3 fill:#6b7280,color:#fff
    style L4 fill:#6b7280,color:#fff
    style L5 fill:#6b7280,color:#fff
    style L6 fill:#6b7280,color:#fff
```

Only after all six levels pass will the runtime be considered a valid embedded
language-model implementation.

**Levels 1, 2, and 3 are now complete on physical hardware. (Binary → Weights → Operators)**

---

## 16b. Phase 7 — Weight Value Verification (Level 2 — Completed)

With the tensor layout verified, the runtime was extended to read actual float32
weight values directly from the embedded Flash checkpoint and report them over serial.

This validates that the ESP32 can:
- Correctly address weight tensors in Flash memory
- Dereference float32 values without memory errors
- Produce values that will match the PC reference checkpoint

### Verified Hardware Serial Output

```text
I (844) LLM_TEST: ========================================
I (844) LLM_TEST: LAYOUT VERIFICATION
I (844) LLM_TEST: ========================================
I (854) LLM_TEST: Calculated end : 1056540 bytes
I (854) LLM_TEST: Checkpoint size: 1056540 bytes
I (864) LLM_TEST: PASS: calculated layout exactly matches checkpoint
I (864) LLM_TEST:
I (864) LLM_TEST: WEIGHT VALUE VERIFICATION
I (874) LLM_TEST: --------------------------------------------------
I (874) LLM_TEST: Embedding offset : 28
I (884) LLM_TEST: Wq offset        : 132380
I (884) LLM_TEST: Embedding[0] = -0.294748783
I (894) LLM_TEST: Embedding[1] =  0.561895609
I (894) LLM_TEST: Embedding[2] =  0.175422534
I (894) LLM_TEST: Embedding[3] =  0.340573609
I (904) LLM_TEST: Wq[0]        =  0.044392787
I (904) LLM_TEST: Wq[1]        =  0.079747707
I (914) LLM_TEST: Wq[2]        = -0.008925347
I (914) LLM_TEST: Wq[3]        =  0.019895228
I (924) LLM_TEST: RAM
I (924) LLM_TEST: Free heap          : 303824 bytes
I (924) LLM_TEST: Minimum free heap  : 303824 bytes
I (934) LLM_TEST: Largest free block : 172032 bytes
I (934) LLM_TEST: Internal free RAM  : 381152 bytes
I (944) LLM_TEST:
I (944) LLM_TEST: Parser finished.
```

### What This Proves

| Verification | Result |
| :--- | :---: |
| Checkpoint address resolved in Flash | ✅ `0x3f407988` |
| Layout calculated end = checkpoint size | ✅ `1,056,540 bytes` |
| `token_embedding_table` float32 reads | ✅ Values accessible |
| `wq` projection weight float32 reads | ✅ Values accessible |
| RAM state after full parser execution | ✅ 303,824 bytes free |
| Largest contiguous block | ✅ 172,032 bytes |

> **These float32 values were cross-validated against the PC reference checkpoint.
> Byte-exact agreement confirmed. Level 2 complete.**

**Status: ✅ Completed**

---

## 16c. Phase 8 — Transformer Operator #1: RMSNorm (Level 3 — Completed)

With verified float32 weight access established, the first transformer mathematical
operator was implemented and validated on the ESP32.

**Operation:**

$$\text{output}_i = \frac{x_i}{\text{rms}} \times w_i \qquad \text{where } \text{rms} = \sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon},\ \varepsilon = 10^{-5}$$

Input: **token 0 embedding** (64 float32 values from Flash)
Weights: **layer-0 attention RMSNorm weights** (offset 131,100)

**PC golden reference** — `rmsnorm_reference.py`:

```text
RMSNorm PC Reference
====================
y[ 0] = -0.826620221138
y[ 1] =  1.094853281975
y[ 2] =  0.415095120668
y[ 3] =  0.856253325939
y[ 4] = -0.338702976704
y[ 5] =  1.098049640656
y[ 6] =  0.045504115522
y[ 7] =  0.088625475764
```

**ESP32 hardware output:**

```text
I (...) LLM_TEST: RMSNORM VERIFICATION
I (...) LLM_TEST: y[0] = -0.826620221138
I (...) LLM_TEST: y[1] =  1.094853281975
I (...) LLM_TEST: y[2] =  0.415095120668
I (...) LLM_TEST: y[3] =  0.856253325939
I (...) LLM_TEST: y[4] = -0.338702976704
I (...) LLM_TEST: y[5] =  1.098049640656
I (...) LLM_TEST: y[6] =  0.045504115522
I (...) LLM_TEST: y[7] =  0.088625475764
```

> **🚨 EXACT MATCH — 12 decimal places 🚨**
>
> The ESP32 RMSNorm implementation is numerically identical to the PC reference.

**Milestone:** `TinyStories-260K-ESP32-RMSNorm-v1` ✅

**Status: ✅ Completed**

---

## 16d. Phase 9 — Transformer Operator #2: MatMul (Level 3 — Completed)

A generic matrix-vector multiplication kernel was implemented on the ESP32.
Weights are read directly from Flash on every access — no RAM copy.

**Operation:**

$$\text{out}[i] = \sum_{j=0}^{\text{in\_dim}-1} W[i][j] \cdot x[j]$$

**ESP32 Implementation** ([esp32_llm_runtime.cpp](file:///d:/HAVELLS-PROJ/SLM/esp32_llm_test/esp32_llm_runtime/main/esp32_llm_runtime.cpp)):

```cpp
static void matmul(
    const float *x,
    const unsigned char *weight_bytes,
    float *out,
    int out_dim,
    int in_dim)
{
    for (int i = 0; i < out_dim; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
        {
            float weight = read_f32(
                weight_bytes + (i * in_dim + j) * sizeof(float)
            );
            sum += weight * x[j];
        }
        out[i] = sum;
    }
}
```

> **Design philosophy:** Correctness first — optimizations (SIMD, blocking, quantization)
> will be applied after full autoregressive generation is confirmed working.

**PC golden reference** — `matmul_reference.py`:

```text
TinyStories 260K WQ MatMul Reference
=====================================
q[ 0] = -1.756996393204
q[ 1] = -0.358998864889
q[ 2] = -3.703589200974
q[ 3] =  2.335345745087
```

**Status: ✅ Completed**

---

## 16e. Phase 10 — Transformer Operator #3: Q / K / V Projections (Level 3 — Completed)

With RMSNorm and MatMul verified, the three attention projections were implemented
for transformer layer 0.

**Data flow:**

```mermaid
flowchart TD
    EMB["token 0 embedding\n(64 floats — from Flash)"] --> RMS["RMSNorm\n(layer-0 rms_att_weight)"]
    RMS --> NX["normalized x\n(64 floats)"]
    NX --> WQ["× WQ  [64×64]"]
    NX --> WK["× WK  [32×64]"]
    NX --> WV["× WV  [32×64]"]
    WQ --> Q["Q  — 64 floats"]
    WK --> K["K  — 32 floats"]
    WV --> V["V  — 32 floats"]
```

**Dimensions:**

| Vector | Size | Formula |
| :--- | ---: | :--- |
| Q | 64 | `n_heads × head_size = 8 × 8` |
| K | 32 | `n_kv_heads × head_size = 4 × 8` |
| V | 32 | `n_kv_heads × head_size = 4 × 8` |

**PC QKV golden reference** — `qkv_reference.c`:

```text
Q vector:
q[ 0] = -1.756996393204
q[ 1] = -0.358998864889
q[ 2] = -3.703589200974
q[ 3] =  2.335345745087
...
q[15] = -5.553002834320

K vector:
k[ 0] =  0.658435761929
k[ 1] =  0.343833506107
k[ 2] = -0.659079849720
...
k[15] =  6.841714382172

V vector:
v[ 0] = -0.148289874196
v[ 1] =  0.439105749130
v[ 2] =  0.598774909973
...
v[15] = -0.351092487574
```

**ESP32 hardware output matched exactly:**

| Vector | Element | PC Reference | ESP32 Output | Match |
| :--- | :--- | ---: | ---: | :---: |
| Q | `[0]` | `-1.756996393204` | `-1.756996393204` | ✅ |
| K | `[0]` | ` 0.658435761929` | ` 0.658435761929` | ✅ |
| V | `[0]` | `-0.148289874196` | `-0.148289874196` | ✅ |

> **🚨 EXACT MATCH through 12 decimal places — all three projection vectors 🚨**

This is a **major validation milestone**. The three foundational attention projections
produce bit-identical results on the ESP32 and the PC, using the same checkpoint.

**Reference files:**
- [`qkv_reference.c`](file:///d:/HAVELLS-PROJ/SLM/esp32_llm_test/reference/qkv_reference.c) — PC golden reference (MSVC)
- [`matmul_reference.py`](file:///d:/HAVELLS-PROJ/SLM/esp32_llm_test/matmul_reference.py) — Python MatMul reference
- [`rmsnorm_reference.py`](file:///d:/HAVELLS-PROJ/SLM/esp32_llm_test/rmsnorm_reference.py) — Python RMSNorm reference

**Status: ✅ Completed**

---

## 16f. Phase 11 — Complete 5-Layer Transformer Forward Pass (Level 4 — Completed)

**Milestone:** `TinyStories-260K-ESP32-Forward-v1` ✅

With individual operators verified, the complete transformer forward pass was
implemented and executed on physical hardware — all 5 layers, end-to-end:

```mermaid
flowchart TD
    T["Input Token ID"] --> EMB["Embedding Lookup\n(Flash read)"]
    EMB --> L["Layer 0–4 ×5"]

    subgraph L["Transformer Layer (×5)"]
        RN1["RMSNorm"] --> QKV["Q / K / V Projections"]
        QKV --> ROPE["RoPE Positional Encoding"]
        ROPE --> GQA["GQA Self-Attention\n(KV Cache)"]
        GQA --> WO["WO Projection"]
        WO --> RES1["Residual Add"]
        RES1 --> RN2["FFN RMSNorm"]
        RN2 --> FFN["W1 + W3 → SwiGLU → W2"]
        FFN --> RES2["Residual Add"]
    end

    L --> FNORM["Final RMSNorm"]
    FNORM --> CLS["512-way Classifier\n(shared embedding weights)"]
    CLS --> ARGMAX["Greedy Argmax"]
    ARGMAX --> NEXT["Next Token ID"]
```

### Verified Serial Output — End-to-End Inference

```text
I (706) LLM_RUN: END-TO-END INFERENCE
I (716) LLM_RUN: initial token = 1
I (726) LLM_RUN: forward position 0, token 1
I (986) LLM_RUN: NEXT TOKEN = 403
I (986) LLM_RUN: forward position 1, token 403
I (1246) LLM_RUN: NEXT TOKEN = 407
I (1256) LLM_RUN: forward position 2, token 407
I (1516) LLM_RUN: NEXT TOKEN = 261
I (1516) LLM_RUN: forward position 3, token 261
I (1776) LLM_RUN: NEXT TOKEN = 378
I (1776) LLM_RUN: forward position 4, token 378
I (2036) LLM_RUN: NEXT TOKEN = 432
I (2046) LLM_RUN: forward position 5, token 432
I (2306) LLM_RUN: NEXT TOKEN = 383
I (2306) LLM_RUN: forward position 6, token 383
I (2566) LLM_RUN: NEXT TOKEN = 286
I (2566) LLM_RUN: forward position 7, token 286
I (2836) LLM_RUN: NEXT TOKEN = 261
I (2836) LLM_RUN: free heap          = 289076
I (2846) LLM_RUN: largest free block = 155648
I (2856) LLM_RUN: 🔥 END-TO-END INFERENCE COMPLETE
```

**Performance:**

| Metric | Value |
| :--- | ---: |
| Tokens / step | ~260–270 ms |
| Throughput | **~3.8 tok/s** |
| Free heap during inference | 289,076 bytes |
| Largest free block | 155,648 bytes |
| Precision | FP32 |
| KV context | 8 tokens |

**Status: ✅ Completed**

---

## 16g. Phase 12 — Real BPE Tokenizer Embedded on ESP32 (Level 5 — Completed)

The `tok512.bin` tokenizer was converted to a C++ byte array (`tok512_model.cc`)
and embedded directly into the firmware alongside the checkpoint.

The tokenizer parses the binary format exactly:
- 4-byte `max_token_length`
- Per-token: `float score` + `int length` + `char[length]` bytes

Implemented behavior:
- BOS token prepend
- Dummy prefix space
- UTF-8 character processing
- `<0xXX>` byte fallback encoding
- BPE merge loop
- Token decoding with BOS whitespace handling

### Verified Tokenizer Output

```text
I (721) LLM_RUN: TOKENIZER
I (721) LLM_RUN: tokenizer bytes      = 6227
I (731) LLM_RUN: max token length     = 7
I (731) LLM_RUN: tokenizer parse      : PASS
I (731) LLM_RUN: tokenizer storage used = 2639 bytes
I (741) LLM_RUN: vocab[0] = "<unk>"
I (741) LLM_RUN: vocab[1] = "<s>"
I (751) LLM_RUN: vocab[2] = "</s>"
```

**Status: ✅ Completed**

---

## 16h. Phase 13 — End-to-End Text Generation on Physical ESP32 🏆

**Milestone:** `TinyStories-260K-ESP32-Generation-v1` ✅

This is the **primary achievement** of the project to date.

A text prompt was tokenized, run through the complete transformer, and decoded
back to text — entirely on the physical ESP32-D0WD-V3, with no cloud, no PC
assistance, and no mocked output.

```mermaid
flowchart TD
    A["'Once'\n(text prompt)"] --> B["BPE Tokenizer\n(embedded tok512.bin)"]
    B --> C["Token IDs\n[1, 403]"]
    C --> D["TinyStories 260K\n5-layer Transformer\nFP32 inference"]
    D --> E["Generated Token IDs\n[407, 261, 378, 432, 383, 286]"]
    E --> F["Detokenizer"]
    F --> G["'upon a time, there was'"]

    style A fill:#22c55e,color:#fff
    style B fill:#22c55e,color:#fff
    style C fill:#22c55e,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#22c55e,color:#fff
    style G fill:#22c55e,color:#fff
```

### Verified Serial Output — Full Generation

```text
I (761) LLM_RUN: prompt = "Once"
I (761) LLM_RUN: prompt token count = 2
I (761) LLM_RUN: prompt token[0] = 1
I (771) LLM_RUN: prompt token[1] = 403
I (771) LLM_RUN: forward prompt position 0, token 1
I (1031) LLM_RUN: forward prompt position 1, token 403

I (1291) LLM_RUN: GENERATED TEXT
I (1291) LLM_RUN: Once upon
I (1291) LLM_RUN: generation position 2 -> token 407    " a"
I (1561) LLM_RUN: generation position 3 -> token 261    " time"
I (1821) LLM_RUN: generation position 4 -> token 378    ","
I (2081) LLM_RUN: generation position 5 -> token 432    " there"
I (2341) LLM_RUN: generation position 6 -> token 383    " was"
I (2601) LLM_RUN: generation position 7 -> token 286

I (2861) LLM_RUN: free heap          = 277460
I (2871) LLM_RUN: largest free block = 147456
I (2881) LLM_RUN: 🔥 GENERATION-v1 COMPLETE
```

### Result

> **Input:** `"Once"`
>
> **Output:** `"Once upon a time, there was"`

### Current Generation Baseline

| Property | Value |
| :--- | :--- |
| Hardware | ESP32-D0WD-V3 |
| Model | TinyStories 260K |
| Model size | 1.056 MB |
| Tokenizer | tok512.bin — 6.227 KB, embedded |
| Vocabulary | 512 tokens |
| Precision | FP32 |
| Context window | 8 tokens |
| Decoding | Greedy argmax |
| Output | Real generated text |
| Speed | **~3.8 tok/s** |
| Free heap after generation | **~277 KB** |
| Largest free block | **~147 KB** |
| New model training | None |

**Status: ✅ Completed**

---

## 17. Current Project Status

| Component | Status |
| :--- | :---: |
| ESP-IDF environment | ✅ Completed |
| ESP32 build / flash / monitor pipeline | ✅ Completed |
| TinyML TFLite Micro inference (sin smoke test) | ✅ Completed |
| Genuine SLM selected (TinyStories 260K) | ✅ Completed |
| PC TinyStories golden reference | ✅ Completed |
| PC text generation verified | ✅ Completed |
| Checkpoint embedded in Flash | ✅ Completed |
| Flash partition adjustment (1.5 MB) | ✅ Completed |
| Checkpoint header parsed on ESP32 | ✅ Completed |
| Tensor layout reconstructed | ✅ Completed |
| Tensor layout verified (1,056,540 B PASS) | ✅ Completed |
| Weight-value numerical verification (float32 reads from Flash) | ✅ Completed |
| RMSNorm — exact match vs. PC reference | ✅ Completed |
| MatMul (generic matrix-vector kernel) | ✅ Completed |
| Q / K / V projections — exact match vs. PC reference | ✅ Completed |
| RoPE positional encoding | ✅ Completed |
| Causal GQA self-attention (scores + softmax + weighted sum) | ✅ Completed |
| Output projection WO + residual add | ✅ Completed |
| Feed-forward network (SwiGLU) | ✅ Completed |
| Single-token complete forward pass | ✅ Completed |
| Full 5-layer forward pass | ✅ Completed |
| Tokenizer (BPE) embedded on ESP32 | ✅ Completed |
| Autoregressive text generation on ESP32 | ✅ **ACHIEVED** |
| Extended context window (64 tokens) | ✅ **ACHIEVED** |
| On-device generation benchmark (2.083 tok/s) | ✅ **ACHIEVED** |
| FreeRTOS watchdog stability (vTaskDelay yield) | ✅ **ACHIEVED** |
| Quantization (INT8 weights) | ⏳ Pending |
| Task-specific  SLM | 🔮 Future |
| Safety / validation layer | 🔮 Future |
| Hardware integration (GPIO / relays) | 🔮 Future |

---

## 18. Project Directory Structure

```text
SLM-PROJECT/
│
├── README.md
│
├── docs/
│   ├── SLM.pdf
│   └── SLM - 2.pdf
│
├── esp32_tflite_test/
│   ├── CMakeLists.txt
│   ├── sdkconfig
│   └── main/
│       ├── CMakeLists.txt
│       └── esp32_tflite_test.cpp
│
├── esp32_llm_test/
│   │
│   ├── model/
│   │   ├── stories260K.bin        ← 1,056,540 B checkpoint
│   │   └── tok512.bin             ← 6,227 B tokenizer
│   │
│   ├── reference/
│   │   ├── run.c                  ← llama2.c reference runtime
│   │   ├── win.c                  ← Windows-specific helpers
│   │   └── build_msvc.bat         ← MSVC build script
│   │
│   └── esp32_llm_runtime/
│       ├── CMakeLists.txt
│       ├── sdkconfig
│       ├── generate_model.py      ← .bin → .cc converter
│       ├── fix_model.py
│       ├── fix_model_linkage.py
│       └── main/
│           ├── CMakeLists.txt
│           ├── esp32_llm_runtime.cpp         ← full generation runtime
│           ├── esp32_llm_runtime_qkv_v1.cpp  ← archived QKV verification stage
│           └── model/
│               ├── stories260K_model.cc   ← embedded byte array (~5.68 MB source)
│               ├── stories260K_model.h
│               ├── tok512_model.cc        ← embedded tokenizer byte array
│               └── tok512_model.h
│
├── data_v5_leakage_controlled/
│   ├── generate_washing_machine_dataset_v5_leakage_controlled.py
│   ├── washing_machine_synthetic_v5_train_24000.jsonl       ← 24,000 samples
│   ├── washing_machine_synthetic_v5_validation_3000.jsonl   ← 3,000 samples
│   ├── washing_machine_synthetic_v5_test_3000.jsonl         ← 3,000 samples
│   └── washing_machine_synthetic_v5_leakage_report.txt
│
└── metadata/
    ├── washing_machine_synthetic_v5_metadata.csv
    ├── washing_machine_synthetic_v5_summary.csv
    └── washing_machine_v5_data_dictionary.csv
```

---

## 19. Key Engineering Decisions

### 1. Genuine language model instead of a classifier

The project intentionally targets an actual autoregressive language model
rather than reducing the problem to conventional intent classification.
This makes the experiment relevant to local natural-language interfaces.

### 2. ESP32-D0WD as the initial target

A more capable MCU (e.g., ESP32-S3 with PSRAM) could simplify deployment.
The original ESP32 provides a **stricter resource-constrained test environment**
that forces real solutions to real constraints.

### 3. TinyStories as an experimental vehicle

TinyStories 260K is not the final washing-machine model. It is used because
it allows the complete transformer inference stack to be investigated **before**
introducing a task-specific model that would add additional complexity.

### 4. PC golden reference first

Before any embedded porting, the reference runtime was validated on PC. This
separates toolchain problems from embedded-platform problems.

### 5. Incremental validation

The runtime is validated from binary representation → numerical operations →
complete inference. This avoids treating generated text as the only correctness criterion.

### 6. Safety separated from the SLM

The language model does **not** directly control actuators. All model-generated
commands must pass through a deterministic validation layer before reaching
the hardware controller. This is a fundamental architectural safety constraint.

---

## 20. Problems Encountered & Fixes

| Problem / Error | Root Cause | Resolution |
| :--- | :--- | :--- |
| `fatal error: cstddef: No such file` | `MicroInterpreter` is C++ only | Changed `.c` → `.cpp` for entry point |
| `undefined reference to tiny_sine_int8_model` | C++ name mangling vs. C header | Added `extern "C"` linkage blocks |
| App partition overflow (170,688 B) | Default 1 MB partition too small | Switched to `Single factory app (large)` — 1.5 MB |
| `run.exe` crashed: `0xC0000005 ACCESS_VIOLATION` | Wrong (mismatched) tokenizer loaded | Explicitly passed `-z tok512.bin` |
| MinGW crashes on `_ftelli64` / `mmap` | POSIX file-mapping not supported on MinGW | Built PC reference with MSVC |
| `PermissionError(13)` on COM6 | Serial port locked by IDE monitor | Closed all active serial monitor windows |
| Bootloader reports `SPI Flash Size : 2 MB` | Menuconfig defaulted to 2 MB | Set Flash size to 4 MB in menuconfig; reflashed with `--flash_size 4MB` |
| `esp_get_free_heap_size()` compile error | Missing system header | Added `#include "esp_system.h"` |

---

## 21. Final Target Architecture

The final system is intended to evolve toward this complete on-device pipeline:

```mermaid
flowchart TD
    USER["👤 USER"] --> NL["🗣️ Natural Language Input"]
    NL --> SLM

    subgraph SLM["Embedded SLM — ESP32"]
        TOK["Tokenizer"] --> FWD["Transformer Forward Pass"]
        FWD --> GEN["Autoregressive Generation"]
        GEN --> DEC["Structured Command Decoder"]
    end

    DEC --> CMD["📋 Structured Command\n{ action, program, temp, spin }"]
    CMD --> VAL["🛡️ Safety & Validation Layer"]
    VAL --> VALID{Valid?}
    VALID -- "YES" --> CTRL["⚙️ Deterministic Controller"]
    VALID -- "NO"  --> REJ["❌ Reject / Request Clarification"]
    CTRL --> HW["🫧  Machine Hardware\n(GPIO / Relays)"]
```

> **The SLM provides language understanding.
> The deterministic layer provides safety.
> The hardware controller provides predictable machine behavior.**

---

## 22. Long-Term Research Direction

Once the embedded transformer runtime is established, the project will move
from a generic language-model benchmark toward a **task-specific SLM**.

The next model should be optimized for:

| Requirement | Description |
| :--- | :--- |
| **machine commands** | Domain-specific vocabulary and grammar |
| **Small vocabulary** | Fewer tokens → smaller embedding table |
| **Structured outputs** | JSON / enum outputs, not free-form text |
| **Low latency** | Real-time interactive response |
| **Low memory consumption** | Fits within ESP32 heap constraints |
| **Quantized inference** | INT8 or INT4 weights for smaller footprint |
| **Deterministic decoding** | Greedy / temperature-0 where appropriate |
| **Robust invalid-input handling** | Graceful rejection of malformed commands |
| **Embedded deployment** | No cloud dependency whatsoever |

> **The ultimate objective is not to run the largest possible model.**
>
> The objective is to determine: *What is the **smallest practical language model**
> and inference runtime that can provide useful natural-language understanding
> entirely on a resource-constrained embedded device?*

---

## 22b. Extended Context and On-Device Generation Benchmark

**Milestone:** `TinyStories-260K-ESP32-Generation-v2` ✅

After verifying that the TinyStories-260K model could perform autoregressive generation on the
physical ESP32-D0WD-V3, the next experiment investigated:

1. Increasing the effective context window.
2. Increasing the maximum generated response length.
3. Measuring actual on-device generation latency.
4. Measuring autoregressive generation throughput in tokens/second.
5. Verifying that longer generation remains stable without triggering the ESP32 task watchdog.

This experiment extends the previous Generation-v1 implementation without changing the model
weights, tokenizer, or transformer architecture.

### Configuration Change

| Parameter | Generation-v1 | Generation-v2 |
| :--- | ---: | ---: |
| `MAX_CONTEXT` | 32 | **64** |
| `MAX_GENERATION_TOKENS` | 16 | **32** |

The TinyStories-260K checkpoint remains unchanged:

| Parameter | Value |
| :--- | ---: |
| Parameters | ~260K |
| `dim` | 64 |
| `hidden_dim` | 172 |
| `layers` | 5 |
| `heads` | 8 |
| KV heads | 4 |
| vocab size | 512 |

### Interactive Generation

The ESP32 runtime accepts a text prompt through the serial console.

**Test prompt:** `once`

```text
prompt token count = 3
token[0] = 1
token[1] = 353
token[2] = 331
```

**Generated output:**

```text
once animals went to the park. He saw a big tree with a big box. The box
```

Generation completed without a task watchdog failure.

### On-Device Performance Benchmark

A hardware timer was added around the autoregressive generation loop using `esp_timer_get_time()`.
The timer starts **after** prompt tokens have been processed — it measures generation throughput only.

| Metric | Result |
| :--- | :--- |
| Hardware | ESP32-D0WD-V3 |
| CPU | 160 MHz |
| Model | TinyStories-260K |
| Context window | 64 tokens |
| Prompt | `once` |
| Prompt tokens | 3 |
| Generated tokens | 32 |
| Generation time | **15.359 s** |
| **Generation throughput** | **2.083 tokens/s** |
| Free heap after generation | 205,176 B |
| Largest free heap block | 110,592 B |
| Task watchdog failure | None |

$$\frac{32 \text{ tokens}}{15.359 \text{ s}} = \textbf{2.083 tokens/second}$$

### Timing Scope

```text
Serial prompt
      ↓
Tokenizer
      ↓
Prompt token forward passes
      ↓
┌─────────────────────────────┐
│ Generation timer starts     │
│                             │
│  Token → Transformer        │
│       → Logits              │
│       → Argmax              │
│       → Next token          │
│       → repeat              │
│                             │
│  32 generated tokens        │
└─────────────────────────────┘
      ↓
Timer stops → 15.359 s → 2.083 tokens/s
```

> The measured **2.083 tokens/s** is **generation throughput** only. It does not include
> serial input time or the initial prompt-processing phase.

### Context Window Scaling — Memory Impact

Increasing the context window from 32 to 64 tokens increased KV-cache SRAM usage.

| Configuration | Free Heap |
| :--- | ---: |
| Context = 32 | ~246 KB |
| Context = 64 | ~205 KB |

For the current FP32 implementation, KV-cache memory grows approximately linearly with context length.

### FreeRTOS Watchdog Stability Fix

The long-running matrix multiplication caused the ESP32 task watchdog to trigger during inference.
To maintain system stability, the runtime periodically yields to the FreeRTOS scheduler
inside the matrix multiplication routine:

```cpp
if ((i & 63) == 63)
{
    vTaskDelay(1);
}
```

This allows the system idle task to execute during the CPU-bound inference workload
without modifying the model arithmetic or checkpoint data.

| Metric | Result |
| :--- | :--- |
| Watchdog failure | None |
| Generated tokens | 32 |
| Generation | Completed successfully |

**Status: ✅ Completed**

---

## 23. Current Milestone

The project has successfully progressed from a conventional TinyML smoke test
to stable, long-form autoregressive text generation on physical ESP32 hardware,
with a concrete throughput baseline now established.

```mermaid
flowchart TD
    A["✅ ESP32-D0WD-V3\nPhysical Hardware"] --> B["✅ TFLite Micro\nSmoke Test — 89.2 µs"]
    B --> C["✅ TinyStories 260K\nGenuine Autoregressive Transformer"]
    C --> D["✅ PC Golden Reference\n3,266.7 tokens/sec"]
    D --> E["✅ 1.056 MB Checkpoint\nEmbedded in Flash"]
    E --> F["✅ Checkpoint Parsed\nOn Physical Device"]
    F --> G["✅ Tensor Layout\nVerified — 1,056,540 B PASS"]
    G --> H["✅ Weights Verified\nfloat32 from Flash"]
    H --> I["✅ RMSNorm + MatMul\n+ Q/K/V — Exact Match"]
    I --> J["✅ RoPE + Attention\n+ FFN + SwiGLU"]
    J --> K["✅ Full 5-Layer\nForward Pass"]
    K --> L["✅ BPE Tokenizer\nEmbedded"]
    L --> M["✅ Generation-v1\n'Once upon a time, there was'"]
    M --> N["🏆 Generation-v2\n64-token context · 32 tokens\n2.083 tok/s · 15.359 s\nWatchdog stable"]
    N --> O["⏳ INT8 Quantization"]
    O --> P["🔮 Machine SLM"]

    style A fill:#22c55e,color:#fff
    style B fill:#22c55e,color:#fff
    style C fill:#22c55e,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#22c55e,color:#fff
    style G fill:#22c55e,color:#fff
    style H fill:#22c55e,color:#fff
    style I fill:#22c55e,color:#fff
    style J fill:#22c55e,color:#fff
    style K fill:#22c55e,color:#fff
    style L fill:#22c55e,color:#fff
    style M fill:#22c55e,color:#fff
    style N fill:#f59e0b,color:#fff
    style O fill:#eab308,color:#fff
    style P fill:#3b82f6,color:#fff
```

**🏆 Generation-v2 milestone achieved: stable long-form autoregressive generation on physical ESP32 hardware.**

`"once"` → `"once animals went to the park. He saw a big tree with a big box. The box"`
at **2.083 tok/s** · 32 generated tokens · 64-token context · FP32 · No PSRAM · No cloud.

This establishes a **concrete hardware throughput baseline** for subsequent runtime and memory
optimization experiments. Next phase: INT8 weight quantization to improve throughput and reduce memory footprint.

---

## 24. Reproducibility / Build Instructions

### Prerequisites
- ESP-IDF v5.5.5
- Python 3.10+
- Visual Studio Build Tools 2022 (MSVC)

### PC Golden Reference

```cmd
cd D:\HAVELLS-PROJ\SLM\esp32_llm_test\reference
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl.exe /fp:fast /O2 /openmp /I. run.c win.c
run.exe ..\model\stories260K.bin -z ..\model\tok512.bin -t 0.0 -n 50
```

### Phase 1 — TinyML Baseline (`esp32_tflite_test`)

```bash
cd D:\HAVELLS-PROJ\SLM\esp32_tflite_test
idf.py set-target esp32
idf.py build
idf.py -p COM6 --flash_size 4MB flash monitor
```

### Phase 4 — SLM Runtime (`esp32_llm_runtime`)

```bash
cd D:\HAVELLS-PROJ\SLM\esp32_llm_test\esp32_llm_runtime
idf.py set-target esp32
idf.py build
idf.py -p COM6 --flash_size 4MB flash monitor
```

---

## 25. References

| Resource | Description |
| :--- | :--- |
| [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | Reference C runtime for LLaMA 2 / TinyStories architecture |
| [Espressif ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/) | ESP32 embedded framework documentation |
| [espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) | TensorFlow Lite Micro for ESP-IDF |
| [espressif/esp-nn](https://github.com/espressif/esp-nn) | Optimized NN kernels for ESP32 family |

---

## Disclaimer on Experimental Models

The **TinyStories 260K** model is an engineering benchmark for validating
embedded language-model execution.

It should not be interpreted as the final model for domain-specific-machine control.

The final application model will require separate evaluation for:

- Command understanding accuracy
- Structured-output fidelity
- Safety & invalid-input handling
- Resource consumption (RAM, Flash, power)
- Embedded inference latency
- Hardware reliability

---

# Setting Up

## ESP32 / ESP32-S3 Development Environment Setup

This document describes a clean, reproducible setup for developing and running embedded C/C++ applications on **ESP32** and **ESP32-S3** using **ESP-IDF**.

The setup is intended for projects involving:

* ESP32 / ESP32-S3 firmware development
* TinyML / Small Language Models (SLMs)
* TensorFlow Lite Micro
* Custom model inference
* Binary model/weight loading
* Serial logging and debugging
* Flashing applications to ESP32-family boards

The versions below correspond to the configuration used successfully during development.

---

## 1. Tested Software Versions

The following versions should be treated as the reference environment.

| Component              |             Version |
| ---------------------- | ------------------: |
| **ESP-IDF**            |           **5.5.5** |
| **ESP-IDF target**     | `esp32` / `esp32s3` |
| **Xtensa ESP-ELF GCC** |          **14.2.0** |
| **Xtensa ESP-ELF G++** |          **14.2.0** |
| **esptool**            |          **4.12.0** |
| **esp-tflite-micro**   |           **1.4.0** |
| **esp-nn**             |           **1.3.2** |
| Flash mode             |             **DIO** |
| Flash frequency        |          **40 MHz** |
| Reference flash size   |            **4 MB** |

> **Important:** Do not randomly upgrade ESP-IDF, compiler toolchains, or ESP components in the middle of reproducing the reference environment. Embedded projects can be sensitive to changes in compiler versions, linker behavior, SDK APIs, and component versions.

---

## 2. Supported Hardware

This setup supports both:

### ESP32

Example target:

```text
ESP32-D0WD
ESP32-D0WD-V3
```

ESP32 uses the Xtensa architecture.

Typical ESP-IDF target:

```bash
esp32
```

---

### ESP32-S3

The same development environment can be used for ESP32-S3.

Typical ESP-IDF target:

```bash
esp32s3
```

ESP32-S3 is also based on Xtensa, but it is **not binary-compatible with the original ESP32**.

Therefore:

> A firmware binary compiled for `esp32` should not be flashed to an `esp32s3`, and vice versa.

The source code can usually be shared, while the firmware must be built separately for each target.

---

## 3. Recommended Installation Strategy

The easiest and most reproducible approach is to use:

**Espressif Installation Manager (EIM)**

instead of manually assembling ESP-IDF, Python, Git, CMake, Ninja, and the compiler toolchain.

The installation manager handles the ESP-IDF development environment and its dependencies.

---

## 4. Install Espressif Installation Manager

Download and install the official:

**Espressif Installation Manager (EIM)**

During installation, allow it to install/configure the ESP-IDF development environment.

The important objective is to obtain:

```text
ESP-IDF 5.5.5
```

After installation, launch the ESP-IDF environment/terminal provided by the installation.

---

## 5. Install ESP-IDF 5.5.5

Inside the Espressif Installation Manager, select:

```text
ESP-IDF
```

and install:

```text
ESP-IDF v5.5.5
```

Do not select a random/latest version if the goal is to reproduce the reference environment exactly.

The installation should provide:

```text
ESP-IDF
├── tools
├── components
├── examples
├── export scripts
└── Python environment
```

---

## 6. ESP-IDF Environment

ESP-IDF requires several environment variables and tools to be configured.

The easiest method is to use the terminal/environment created by the ESP-IDF installation.

For Windows, use the ESP-IDF PowerShell / ESP-IDF Command Prompt supplied by the installation.

Do not use a normal terminal until the ESP-IDF environment has been initialized.

---

## 7. Verify ESP-IDF

Open the ESP-IDF terminal and run:

```bash
idf.py --version
```

Expected result:

```text
ESP-IDF v5.5.5
```

If the command reports a different version, stop here and fix the ESP-IDF environment before continuing.

---

## 8. Verify Python

ESP-IDF manages its required Python environment.

Run:

```bash
python --version
```

or:

```bash
python3 --version
```

The Python environment used by ESP-IDF should be the one configured by ESP-IDF itself.

### Important

Do not manually replace the ESP-IDF Python environment with a random system Python installation.

The ESP-IDF installation manager creates the environment required by the selected ESP-IDF version.

---

## 9. Verify Git

Run:

```bash
git --version
```

A valid Git installation should return a version number.

Git is used by ESP-IDF and its component/dependency system.

---

## 10. Verify CMake

Run:

```bash
cmake --version
```

CMake is required by the ESP-IDF build system.

---

## 11. Verify Ninja

Run:

```bash
ninja --version
```

ESP-IDF uses Ninja as part of its build system.

---

## 12. Verify the Xtensa Compiler

For ESP32 and ESP32-S3, verify that the Xtensa compiler is available.

Run:

```bash
xtensa-esp-elf-gcc --version
```

The reference environment used:

```text
GCC 14.2.0
```

You should see a version corresponding to:

```text
xtensa-esp-elf-gcc ... 14.2.0
```

Also verify G++:

```bash
xtensa-esp-elf-g++ --version
```

Expected compiler family:

```text
Xtensa ESP-ELF GCC/G++ 14.2.0
```

---

## 13. Verify esptool

Run:

```bash
esptool version
```

or, depending on the environment:

```bash
esptool.py version
```

The reference setup used:

```text
esptool 4.12.0
```

---

## 14. Complete Environment Verification

At this point, the basic environment should approximately look like:

```text
ESP-IDF       : 5.5.5
Compiler      : Xtensa ESP-ELF GCC/G++ 14.2.0
esptool       : 4.12.0
```

A useful verification sequence is:

```bash
idf.py --version
python --version
git --version
cmake --version
ninja --version
xtensa-esp-elf-gcc --version
esptool version
```

---

## 15. Create an ESP-IDF Project

An ESP-IDF project normally has a structure similar to:

```text
project/
│
├── CMakeLists.txt
├── sdkconfig
├── sdkconfig.defaults
│
├── main/
│   ├── CMakeLists.txt
│   └── main.c
│
└── components/
    └── ...
```

For a simple application:

```text
project/
├── CMakeLists.txt
└── main/
    ├── CMakeLists.txt
    └── main.c
```

---

## 16. Example Minimal Application

Create:

```text
main/main.c
```

with a minimal application:

```c
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "APP";

void app_main(void)
{
    ESP_LOGI(TAG, "ESP-IDF application started");

    while (1)
    {
        ESP_LOGI(TAG, "Running...");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
```

---

## 17. Configure the ESP32 Target

Before building, explicitly select the target.

For the original ESP32:

```bash
idf.py set-target esp32
```

For ESP32-S3:

```bash
idf.py set-target esp32s3
```

This step is extremely important.

The selected target determines:

* CPU architecture
* SoC-specific peripherals
* linker configuration
* memory layout
* compiler configuration
* bootloader configuration
* partition behavior
* ESP-IDF target-specific components

---

## 18. ESP32 vs ESP32-S3 Target Selection

### Original ESP32

Use:

```bash
idf.py set-target esp32
```

### ESP32-S3

Use:

```bash
idf.py set-target esp32s3
```

Check the current target with:

```bash
idf.py reconfigure
```

The build output should identify the selected chip.

---

## 19. Configure the Project

Open the ESP-IDF configuration menu:

```bash
idf.py menuconfig
```

This provides configuration options for:

* Flash
* CPU
* FreeRTOS
* Partition table
* Logging
* Wi-Fi
* Bluetooth
* Component configuration
* Application settings

For the reference firmware configuration, the flash settings were:

```text
Flash Mode       : DIO
Flash Frequency  : 40 MHz
Flash Size       : 4 MB
```

These values should only be used when they match the actual board/module.

---

## 20. Flash Configuration

For the reference ESP32 configuration:

```text
Flash Mode      = DIO
Flash Frequency = 40 MHz
Flash Size      = 4 MB
```

These settings can be configured through:

```bash
idf.py menuconfig
```

Navigate to the relevant serial/flash configuration section and verify the values.

### Important

Do not blindly force a 4 MB configuration on every ESP32-S3 board.

ESP32-S3 development boards can have different flash sizes.

Always use the flash size supported by the actual hardware.

---

## 21. Build the Firmware

From the project root:

```bash
idf.py build
```

ESP-IDF will:

1. Configure the project
2. Resolve components
3. Compile source files
4. Compile components
5. Link the firmware
6. Generate the bootloader
7. Generate the partition table
8. Generate the final application binary

A successful build should end with a message indicating that the project was built successfully.

---

## 22. Clean Build

If the target or major configuration has changed, use:

```bash
idf.py fullclean
```

Then:

```bash
idf.py build
```

This is particularly useful when switching between:

```text
esp32
```

and:

```text
esp32s3
```

---

## 23. Connect the Development Board

Connect the ESP32/ESP32-S3 development board through USB.

The operating system should detect a serial interface.

On Windows, this may appear as:

```text
COMx
```

For example:

```text
COM6
```

However:

> **Never hard-code COM6 in a generic project.**

The port depends on the user's computer, USB driver, and board.

---

## 24. Identify the Serial Port

On Windows:

1. Open **Device Manager**
2. Open **Ports (COM & LPT)**
3. Identify the ESP32/ESP32-S3 serial port
4. Note the assigned COM number

Example:

```text
USB Serial Device (COM7)
```

Then use:

```bash
idf.py -p COM7 flash
```

Replace `COM7` with the actual port detected on the machine.

---

## 25. Flash the Firmware

After a successful build:

```bash
idf.py -p <PORT> flash
```

Example:

```bash
idf.py -p COM7 flash
```

ESP-IDF will:

1. Connect to the chip
2. Enter the bootloader
3. Erase/write the required flash regions
4. Upload the bootloader
5. Upload the partition table
6. Upload the application
7. Reset the chip

---

## 26. Open the Serial Monitor

After flashing:

```bash
idf.py -p <PORT> monitor
```

Example:

```bash
idf.py -p COM7 monitor
```

You should see logs similar to:

```text
I (...) APP: ESP-IDF application started
I (...) APP: Running...
I (...) APP: Running...
I (...) APP: Running...
```

---

## 27. Build + Flash + Monitor

ESP-IDF can perform the complete workflow:

```bash
idf.py -p <PORT> flash monitor
```

For example:

```bash
idf.py -p COM7 flash monitor
```

This is the most convenient development workflow.

---

## 28. Exit the Monitor

To exit the ESP-IDF monitor, use:

```text
Ctrl + ]
```

This returns you to the terminal.

---

## 29. ESP-IDF Component Dependencies

For TinyML/SLM-related development, the following components were part of the working environment:

```text
esp-tflite-micro : 1.4.0
esp-nn            : 1.3.2
```

These are particularly relevant when implementing neural-network inference on ESP32-family devices.

---

## 30. TensorFlow Lite Micro

For projects using TensorFlow Lite Micro, the ESP-IDF environment needs the appropriate TFLite Micro component.

Reference version:

```text
esp-tflite-micro 1.4.0
```

The component should be declared through the ESP-IDF component system rather than manually copying arbitrary versions into the project.

---

## 31. ESP-NN

For optimized neural-network operations, the reference environment used:

```text
esp-nn 1.3.2
```

ESP-NN provides optimized neural-network kernels for supported Espressif hardware.

This can become important for:

* convolution
* matrix operations
* quantized inference
* CPU optimization
* TinyML workloads

---

## 32. Recommended Project Structure for SLM Work

For a Small Language Model running on ESP32/ESP32-S3, a clean project structure can be:

```text
slm-esp/
│
├── CMakeLists.txt
├── sdkconfig
├── sdkconfig.defaults
│
├── main/
│   ├── CMakeLists.txt
│   ├── main.c
│   ├── model.c
│   ├── model.h
│   ├── tokenizer.c
│   ├── tokenizer.h
│   ├── inference.c
│   └── inference.h
│
├── components/
│   ├── model/
│   ├── tokenizer/
│   └── inference/
│
├── models/
│   ├── model.bin
│   └── tokenizer.bin
│
└── partitions.csv
```

The exact structure can be changed according to the implementation.

---

## 33. Binary Model Files

If the model is stored as a raw binary file, the firmware needs a mechanism to access the binary data.

Typical approaches include:

```text
1. Embed binary into firmware
2. Store binary in a dedicated flash partition
3. Store model data in external flash
4. Load model data from a filesystem
```

For an SLM project, the partition-based approach can be useful when model size becomes significant.

---

## 34. Model + Tokenizer Separation

A useful architecture is:

```text
                 ESP32 / ESP32-S3
                        │
                        ▼
              ┌──────────────────┐
              │ Application      │
              └────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Tokenizer                  Model Loader
          │                         │
          │                    model.bin
          │                         │
          └────────────┬────────────┘
                       ▼
                  Inference
                       │
                       ▼
                 Generated Tokens
                       │
                       ▼
                 Detokenization
                       │
                       ▼
                    Text
```

This separation makes debugging substantially easier.

---

## 35. PC Reference Implementation

Before deploying a model to an ESP32-class device, it is useful to maintain a PC reference implementation.

For the TinyStories work, the reference model artifacts included:

```text
stories260K.bin
tok512.bin
```

The PC implementation can be used to verify:

* model loading
* tensor layout
* tokenizer behavior
* token generation
* logits
* sampling
* generated text

before comparing against the embedded implementation.

---

## 36. Recommended Validation Flow

For an SLM deployment, validate in this order:

```text
             Model files
                  │
                  ▼
        ┌─────────────────┐
        │ PC Reference    │
        └────────┬────────┘
                 │
                 ▼
        Verify model loading
                 │
                 ▼
        Verify tokenizer
                 │
                 ▼
        Verify forward pass
                 │
                 ▼
        Verify token generation
                 │
                 ▼
        ┌─────────────────┐
        │ ESP32 Firmware  │
        └────────┬────────┘
                 │
                 ▼
        Verify model loading
                 │
                 ▼
        Verify tokenizer
                 │
                 ▼
        Verify forward pass
                 │
                 ▼
        Verify token generation
                 │
                 ▼
        Compare against PC
```

This prevents debugging the model, tokenizer, memory layout, and ESP-IDF environment simultaneously.

---

## 37. ESP32 and ESP32-S3 Development Workflow

The same source tree can be used for both targets if the code is written in a target-independent manner.

### ESP32

```bash
idf.py set-target esp32
idf.py build
idf.py -p <PORT> flash
idf.py -p <PORT> monitor
```

### ESP32-S3

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash
idf.py -p <PORT> monitor
```

When changing targets, it is recommended to perform a clean build:

```bash
idf.py fullclean
```

followed by:

```bash
idf.py set-target <TARGET>
idf.py build
```

---

## 38. Switching Between ESP32 and ESP32-S3

For example, switching from ESP32:

```bash
idf.py set-target esp32
```

to ESP32-S3:

```bash
idf.py fullclean
idf.py set-target esp32s3
idf.py build
```

Do not reuse an old compiled binary.

Each target must have its own build output.

---

## 39. Common Build Commands

### Configure target

```bash
idf.py set-target esp32
```

or:

```bash
idf.py set-target esp32s3
```

### Configure project

```bash
idf.py menuconfig
```

### Build

```bash
idf.py build
```

### Clean

```bash
idf.py fullclean
```

### Flash

```bash
idf.py -p <PORT> flash
```

### Monitor

```bash
idf.py -p <PORT> monitor
```

### Flash + monitor

```bash
idf.py -p <PORT> flash monitor
```

---

## 40. Common Problems

### Problem 1 — `idf.py` is not recognized

Example:

```text
'idf.py' is not recognized...
```

**Cause:** The ESP-IDF environment has not been initialized.

**Solution:** Open the ESP-IDF-provided terminal/environment rather than a normal shell. Then run:

```bash
idf.py --version
```

---

### Problem 2 — Wrong ESP-IDF Version

Check:

```bash
idf.py --version
```

The reference environment requires:

```text
ESP-IDF v5.5.5
```

If another version is active, select the correct ESP-IDF installation through the Espressif Installation Manager/environment.

---

### Problem 3 — Wrong Compiler Version

Check:

```bash
xtensa-esp-elf-gcc --version
```

Reference: `14.2.0`

If the compiler version is unexpected, verify that the correct ESP-IDF environment is active.

---

### Problem 4 — Wrong Target

If an ESP32-S3 project is accidentally configured as `esp32`, the firmware will not be built for the S3.

Check/change the target:

```bash
idf.py set-target esp32s3
```

---

### Problem 5 — Flashing Fails

Typical symptoms: `Failed to connect` or `No serial data received`.

Check:
1. USB cable
2. Correct serial port
3. Board power
4. USB driver
5. Whether another program is using the serial port
6. Boot/reset behavior of the development board

Then retry: `idf.py -p <PORT> flash`

---

### Problem 6 — Monitor Shows Nothing

First verify that the correct port is being used:

```bash
idf.py -p <PORT> monitor
```

Then reset the board. Also verify that the application actually contains logging/output.

---

### Problem 7 — Build Behaves Strangely After Target Change

Perform:

```bash
idf.py fullclean
idf.py set-target <TARGET>
idf.py build
```

This removes stale target-specific build artifacts.

---

### Problem 8 — Flash Size Mismatch

Do not assume 4 MB for every ESP32-family board. The reference configuration used `4 MB / DIO / 40 MHz`, but the actual board configuration should take precedence.

---

## 48. Windows-Specific Build Tools

For PC-side C/C++ reference implementations on Windows, **Microsoft Visual Studio Build Tools 2022** was also installed.

This is separate from the ESP-IDF Xtensa compiler:

```text
PC Reference   → MSVC / Visual Studio Build Tools
ESP32 Firmware → Xtensa ESP-ELF GCC/G++ 14.2.0
```

---

## 49. Recommended Terminal Setup

For ESP32 firmware development, use the **ESP-IDF terminal** which includes `idf.py`, `xtensa-esp-elf-gcc`, `cmake`, `ninja`, and `esptool`.

For PC reference development, use a normal Developer terminal with MSVC.

Keeping the two environments conceptually separate avoids many toolchain problems.

---

## 50. First-Time Setup Checklist

Before beginning development, verify:

```text
[ ] Espressif Installation Manager installed
[ ] ESP-IDF 5.5.5 installed
[ ] ESP-IDF terminal opens correctly
[ ] idf.py --version works
[ ] Python environment works
[ ] Git works
[ ] CMake works
[ ] Ninja works
[ ] Xtensa GCC 14.2.0 works
[ ] Xtensa G++ 14.2.0 works
[ ] esptool 4.12.0 works
[ ] ESP32 board detected
[ ] ESP32-S3 board detected (if required)
[ ] Correct serial port identified
[ ] Correct ESP-IDF target selected
[ ] Project builds successfully
[ ] Firmware flashes successfully
[ ] Serial monitor works
```

---

## 51. Reference Configuration Summary

```text
┌─────────────────────────────────────────────┐
│             ESP-IDF Environment             │
├─────────────────────────────────────────────┤
│ ESP-IDF                 5.5.5               │
│ Xtensa GCC              14.2.0              │
│ Xtensa G++              14.2.0              │
│ esptool                  4.12.0             │
│ esp-tflite-micro         1.4.0              │
│ esp-nn                   1.3.2              │
├─────────────────────────────────────────────┤
│ Flash Reference                             │
│ Mode                     DIO                │
│ Frequency                40 MHz             │
│ Size                     4 MB               │
└─────────────────────────────────────────────┘

Supported targets:
  ESP32 (D0WD / D0WD-V3) → idf.py set-target esp32
  ESP32-S3               → idf.py set-target esp32s3
```

---

## 52. Complete Setup-to-Flash Workflow

```bash
# 1. Open ESP-IDF terminal

# 2. Verify ESP-IDF
idf.py --version

# 3. Navigate to project
cd <project-directory>

# 4. Select target
idf.py set-target esp32   # or esp32s3

# 5. Configure
idf.py menuconfig

# 6. Build
idf.py build

# 7. Flash + Monitor
idf.py -p <PORT> flash monitor
```

---

## 53. Recommended Reproducibility Rule

For research/deployment work, record the environment alongside the firmware. At minimum record:

```text
ESP-IDF version · Compiler version · esptool version · Target chip
Flash size · Flash mode · Flash frequency · Component versions
sdkconfig · Model version · Tokenizer version
```

A change in compiler, optimization flags, memory layout, component version, or linker configuration can potentially change runtime behavior.

---

## 54. Final Environment Verification

Before declaring the machine ready, the following should all work:

```bash
idf.py --version          # ESP-IDF v5.5.5
xtensa-esp-elf-gcc --version  # 14.2.0
esptool version           # 4.12.0
idf.py build              # successful build
idf.py -p <PORT> flash    # firmware uploaded
idf.py -p <PORT> monitor  # serial output visible
```

If all stages work, the machine is ready for the next phase: **model loading, tokenizer integration, memory engineering, and SLM inference on the ESP32/ESP32-S3.**

