# ESP32-S3 Deployment Feasibility Sandbox Report

## 1. Hardware Constraints & Profile

- **Hardware Target**: `ESP32-S3-N16R8 (Default Project Profile)`
- **CPU**: Dual-Core Xtensa LX7 @ 240 MHz
- **Internal SRAM**: `512.0` KB
- **ROM**: `384.0` KB | **RTC Fast**: `8.0` KB | **RTC Slow**: `8.0` KB
- **I-Cache**: `32.0` KB | **D-Cache**: `64.0` KB
- **External Flash**: `16.0` MB (Octal SPI (OPI) / QPI)
- **External PSRAM**: `8.0` MB (Octal SPI (OPI))

## 2. Model Architecture & Parameter Accounting

- **Checkpoint Step**: `5750`
- **Architecture**: Decoder-only PLE Transformer (`d_model=128, L=6, H=4, F=256, P=64`)
- **Total Parameters**: `1,657,280`
  - **Core Parameters** ($N_{\text{core}}$): `1,132,992`
  - **PLE Table Parameters** ($N_{\text{table}}$): `393,216`
  - **Output Head Parameters** ($N_{\text{stream}}$): `131,072`

## 3. Precision Memory Footprints

| Precision | Total Storage | Core Footprint | PLE Table Footprint | Stream Head |
|---|---|---|---|---|
| **FP32** | `6.32 MB` | `4425.8 KB` | `1536.0 KB` | `512.0 KB` |
| **FP16** | `3.16 MB` | `2212.9 KB` | `768.0 KB` | `256.0 KB` |
| **INT8** | `1.58 MB` | `1106.4 KB` | `384.0 KB` | `128.0 KB` |
| **INT4 (group=64)** | **`0.84 MB` (859.8 KB)** | `587.8 KB` | `204.0 KB` | `68.0 KB` |

## 4. Memory Placement Architecture & Hierarchy

### FLASH Memory (16.0 MB Available)
- Mapped INT4 Quantized Model Image: `859.8 KB`
- Tokenizer constants (BTK1 header): `~43.0 KB`
- Firmware binary & constants reserve: `~256.0 KB`
- **Total Flash Used**: `1158.8 KB` (`1.13 MB` / `16.0 MB` — **`7.07%`**)
- **Flash Headroom**: `14.87 MB` $\rightarrow$ **FIT** ✅

### External PSRAM (8.0 MB Available)
- Staged INT8 Core Dense Weights: `1106.4 KB`
- KV-Cache at `ctx=128` ($2 \times 6 \times 128 \times 128 \times 4$ B): `768.0 KB`
- PSRAM Scratch / Working Reserve: `128.0 KB`
- **Total PSRAM Used**: `2002.4 KB` (`1.96 MB` / `8.0 MB` — **`24.44%`**)
- **PSRAM Headroom**: `6.04 MB` (`6189.6 KB`) $\rightarrow$ **FIT** ✅

### Internal SRAM (512 KB Available)
- Exact Hot Model Working Buffers (`llm.h` `Scratch` struct): `24.25 KB`
- Runtime Stack Reserve: `16.0 KB`
- Tokenizer & JSON Runtime: `6.0 KB`
- Configurable OS/Network Runtime Reserve (`SRAM_RUNTIME_RESERVE_KB`): `128.0 KB`
- **Total SRAM Used (Realistic)**: `174.2 KB` (**`34.03%`** of 512 KB)
- **SRAM Headroom (Realistic)**: `337.8 KB` $\rightarrow$ **FIT** ✅
- **Total SRAM Used (Theoretical Reserve=0)**: `46.2 KB` (`9.03%` of 512 KB) $\rightarrow$ **FIT** ✅

## 5. Context-Length Analytical Sweep

| Context | KV-Cache (KB) | SRAM Used (KB) | PSRAM Used (KB) | Total RAM (MB) | Status |
|---|---|---|---|---|---|
| `32` | `192.0` | `173.9` | `1426.4` | `1.56 MB` | **`FIT`** |
| `64` | `384.0` | `174.0` | `1618.4` | `1.75 MB` | **`FIT`** |
| `96` | `576.0` | `174.1` | `1810.4` | `1.94 MB` | **`FIT`** |
| `128` | `768.0` | `174.2` | `2002.4` | `2.13 MB` | **`FIT`** |
| `160` | `960.0` | `174.4` | `2194.4` | `2.31 MB` | **`FIT`** |
| `192` | `1152.0` | `174.5` | `2386.4` | `2.5 MB` | **`FIT`** |
| `256` | `1536.0` | `174.8` | `2770.4` | `2.88 MB` | **`FIT`** |

## 6. Generation-Length Feasibility Analysis

| Generated Tokens | Active Seq Len | Active KV-Cache | Total PSRAM | PSRAM Headroom |
|---|---|---|---|---|
| `16` | `26` | `156.0 KB` | `1390.4 KB` | `6.64 MB` |
| `32` | `42` | `252.0 KB` | `1486.4 KB` | `6.55 MB` |
| `48` | `58` | `348.0 KB` | `1582.4 KB` | `6.45 MB` |
| `64` | `74` | `444.0 KB` | `1678.4 KB` | `6.36 MB` |
| `80` | `90` | `540.0 KB` | `1774.4 KB` | `6.27 MB` |
| `96` | `106` | `636.0 KB` | `1870.4 KB` | `6.17 MB` |
| `128` | `138` | `828.0 KB` | `2062.4 KB` | `5.99 MB` |

## 7. Functional PC Reference Inferences

- **Input**: `pause the washing machine`
- **Output**: `{"intent":"PAUSE"}`
- **Tokens**: `5` | **Valid JSON**: `True`
- **Disclaimer**: `ESP32 latency not measured; analytical estimate only.`

- **Input**: `set the spin to 600 rpm`
- **Output**: `{"intent":"SET_SPIN_SPEED","spin_speed":600}`
- **Tokens**: `15` | **Valid JSON**: `True`
- **Disclaimer**: `ESP32 latency not measured; analytical estimate only.`

- **Input**: `run the cotton wash at 30 degrees`
- **Output**: `{"intent":"START_CYCLE","program":"COTTON"}`
- **Tokens**: `11` | **Valid JSON**: `True`
- **Disclaimer**: `ESP32 latency not measured; analytical estimate only.`

## 8. Main Bottlenecks & Overall Recommendation

1. **PSRAM Pressure**: At `ctx=128`, model uses 1.96 MB of PSRAM. On ESP32-S3-N16R8 (8MB PSRAM), headroom is ample (**6.19 MB**). On an 8MB Flash / 2MB PSRAM module (N8R2), headroom is tight (**43.6 KB remaining**).
2. **SRAM Pressure**: Internal SRAM usage is **148.4 KB** out of 512 KB (leaving **363.6 KB** free for FreeRTOS and WiFi/Bluetooth stacks).
3. **Largest Feasible Context**: **`ctx=512+`** on 8MB PSRAM; **`ctx=128`** on 2MB PSRAM.
4. **Recommended Deployment Precision**: **INT4 Weights (group=64) with INT8 Core Staging in PSRAM**.

### Final Verdict: **GO (FEASIBLE)** ✅
