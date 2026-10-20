"""run_constrained_inference.py — ESP32-S3 Constrained Inference Sandbox
========================================================================
Interactive & CLI inference sandbox enforcing ESP32-S3 memory and hardware
resource constraints for the Washing Machine SLM.

Reads:
  E:\\havells\\implementation\\washing_machine\\checkpoints\\best_washer_slm.pt
  E:\\havells\\implementation\\washing_machine\\tokenizer\\tokenizer.json

Appends metrics history to:
  E:\\havells\\implementation\\washing_machine\\esp32_s3_sandbox\\interactive_results.csv

Usage:
  CLI single prompt test:
    python E:\\havells\\implementation\\washing_machine\\esp32_s3_sandbox\\run_constrained_inference.py --prompt "pause the washing machine"

  Interactive mode:
    python E:\\havells\\implementation\\washing_machine\\esp32_s3_sandbox\\run_constrained_inference.py
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
from tokenizers import Tokenizer

# Ensure repository root is in sys.path
ROOT_DIR = Path(r"E:\havells")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from implementation.washing_machine.model_config import Config, TinyLM

# Paths
WASH_DIR = ROOT_DIR / "implementation" / "washing_machine"
SANDBOX_DIR = WASH_DIR / "esp32_s3_sandbox"
CKPT_PATH = WASH_DIR / "checkpoints" / "best_washer_slm.pt"
TOK_PATH = WASH_DIR / "tokenizer" / "tokenizer.json"
CSV_PATH = SANDBOX_DIR / "interactive_results.csv"

# Special Tokens
BOS_ID = 190
SEP_ID = 219
EOS_ID = 191
MAX_NEW_TOKENS = 80
MAX_CONTEXT = 128

# ESP32-S3 Hardware Limits
CPU_FREQ_MHZ = 240
TOTAL_SRAM_KB = 512.0
SRAM_RESERVE_KB = 128.0
USABLE_SRAM_KB = TOTAL_SRAM_KB - SRAM_RESERVE_KB  # 384.0 KB
TOTAL_PSRAM_KB = 8.0 * 1024.0                     # 8,192.0 KB (8 MB)
TOTAL_FLASH_KB = 16.0 * 1024.0                    # 16,384.0 KB (16 MB)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def calculate_esp32_memory_ledger(cfg: Config, budget: Dict[str, int], prompt_tokens_count: int, gen_tokens_count: int) -> Dict[str, Any]:
    total_seq_len = prompt_tokens_count + gen_tokens_count

    # Flash: INT4 Quantized Model Image (0.53125 B/param) + Tokenizer assets (~43 KB) + Firmware reserve (256 KB)
    int4_model_kb = (budget["total"] * 0.53125) / 1024.0
    flash_used_kb = int4_model_kb + 43.0 + 256.0
    flash_available_kb = TOTAL_FLASH_KB
    flash_headroom_kb = flash_available_kb - flash_used_kb
    flash_fit = flash_headroom_kb >= 0

    # PSRAM: Staged INT8 Core Weights (1 B/param) + KV-Cache (6,144 * total_seq_len bytes) + PSRAM scratch (128 KB)
    staged_core_int8_kb = (budget["core"] * 1.0) / 1024.0
    kv_cache_bytes = 2 * cfg.n_layers * total_seq_len * cfg.d_model * 4
    kv_cache_kb = kv_cache_bytes / 1024.0
    psram_used_kb = staged_core_int8_kb + kv_cache_kb + 128.0
    psram_available_kb = TOTAL_PSRAM_KB
    psram_headroom_kb = psram_available_kb - psram_used_kb
    psram_fit = psram_headroom_kb >= 0

    # SRAM: Hot activation vectors, RMSNorm gains, attention scratch, stack reserve (16 KB), JSON buffer, tokenizer runtime
    D = cfg.d_model
    F = cfg.ffn_hidden
    P = cfg.ple_dim
    V = cfg.vocab_size
    hot_buffers_bytes = (
        (D * 4) + (D * 4) + (3 * D * 4) + (D * 4) + (F * 4) + (F * 4) +
        (cfg.n_layers * P * 4 * 3) + (total_seq_len * 4) + (V * 4) + max(D, F, P) + ((cfg.n_layers * 3 + 2) * D * 4)
    )
    sram_hot_kb = hot_buffers_bytes / 1024.0
    sram_used_kb = SRAM_RESERVE_KB + sram_hot_kb + 16.0 + 4.0 + 2.0  # 128 reserve + hot + 16 stack + 4 tok + 2 json
    sram_available_kb = TOTAL_SRAM_KB
    sram_headroom_kb = sram_available_kb - sram_used_kb
    sram_fit = sram_used_kb <= TOTAL_SRAM_KB

    # Context fit
    context_fit = total_seq_len <= MAX_CONTEXT

    overall_fit = flash_fit and psram_fit and sram_fit and context_fit

    failed_resources = []
    if not flash_fit: failed_resources.append("FLASH")
    if not psram_fit: failed_resources.append("PSRAM")
    if not sram_fit: failed_resources.append("SRAM")
    if not context_fit: failed_resources.append("CONTEXT_WINDOW")

    return {
        "flash_used_kb": round(flash_used_kb, 1),
        "flash_available_kb": flash_available_kb,
        "flash_headroom_kb": round(flash_headroom_kb, 1),
        "flash_fit": flash_fit,
        "psram_used_kb": round(psram_used_kb, 1),
        "psram_available_kb": psram_available_kb,
        "psram_headroom_kb": round(psram_headroom_kb, 1),
        "staged_core_int8_kb": round(staged_core_int8_kb, 1),
        "kv_cache_kb": round(kv_cache_kb, 1),
        "kv_cache_bytes": kv_cache_bytes,
        "psram_fit": psram_fit,
        "sram_used_kb": round(sram_used_kb, 1),
        "usable_sram_kb": USABLE_SRAM_KB,
        "sram_available_kb": sram_available_kb,
        "sram_headroom_kb": round(sram_headroom_kb, 1),
        "sram_fit": sram_fit,
        "total_seq_len": total_seq_len,
        "max_context": MAX_CONTEXT,
        "context_fit": context_fit,
        "overall_fit": overall_fit,
        "failed_resources": failed_resources,
    }


def calculate_analytical_esp32_ops(cfg: Config, budget: Dict[str, int], total_tokens: int) -> Tuple[int, int, str]:
    """Analytical floating point operations / cycles estimate per token on ESP32-S3."""
    # Core dense GEMM ops per token ≈ 2 * core_params
    ops_per_token = 2 * budget["core"]
    # At 240 MHz, assuming ~1.5 cycles per MAC on Xtensa SIMD
    cycles_per_token = int(ops_per_token * 0.75)
    disclaimer = "ESP32-S3 speed: NOT MEASURED (analytical estimate only)"
    return ops_per_token, cycles_per_token, disclaimer


@torch.no_grad()
def run_constrained_prompt(
    model: TinyLM,
    tok: Tokenizer,
    budget: Dict[str, int],
    user_input: str,
    device: str
) -> Dict[str, Any]:
    model.eval()

    # 1. Encode Input
    input_tokens = tok.encode(user_input, add_special_tokens=False).ids
    prompt_ids = [BOS_ID] + input_tokens + [SEP_ID]
    prompt_token_count = len(prompt_ids)

    # 2. Pre-check resource limits before running
    pre_ledger = calculate_esp32_memory_ledger(model.cfg, budget, prompt_token_count, 1)
    if not pre_ledger["overall_fit"]:
        print("\n" + "!" * 60)
        print("RESOURCE LIMIT EXCEEDED BEFORE INFERENCE!")
        print(f"Failed resources: {', '.join(pre_ledger['failed_resources'])}")
        print("!" * 60 + "\n")
        return {"resource_exceeded": True, "ledger": pre_ledger}

    # 3. Measure PC Prefill Time
    t_start = time.perf_counter()
    curr_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(curr_tensor)
    t_prefill_end = time.perf_counter()
    prefill_ms = (t_prefill_end - t_start) * 1000.0

    # 4. Greedy Autoregressive Generation & Decode Timing
    gen_ids = []
    t_decode_start = time.perf_counter()

    allowed_gen = min(MAX_NEW_TOKENS, MAX_CONTEXT - prompt_token_count)

    for step in range(allowed_gen):
        input_window = curr_tensor[:, -MAX_CONTEXT:]
        logits, _ = model(input_window)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()

        if next_token_id == EOS_ID:
            break

        gen_ids.append(next_token_id)
        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        curr_tensor = torch.cat([curr_tensor, next_tensor], dim=1)

        # Check budget fit dynamically as context grows
        step_ledger = calculate_esp32_memory_ledger(model.cfg, budget, prompt_token_count, len(gen_ids))
        if not step_ledger["overall_fit"]:
            print("\n" + "!" * 60)
            print(f"RESOURCE LIMIT EXCEEDED AT GENERATION STEP {step + 1}!")
            print(f"Failed resources: {', '.join(step_ledger['failed_resources'])}")
            print("!" * 60 + "\n")
            break

    t_decode_end = time.perf_counter()
    decode_ms = (t_decode_end - t_decode_start) * 1000.0
    total_ms = (t_decode_end - t_start) * 1000.0

    gen_count = len(gen_ids)
    total_tokens = prompt_token_count + gen_count

    pc_tokens_per_sec = (gen_count / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
    pc_ms_per_token = (decode_ms / gen_count) if gen_count > 0 else 0.0

    # 5. Decode text & Validate JSON
    gen_text = tok.decode(gen_ids).strip()
    is_valid_json = False
    try:
        parsed = json.loads(gen_text)
        is_valid_json = isinstance(parsed, dict)
    except Exception:
        is_valid_json = False

    # 6. Final Resource Ledger
    ledger = calculate_esp32_memory_ledger(model.cfg, budget, prompt_token_count, gen_count)

    # 7. Analytical ESP32 Compute Estimate
    esp32_ops, esp32_cycles, esp32_disclaimer = calculate_analytical_esp32_ops(model.cfg, budget, total_tokens)

    # PC Execution Memory Footprint
    if device == "cuda":
        pc_exec_mem = f"{torch.cuda.max_memory_allocated() / (1024*1024):.2f} MB (CUDA VRAM)"
    else:
        pc_exec_mem = f"~{(budget['total'] * 4) / (1024*1024):.2f} MB (System PyTorch FP32)"

    return {
        "resource_exceeded": False,
        "input_text": user_input,
        "output_text": gen_text,
        "is_valid_json": is_valid_json,
        "prompt_tokens_count": prompt_token_count,
        "gen_tokens_count": gen_count,
        "total_tokens_count": total_tokens,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "pc_tokens_per_sec": round(pc_tokens_per_sec, 2),
        "pc_ms_per_token": round(pc_ms_per_token, 2),
        "esp32_ops_per_token": esp32_ops,
        "esp32_cycles_per_token": esp32_cycles,
        "esp32_disclaimer": esp32_disclaimer,
        "pc_exec_mem": pc_exec_mem,
        "ledger": ledger
    }


def print_prompt_results(res: Dict[str, Any]):
    if res.get("resource_exceeded"):
        return

    l = res["ledger"]

    print("\n" + "-" * 65)
    print("CATEGORY 1: FUNCTIONAL MODEL OUTPUT")
    print(f"INPUT:               {res['input_text']}")
    print(f"OUTPUT:              {res['output_text']}")
    print(f"VALID JSON:          {res['is_valid_json']}")
    print(f"PROMPT TOKENS:       {res['prompt_tokens_count']}")
    print(f"TOKENS GENERATED:    {res['gen_tokens_count']}")
    print(f"TOTAL TOKENS:        {res['total_tokens_count']}")

    print("\nCATEGORY 2: PC MEASURED GENERATION SPEED (PC REFERENCE ONLY)")
    print(f"PREFILL TIME:        {res['prefill_ms']} ms")
    print(f"DECODE TIME:         {res['decode_ms']} ms")
    print(f"TOTAL TIME:          {res['total_ms']} ms")
    print(f"PC REFERENCE SPEED:  {res['pc_tokens_per_sec']} tokens/sec ({res['pc_ms_per_token']} ms/token)")

    print("\nCATEGORY 3: ESP32-S3 ANALYTICAL COMPUTE ESTIMATE")
    print(f"ESP32 SPEED STATUS:  {res['esp32_disclaimer']}")
    print(f"OPS / TOKEN:         {res['esp32_ops_per_token']:,} FLOPs")
    print(f"ESTIMATED CYCLES:    ~{res['esp32_cycles_per_token']:,} cycles/token (@ 240 MHz)")

    print("\nCATEGORY 4: ESP32-S3 MODELED MEMORY FEASIBILITY & LEDGER")
    print(f"PC EXECUTION MEMORY: {res['pc_exec_mem']}")
    print(f"MODEL WEIGHT INT4:   {l['flash_used_kb'] - 299.0:.1f} KB (Flash image format)")
    print(f"STAGED CORE INT8:    {l['staged_core_int8_kb']} KB (PSRAM staged)")
    print(f"KV CACHE MEMORY:     {l['kv_cache_kb']} KB ({l['kv_cache_bytes']:,} bytes)")
    print(f"SRAM USED:           {l['sram_used_kb']} KB / {l['sram_available_kb']} KB (Usable Budget: {l['usable_sram_kb']} KB)")
    print(f"PSRAM USED:          {l['psram_used_kb']} KB / {l['psram_available_kb']} KB ({l['psram_used_kb']/1024:.2f} MB / 8 MB)")
    print(f"FLASH USED:          {l['flash_used_kb']} KB / {l['flash_available_kb']} KB ({l['flash_used_kb']/1024:.2f} MB / 16 MB)")
    print(f"CONTEXT USED:        {l['total_seq_len']} / {MAX_CONTEXT} max context tokens")
    print(f"MEMORY STATUS:       {'FIT' if l['overall_fit'] else 'NOT FIT'}")
    print("-" * 65 + "\n")


def append_to_csv(res: Dict[str, Any]):
    if res.get("resource_exceeded"):
        return

    l = res["ledger"]
    file_exists = CSV_PATH.exists()

    fieldnames = [
        "timestamp", "input", "output", "valid_json", "input_tokens",
        "output_tokens", "total_tokens", "prefill_ms", "decode_ms", "total_ms",
        "pc_tokens_per_sec", "esp32_estimated_ops", "esp32_estimated_cycles",
        "sram_used_kb", "psram_used_kb", "flash_used_kb", "memory_fit"
    ]

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(),
            "input": res["input_text"],
            "output": res["output_text"],
            "valid_json": res["is_valid_json"],
            "input_tokens": res["prompt_tokens_count"],
            "output_tokens": res["gen_tokens_count"],
            "total_tokens": res["total_tokens_count"],
            "prefill_ms": res["prefill_ms"],
            "decode_ms": res["decode_ms"],
            "total_ms": res["total_ms"],
            "pc_tokens_per_sec": res["pc_tokens_per_sec"],
            "esp32_estimated_ops": res["esp32_ops_per_token"],
            "esp32_estimated_cycles": res["esp32_cycles_per_token"],
            "sram_used_kb": l["sram_used_kb"],
            "psram_used_kb": l["psram_used_kb"],
            "flash_used_kb": l["flash_used_kb"],
            "memory_fit": l["overall_fit"],
        })


def main():
    parser = argparse.ArgumentParser(description="ESP32-S3 Constrained Inference Sandbox")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt CLI test mode")
    args = parser.parse_args()

    device = get_device()
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")
    if not TOK_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOK_PATH}")

    # Load Checkpoint & Model
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = Config(**ckpt["model_config"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    budget = model.param_budget()
    tok = Tokenizer.from_file(str(TOK_PATH))

    # Print Startup Display
    print("=" * 70)
    print("ESP32-S3 CONSTRAINED INFERENCE SANDBOX")
    print("=" * 70)
    print(f"Checkpoint:             {CKPT_PATH}")
    print(f"Best checkpoint step:   {ckpt.get('step', 'N/A')}")
    print(f"Model parameters:       {budget['total']:,} (Core: {budget['core']:,}, Table: {budget['table']:,})")
    print(f"Tokenizer:              {TOK_PATH.name}")
    print(f"Context:                {MAX_CONTEXT} tokens")
    print(f"CPU:                    Xtensa LX7 @ {CPU_FREQ_MHZ} MHz")
    print(f"SRAM:                   {TOTAL_SRAM_KB} KB total")
    print(f"PSRAM:                  {TOTAL_PSRAM_KB/1024:.1f} MB ({TOTAL_PSRAM_KB:,} KB)")
    print(f"Flash:                  {TOTAL_FLASH_KB/1024:.1f} MB ({TOTAL_FLASH_KB:,} KB)")
    print(f"SRAM runtime reserve:   {SRAM_RESERVE_KB} KB")
    print(f"Usable SRAM:            {USABLE_SRAM_KB} KB")
    print(f"Max generation tokens:  {MAX_NEW_TOKENS}")
    print(f"Execution Device:       {device}")
    print("=" * 70)

    # Mode 1: Single CLI prompt test
    if args.prompt:
        print(f"\n[CLI Mode] Executing prompt: '{args.prompt}'")
        res = run_constrained_prompt(model, tok, budget, args.prompt, device)
        print_prompt_results(res)
        append_to_csv(res)
        return

    # Mode 2: Interactive loop
    print("Enter natural language commands. Type 'exit' or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("USER > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting sandbox.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting sandbox.")
            break

        res = run_constrained_prompt(model, tok, budget, user_input, device)
        print_prompt_results(res)
        append_to_csv(res)


if __name__ == "__main__":
    main()
