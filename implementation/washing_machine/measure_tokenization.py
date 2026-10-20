"""
measure_tokenization.py — Phase 3, Task 4 & 5
===============================================
For every candidate vocabulary size, compute:
  - INPUT token lengths (min, median, mean, p95, p99, max)
  - TARGET token lengths
  - COMBINED sequence lengths: [BOS] input [SEP] target [EOS]
  - Percentage of examples exceeding given thresholds
  - JSON field & value tokenization efficiency

Results saved to:
  implementation/washing_machine/tokenizer/tokenization_stats.json
"""

import json
import math
import os
import re
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
PREPARED_MINIMAL_DIR = ROOT / "data_v5_leakage_controlled" / "prepared_minimal"
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"

CANDIDATE_VOCAB_SIZES = [512, 768, 1024, 1374]  # 1536/2048/4096 all saturate at 1374

SPLITS = {
    "train": PREPARED_MINIMAL_DIR / "train.jsonl",
    "validation": PREPARED_MINIMAL_DIR / "validation.jsonl",
    "test": PREPARED_MINIMAL_DIR / "test.jsonl",
}

THRESHOLDS = [32, 48, 64, 80, 96, 128, 160, 192, 256]

BOS_CHAR = "\x02"
SEP_CHAR = "\x1f"
EOS_CHAR = "\x03"

SCHEMA_KEYS = [
    "intent", "program", "load_type", "soil_level", "stain_type",
    "temperature", "temperature_constraint", "spin_speed", "spin_speed_constraint",
    "water_level", "wash_duration", "soak_duration", "rinse_count", "dry_duration",
    "options", "disabled_options", "constraints",
]

COMMON_VALUES = [
    "START_CYCLE", "SELECT_PROGRAM", "SET_TEMPERATURE", "SET_SPIN_SPEED",
    "PAUSE", "RESUME", "STOP", "CANCEL", "QUERY_STATUS", "QUERY_PROGRAM",
    "QUERY_REMAINING_TIME", "DISABLE_OPTION", "ENABLE_OPTION",
    "SET_WASH_DURATION", "SET_WATER_LEVEL", "SET_SOAK", "SET_DRY", "SET_RINSE",
    "COTTON", "DELICATE", "WOOL", "QUICK", "NORMAL", "JEANS", "MIXED",
    "SYNTHETIC", "HEAVY", "GENTLE", "DRAIN", "RINSE", "SPIN", "DRUM_CLEAN", "DRY",
    "BABY_CLOTHES", "LIGHT", "SWEAT", "FOOD", "DIRT", "OIL",
    "COLD", "WARM", "NOT_HOT", "NOT_COLD",
    "LOW", "MEDIUM", "HIGH", "AUTO",
    "NOT_HIGH", "NOT_LOW",
    "STAIN_REMOVAL", "EXTRA_RINSE", "PREWASH", "STEAM", "CHILD_LOCK", "SOAK",
    "null", "true", "false",
]

JSON_SYNTAX = ["{", "}", "[", "]", ":", ",", '"', "null"]


def canonical_serialize(target):
    return json.dumps(target, separators=(',', ':'), ensure_ascii=False)


def compute_stats(numbers):
    if not numbers:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "p99": 0, "max": 0}
    s = sorted(numbers)
    n = len(s)

    def pct(p):
        k = (n - 1) * p / 100.0
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[int(f)] * (c - k) + s[int(c)] * (k - f)

    return {
        "min": int(s[0]),
        "median": round(float(pct(50)), 2),
        "mean": round(float(sum(s) / n), 2),
        "p95": round(float(pct(95)), 2),
        "p99": round(float(pct(99)), 2),
        "max": int(s[-1]),
    }


def pct_exceeding(numbers, threshold):
    if not numbers:
        return 0.0
    return round(100.0 * sum(1 for x in numbers if x > threshold) / len(numbers), 2)


def tokenize_combined(tok, inp_text, tgt_text):
    """Tokenize [BOS] input [SEP] target [EOS] and return all token id lists."""
    bos_ids = tok.encode(BOS_CHAR, add_special_tokens=False).ids
    sep_ids = tok.encode(SEP_CHAR, add_special_tokens=False).ids
    eos_ids = tok.encode(EOS_CHAR, add_special_tokens=False).ids
    inp_ids = tok.encode(inp_text, add_special_tokens=False).ids
    tgt_ids = tok.encode(tgt_text, add_special_tokens=False).ids
    combined = bos_ids + inp_ids + sep_ids + tgt_ids + eos_ids
    return inp_ids, tgt_ids, combined


def measure_json_token_efficiency(tok):
    """Measure how many tokens each JSON syntax char, key, and value uses."""
    results = {}

    for char in JSON_SYNTAX:
        ids = tok.encode(char, add_special_tokens=False).ids
        results[f"syntax_{char}"] = {"text": char, "tokens": ids, "count": len(ids)}

    for key in SCHEMA_KEYS:
        # Keys appear inside JSON as: "key":
        # Test both standalone and in JSON context
        ids_standalone = tok.encode(key, add_special_tokens=False).ids
        ids_quoted = tok.encode(f'"{key}":', add_special_tokens=False).ids
        results[f"key_{key}"] = {
            "text": key,
            "standalone_tokens": len(ids_standalone),
            "quoted_colon_tokens": len(ids_quoted),
            "quoted_colon_ids": ids_quoted,
        }

    for val in COMMON_VALUES:
        ids = tok.encode(val, add_special_tokens=False).ids
        results[f"val_{val}"] = {"text": val, "tokens": ids, "count": len(ids)}

    return results


def load_split_data(split_path):
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def measure_vocab(vocab_size):
    """Measure all stats for one vocabulary candidate."""
    tok_path = TOKENIZER_DIR / f"vocab_{vocab_size}" / "tokenizer.json"
    if not tok_path.exists():
        print(f"  SKIP vocab_{vocab_size}: tokenizer.json not found")
        return None

    tok = Tokenizer.from_file(str(tok_path))
    actual_vocab = tok.get_vocab_size()
    print(f"  vocab_size={vocab_size} (actual {actual_vocab})")

    result = {
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": actual_vocab,
        "splits": {},
        "json_token_efficiency": measure_json_token_efficiency(tok),
    }

    for split_name, split_path in SPLITS.items():
        print(f"    Measuring {split_name}...")
        records = load_split_data(split_path)

        inp_lens, tgt_lens, combined_lens = [], [], []

        for rec in records:
            inp_text = rec["input"]
            tgt_text = canonical_serialize(rec["target"])
            inp_ids, tgt_ids, combined = tokenize_combined(tok, inp_text, tgt_text)
            inp_lens.append(len(inp_ids))
            tgt_lens.append(len(tgt_ids))
            combined_lens.append(len(combined))

        split_result = {
            "total_examples": len(records),
            "input_token_lengths": compute_stats(inp_lens),
            "target_token_lengths": compute_stats(tgt_lens),
            "combined_token_lengths": compute_stats(combined_lens),
            "percentage_exceeding_thresholds": {
                str(t): {
                    "combined": pct_exceeding(combined_lens, t),
                    "input": pct_exceeding(inp_lens, t),
                    "target": pct_exceeding(tgt_lens, t),
                }
                for t in THRESHOLDS
            },
        }
        result["splits"][split_name] = split_result

    return result


def main():
    print("=" * 60)
    print("PHASE 3 — TOKENIZATION MEASUREMENT")
    print("=" * 60)

    all_results = {}

    for vocab_size in CANDIDATE_VOCAB_SIZES:
        print(f"\n--- Measuring vocab_size={vocab_size} ---")
        res = measure_vocab(vocab_size)
        if res:
            all_results[str(vocab_size)] = res

    stats_path = TOKENIZER_DIR / "tokenization_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved tokenization stats: {stats_path}")

    # Print a concise summary table to stdout
    print("\n=== SUMMARY: Combined Sequence Token Lengths (train) ===")
    header = f"{'Vocab':>8} | {'Min':>5} | {'Med':>6} | {'Mean':>6} | {'P95':>6} | {'P99':>6} | {'Max':>5} | {'>64':>6} | {'>80':>6} | {'>96':>6} | {'>128':>6}"
    print(header)
    print("-" * len(header))
    for v_str, res in all_results.items():
        stats = res["splits"]["train"]["combined_token_lengths"]
        exc = res["splits"]["train"]["percentage_exceeding_thresholds"]
        row = (
            f"{v_str:>8} | "
            f"{stats['min']:>5} | "
            f"{stats['median']:>6} | "
            f"{stats['mean']:>6} | "
            f"{stats['p95']:>6} | "
            f"{stats['p99']:>6} | "
            f"{stats['max']:>5} | "
            f"{exc['64']['combined']:>5}% | "
            f"{exc['80']['combined']:>5}% | "
            f"{exc['96']['combined']:>5}% | "
            f"{exc['128']['combined']:>5}%"
        )
        print(row)

    print("\nDONE: Tokenization measurement complete.")
    return all_results


if __name__ == "__main__":
    main()
