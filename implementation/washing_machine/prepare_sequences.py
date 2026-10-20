"""
prepare_sequences.py — Phase 3, Task 7 & 8
===========================================
Tokenize and prepare training sequences for the washing machine SLM.

Reads:
  data_v5_leakage_controlled/prepared_minimal/train.jsonl
  data_v5_leakage_controlled/prepared_minimal/validation.jsonl
  data_v5_leakage_controlled/prepared_minimal/test.jsonl

Each record produces:
  - example_id:        unique integer ID
  - input_text:        original NL input string
  - target_text:       canonical serialized minimal JSON
  - input_token_ids:   list[int]
  - target_token_ids:  list[int]
  - combined_sequence_ids: list[int]  [BOS] input [SEP] target [EOS]
  - loss_mask:         list[int] 0/1 per combined_sequence token
                       0 = BOS + input + SEP tokens (no loss)
                       1 = target + EOS tokens (compute loss here)
  - seq_len:           int, total combined sequence length
  - split:             "train" / "validation" / "test"
  - input_len:         number of input tokens (excluding BOS)
  - target_len:        number of target tokens (excluding SEP, excluding EOS)

The loss mask ensures:
  [BOS][...input tokens...][SEP][...target tokens...][EOS]
    0         0...0           0      1...1              1

Does NOT modify original dataset files.

Usage:
  python implementation/washing_machine/prepare_sequences.py [--vocab 1024]

Output:
  implementation/washing_machine/sequences/
    train.jsonl
    validation.jsonl
    test.jsonl
    sequence_stats.json
"""

import argparse
import json
import math
import os
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
PREPARED_MINIMAL_DIR = ROOT / "data_v5_leakage_controlled" / "prepared_minimal"
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"
SEQUENCES_DIR = Path(__file__).resolve().parent / "sequences"

BOS_CHAR = "\x02"  # ASCII STX
SEP_CHAR = "\x1f"  # ASCII US
EOS_CHAR = "\x03"  # ASCII ETX

SPLITS = {
    "train": PREPARED_MINIMAL_DIR / "train.jsonl",
    "validation": PREPARED_MINIMAL_DIR / "validation.jsonl",
    "test": PREPARED_MINIMAL_DIR / "test.jsonl",
}


def canonical_serialize(target):
    """Deterministic compact JSON serialization of minimal target."""
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


def prepare_split(tok, split_name, split_path, out_path, global_id_offset):
    """
    Tokenize all examples in one split.
    Returns (records, seq_lengths, any_errors).
    """
    bos_ids = tok.encode(BOS_CHAR, add_special_tokens=False).ids
    sep_ids = tok.encode(SEP_CHAR, add_special_tokens=False).ids
    eos_ids = tok.encode(EOS_CHAR, add_special_tokens=False).ids

    assert len(bos_ids) == 1, f"BOS must be single token, got {bos_ids}"
    assert len(sep_ids) == 1, f"SEP must be single token, got {sep_ids}"
    assert len(eos_ids) == 1, f"EOS must be single token, got {eos_ids}"

    vocab_size = tok.get_vocab_size()
    seq_lengths = []
    errors = []
    records_written = 0
    example_id = global_id_offset

    with open(split_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue

            item = json.loads(line)
            inp_text = item["input"]
            tgt_obj = item["target"]
            gt_obj = item.get("ground_truth", {})
            tgt_text = canonical_serialize(tgt_obj)

            # Tokenize each component
            inp_ids = tok.encode(inp_text, add_special_tokens=False).ids
            tgt_ids = tok.encode(tgt_text, add_special_tokens=False).ids

            combined = bos_ids + inp_ids + sep_ids + tgt_ids + eos_ids

            # Build loss mask:
            # 0 = no loss: BOS token + all input tokens + SEP token
            # 1 = compute loss: all target tokens + EOS token
            n_prefix = len(bos_ids) + len(inp_ids) + len(sep_ids)  # 1 + inp + 1
            n_target = len(tgt_ids) + len(eos_ids)                 # tgt + 1
            loss_mask = [0] * n_prefix + [1] * n_target

            assert len(combined) == len(loss_mask), \
                f"Combined len {len(combined)} != mask len {len(loss_mask)}"
            assert len(combined) == n_prefix + n_target

            # Safety checks
            out_of_range = [tok_id for tok_id in combined if tok_id >= vocab_size]
            if out_of_range:
                errors.append({
                    "example_id": example_id,
                    "split": split_name,
                    "line_idx": line_idx,
                    "error": f"Token IDs out of range: {out_of_range[:5]}",
                })

            # Verify round-trip for target text
            decoded_tgt = tok.decode(tgt_ids)
            if decoded_tgt != tgt_text:
                errors.append({
                    "example_id": example_id,
                    "split": split_name,
                    "line_idx": line_idx,
                    "error": "Round-trip decode mismatch for target",
                    "original": tgt_text,
                    "decoded": decoded_tgt,
                })

            seq_lengths.append(len(combined))

            record = {
                "example_id": example_id,
                "split": split_name,
                "input_text": inp_text,
                "target_text": tgt_text,
                "input_token_ids": inp_ids,
                "target_token_ids": tgt_ids,
                "combined_sequence_ids": combined,
                "loss_mask": loss_mask,
                "seq_len": len(combined),
                "input_len": len(inp_ids),
                "target_len": len(tgt_ids),
                # Retain reference to structured targets for traceability
                "target_json": tgt_obj,
                "ground_truth_json": gt_obj,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            example_id += 1
            records_written += 1

    return records_written, seq_lengths, errors, example_id


def run(vocab_size=1024):
    tok_path = TOKENIZER_DIR / f"vocab_{vocab_size}" / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"Tokenizer not found: {tok_path}")

    tok = Tokenizer.from_file(str(tok_path))
    actual_vocab = tok.get_vocab_size()
    print(f"Loaded tokenizer: vocab_size={actual_vocab} from {tok_path}")

    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

    # Verify special tokens are single-token IDs
    for name, char in [("BOS", BOS_CHAR), ("SEP", SEP_CHAR), ("EOS", EOS_CHAR)]:
        ids = tok.encode(char, add_special_tokens=False).ids
        assert len(ids) == 1, f"{name} ({repr(char)}) must be single token, got {ids}"
        print(f"  {name}: char={repr(char)}, token_id={ids[0]}")

    all_stats = {}
    all_errors = []
    global_id_offset = 0

    for split_name, split_path in SPLITS.items():
        out_path = SEQUENCES_DIR / f"{split_name}.jsonl"
        print(f"\nProcessing split: {split_name}")
        count, seq_lens, errors, global_id_offset = prepare_split(
            tok, split_name, split_path, out_path, global_id_offset
        )
        print(f"  Written: {count:,} examples -> {out_path}")
        print(f"  Errors:  {len(errors)}")

        stats = compute_stats(seq_lens)
        all_stats[split_name] = {
            "total_examples": count,
            "sequence_length_stats": stats,
            "coverage_at_context_lengths": {
                ctx: round(100.0 * sum(1 for l in seq_lens if l <= ctx) / len(seq_lens), 2)
                for ctx in [64, 80, 96, 128, 160, 192, 256]
            },
            "percentage_exceeding": {
                ctx: round(100.0 * sum(1 for l in seq_lens if l > ctx) / len(seq_lens), 2)
                for ctx in [64, 80, 96, 128, 160]
            },
        }
        all_errors.extend(errors)

    # Context length recommendation
    train_stats = all_stats["train"]["sequence_length_stats"]
    print(f"\n=== CONTEXT LENGTH ANALYSIS ===")
    for ctx in [64, 80, 96, 128, 160, 192]:
        cov = all_stats["train"]["coverage_at_context_lengths"].get(ctx, 0)
        print(f"  ctx={ctx:4d}: train coverage = {cov:.2f}%")

    # Save sequence stats
    stats_record = {
        "vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab,
        "tokenizer_path": str(tok_path),
        "special_token_ids": {
            "BOS": tok.encode(BOS_CHAR, add_special_tokens=False).ids[0],
            "SEP": tok.encode(SEP_CHAR, add_special_tokens=False).ids[0],
            "EOS": tok.encode(EOS_CHAR, add_special_tokens=False).ids[0],
        },
        "loss_mask_spec": {
            "0_no_loss": "BOS token + input tokens + SEP token",
            "1_compute_loss": "target tokens + EOS token",
        },
        "splits": all_stats,
        "total_errors": len(all_errors),
        "errors": all_errors[:20],  # Store first 20 for review
    }

    stats_path = SEQUENCES_DIR / "sequence_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_record, f, indent=2, ensure_ascii=False)
    print(f"\nSaved sequence stats: {stats_path}")

    print(f"\nTotal errors across all splits: {len(all_errors)}")
    if all_errors:
        print("ERROR SUMMARY (first 5):")
        for e in all_errors[:5]:
            print(f"  {e}")

    print("\nDONE: Sequence preparation complete.")
    return stats_record


def main():
    ap = argparse.ArgumentParser(description="Prepare tokenized sequences for washing machine SLM Phase 3.")
    ap.add_argument("--vocab", type=int, default=1024,
                    help="Vocabulary size (must be a trained candidate, default=1024)")
    args = ap.parse_args()
    run(vocab_size=args.vocab)


if __name__ == "__main__":
    main()
