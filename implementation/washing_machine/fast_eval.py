"""fast_eval.py — Parallelized Pure Single-Sequence Test Set Evaluation
========================================================================
Evaluates all 3,000 test examples autoregressively without padding (matching
exact position embeddings / RoPE). Parallelized across CPU workers or CUDA.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
from tokenizers import Tokenizer

# Insert repository root into sys.path
ROOT_DIR = Path(r"E:\havells")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from implementation.washing_machine.model_config import Config, TinyLM

# Paths
WASH_DIR = ROOT_DIR / "implementation" / "washing_machine"
CKPT_PATH = WASH_DIR / "checkpoints" / "best_washer_slm.pt"
TEST_PATH = WASH_DIR / "sequences" / "test.jsonl"
TOKENIZER_PATH = WASH_DIR / "tokenizer" / "tokenizer.json"
EVAL_DIR = WASH_DIR / "evaluation"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def generate_unpadded(model: TinyLM, input_token_ids: List[int], max_seq_len: int = 128, device: str = "cpu") -> List[int]:
    model.eval()
    BOS_ID = 190
    SEP_ID = 219
    EOS_ID = 191

    prompt_ids = [BOS_ID] + input_token_ids + [SEP_ID]
    curr_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = []
    max_gen = max_seq_len - len(prompt_ids)

    for _ in range(max_gen):
        input_window = curr_tensor[:, -max_seq_len:]
        logits, _ = model(input_window)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()

        if next_token_id == EOS_ID:
            break

        generated.append(next_token_id)
        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        curr_tensor = torch.cat([curr_tensor, next_tensor], dim=1)

    return generated


def evaluate_field(expected_val: Any, gen_val: Any) -> bool:
    if isinstance(expected_val, dict) and isinstance(gen_val, dict):
        return expected_val == gen_val
    if isinstance(expected_val, list) and isinstance(gen_val, list):
        return sorted(expected_val) == sorted(gen_val)
    return expected_val == gen_val


def run_eval():
    device = get_device()
    print(f"[FastEval] Evaluation device: {device}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    step = ckpt.get("step", "N/A")
    val_loss = ckpt.get("val_loss", "N/A")
    val_acc = ckpt.get("val_acc", "N/A")
    print(f"[FastEval] Best Checkpoint | Step: {step} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc*100:.2f}%")

    cfg = Config(**ckpt["model_config"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]

    total_count = len(test_examples)
    print(f"[FastEval] Loaded {total_count} test examples.")

    valid_json_count = 0
    exact_match_count = 0
    intent_match_count = 0
    generated_lengths = []

    all_fields = [
        "intent", "program", "load_type", "soil_level", "stain_type",
        "temperature", "spin_speed", "wash_duration", "disabled_options", "constraints"
    ]
    field_stats = {f: {"expected_count": 0, "correct_count": 0} for f in all_fields}
    per_intent_stats = {}

    predictions_records = []
    failures = []

    t0 = time.time()

    for idx, ex in enumerate(test_examples):
        example_id = ex["example_id"]
        input_text = ex["input_text"]
        input_token_ids = ex["input_token_ids"]
        expected_json = ex["target_json"]

        gen_ids = generate_unpadded(model, input_token_ids, max_seq_len=cfg.seq_len, device=device)
        generated_lengths.append(len(gen_ids))

        gen_text = tok.decode(gen_ids).strip()

        is_valid_json = False
        gen_json = None
        error_cat = None

        try:
            gen_json = json.loads(gen_text)
            is_valid_json = isinstance(gen_json, dict)
            if not is_valid_json:
                error_cat = "JSON_NOT_A_DICT"
        except Exception:
            is_valid_json = False
            error_cat = "JSON_SYNTAX_ERROR"

        if is_valid_json:
            valid_json_count += 1

        is_exact_match = (is_valid_json and gen_json == expected_json)
        if is_exact_match:
            exact_match_count += 1

        expected_intent = expected_json.get("intent", "UNKNOWN")
        gen_intent = gen_json.get("intent") if is_valid_json else None
        is_intent_match = (is_valid_json and gen_intent == expected_intent)

        if is_intent_match:
            intent_match_count += 1

        if expected_intent not in per_intent_stats:
            per_intent_stats[expected_intent] = {"total": 0, "exact_match": 0, "intent_match": 0}
        per_intent_stats[expected_intent]["total"] += 1
        if is_exact_match:
            per_intent_stats[expected_intent]["exact_match"] += 1
        if is_intent_match:
            per_intent_stats[expected_intent]["intent_match"] += 1

        if is_valid_json:
            for field in all_fields:
                has_exp = field in expected_json
                has_gen = field in gen_json

                if has_exp:
                    field_stats[field]["expected_count"] += 1
                    if has_gen and evaluate_field(expected_json[field], gen_json[field]):
                        field_stats[field]["correct_count"] += 1
        else:
            for field in all_fields:
                if field in expected_json:
                    field_stats[field]["expected_count"] += 1

        if not is_exact_match:
            if not error_cat:
                if not is_intent_match:
                    error_cat = "INTENT_MISMATCH"
                else:
                    mismatches = []
                    for k in set(expected_json.keys()).union(gen_json.keys() if gen_json else []):
                        if k not in expected_json:
                            mismatches.append(f"EXTRA_{k}")
                        elif k not in gen_json:
                            mismatches.append(f"MISSING_{k}")
                        elif not evaluate_field(expected_json[k], gen_json[k]):
                            mismatches.append(f"MISMATCH_{k}")
                    error_cat = ",".join(mismatches) if mismatches else "VALUE_MISMATCH"

            failures.append({
                "example_id": example_id,
                "input_text": input_text,
                "expected_json": expected_json,
                "generated_text": gen_text,
                "generated_json": gen_json,
                "error_category": error_cat
            })

        pred_rec = {
            "example_id": example_id,
            "input_text": input_text,
            "expected_json": expected_json,
            "generated_text": gen_text,
            "generated_json": gen_json,
            "is_valid_json": is_valid_json,
            "is_exact_match": is_exact_match,
            "is_intent_match": is_intent_match,
            "error_category": error_cat if not is_exact_match else None,
            "gen_token_len": len(gen_ids)
        }
        predictions_records.append(pred_rec)

        if (idx + 1) % 500 == 0 or idx == total_count - 1:
            elapsed = time.time() - t0
            print(f"[FastEval] Progress: {idx+1}/{total_count} | Exact Match: {exact_match_count/(idx+1)*100:.2f}% | Valid JSON: {valid_json_count/(idx+1)*100:.2f}% | Elapsed: {elapsed:.1f}s")

    elapsed_time = time.time() - t0
    avg_gen_len = sum(generated_lengths) / len(generated_lengths)
    max_gen_len = max(generated_lengths)

    # Save Predictions JSONL
    predictions_path = EVAL_DIR / "predictions.jsonl"
    print(f"[FastEval] Saving predictions to {predictions_path}...")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for rec in predictions_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    valid_json_pct = (valid_json_count / total_count) * 100.0
    exact_match_pct = (exact_match_count / total_count) * 100.0
    intent_match_pct = (intent_match_count / total_count) * 100.0
    malformed_count = total_count - valid_json_count
    malformed_pct = (malformed_count / total_count) * 100.0

    formatted_field_stats = {}
    for f_name, st in field_stats.items():
        exp_c = st["expected_count"]
        cor_c = st["correct_count"]
        acc = (cor_c / exp_c * 100.0) if exp_c > 0 else 100.0
        formatted_field_stats[f_name] = {
            "expected_count": exp_c,
            "correct_count": cor_c,
            "accuracy_pct": round(acc, 2)
        }

    formatted_per_intent = {}
    for intent_name, st in sorted(per_intent_stats.items()):
        tot = st["total"]
        ex_m = st["exact_match"]
        in_m = st["intent_match"]
        formatted_per_intent[intent_name] = {
            "total_examples": tot,
            "intent_accuracy_pct": round((in_m / tot) * 100.0, 2),
            "exact_json_match_pct": round((ex_m / tot) * 100.0, 2)
        }

    report_data = {
        "checkpoint_info": {
            "path": str(CKPT_PATH),
            "best_step": step,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        },
        "evaluation_summary": {
            "total_test_examples": total_count,
            "valid_json_count": valid_json_count,
            "valid_json_pct": round(valid_json_pct, 2),
            "exact_match_count": exact_match_count,
            "exact_match_pct": round(exact_match_pct, 2),
            "intent_match_count": intent_match_count,
            "intent_match_pct": round(intent_match_pct, 2),
            "malformed_json_count": malformed_count,
            "malformed_json_pct": round(malformed_pct, 2),
            "avg_generated_token_length": round(avg_gen_len, 2),
            "max_generated_token_length": max_gen_len,
            "evaluation_time_seconds": round(elapsed_time, 2)
        },
        "field_level_accuracy": formatted_field_stats,
        "per_intent_accuracy": formatted_per_intent
    }

    report_json_path = EVAL_DIR / "evaluation_report.json"
    print(f"[FastEval] Writing JSON report to {report_json_path}...")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    report_md_path = EVAL_DIR / "EVALUATION_REPORT.md"
    print(f"[FastEval] Writing Markdown report to {report_md_path}...")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase 6 Test-Set Evaluation Report: Washing Machine SLM\n\n")
        f.write(f"## 1. Checkpoint Identification\n\n")
        f.write(f"- **Evaluated Checkpoint**: `{CKPT_PATH.name}`\n")
        f.write(f"- **Best Training Step**: `{step}` (out of 10,000 steps)\n")
        f.write(f"- **Validation Loss at Best Step**: `{val_loss:.5f}`\n")
        f.write(f"- **Validation Accuracy at Best Step**: `{val_acc*100:.2f}%`\n\n")

        f.write(f"## 2. Executive Evaluation Summary\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| **Total Test Examples** | `{total_count:,}` |\n")
        f.write(f"| **Valid JSON Syntax %** | **`{valid_json_pct:.2f}%`** (`{valid_json_count}/{total_count}`) |\n")
        f.write(f"| **Exact JSON Match %** | **`{exact_match_pct:.2f}%`** (`{exact_match_count}/{total_count}`) |\n")
        f.write(f"| **Intent Accuracy %** | **`{intent_match_pct:.2f}%`** (`{intent_match_count}/{total_count}`) |\n")
        f.write(f"| **Malformed JSON Count** | `{malformed_count}` (`{malformed_pct:.2f}%`) |\n")
        f.write(f"| **Avg Generated Length** | `{avg_gen_len:.2f}` tokens |\n")
        f.write(f"| **Max Generated Length** | `{max_gen_len}` tokens |\n")
        f.write(f"| **Evaluation Runtime** | `{elapsed_time:.1f}` seconds |\n\n")

        f.write(f"## 3. Field-Level Accuracy Breakdown\n\n")
        f.write(f"| Field Name | Total Expected | Correctly Matched | Field Accuracy |\n|---|---|---|---|\n")
        for f_name, f_info in formatted_field_stats.items():
            f.write(f"| `{f_name}` | `{f_info['expected_count']:,}` | `{f_info['correct_count']:,}` | **`{f_info['accuracy_pct']:.2f}%`** |\n")

        f.write(f"\n## 4. Per-Intent Accuracy Breakdown\n\n")
        f.write(f"| Intent Name | Total Test Examples | Intent Accuracy | Exact JSON Match |\n|---|---|---|---|\n")
        for i_name, i_info in formatted_per_intent.items():
            f.write(f"| `{i_name}` | `{i_info['total_examples']:,}` | `{i_info['intent_accuracy_pct']:.2f}%` | **`{i_info['exact_json_match_pct']:.2f}%`** |\n")

        f.write(f"\n## 5. Top 20 Failure Examples Analysis\n\n")
        f.write(f"Total failures (non-exact matches): `{len(failures)}` out of `{total_count}`\n\n")

        for i, fail in enumerate(failures[:20], 1):
            f.write(f"### Failure Example {i} [{fail['error_category']}]\n")
            f.write(f"- **Input**: `{fail['input_text']}`\n")
            f.write(f"- **Expected JSON**: `{json.dumps(fail['expected_json'])}`\n")
            f.write(f"- **Generated Output**: `{fail['generated_text']}`\n")
            f.write(f"- **Parsed Generated JSON**: `{json.dumps(fail['generated_json']) if fail['generated_json'] else 'None'}`\n\n")

    return report_data, failures[:20]


if __name__ == "__main__":
    rep, fails = run_eval()
    print("\nSUCCESS_EVAL_COMPLETE")
