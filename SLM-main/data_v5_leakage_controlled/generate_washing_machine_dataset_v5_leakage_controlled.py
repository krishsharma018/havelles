import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from generate_washing_machine_dataset_v4_grouped import (
    ATOMIC_INTENTS,
    build_semantic_pool,
    canon,
    output_key,
    render,
)


INTENT_WEIGHTS = {
    "START_CYCLE": 0.46,
    "SELECT_PROGRAM": 0.07,
    "SET_TEMPERATURE": 0.07,
    "SET_SPIN_SPEED": 0.07,
    "SET_WATER_LEVEL": 0.05,
    "SET_WASH_DURATION": 0.06,
    "SET_SOAK": 0.045,
    "SET_RINSE": 0.04,
    "SET_DRY": 0.045,
    "ENABLE_OPTION": 0.045,
    "DISABLE_OPTION": 0.045,
    "PAUSE": 0.012,
    "RESUME": 0.012,
    "STOP": 0.012,
    "CANCEL": 0.012,
    "QUERY_STATUS": 0.012,
    "QUERY_REMAINING_TIME": 0.012,
    "QUERY_PROGRAM": 0.012,
}


EXTRA_SUFFIXES = {
    "train": ["", " please", " now", " for this load", " on this cycle", " for today"],
    "validation": ["", " once", " right now", " for these clothes", " this time", " for this wash"],
    "test": ["", " boss", " only", " haan", " for the laundry", " in the machine"],
}


def split_start_semantics(pool, seed):
    rng = random.Random(seed)
    starts = [out for out in pool if out["intent"] == "START_CYCLE"]
    non_start = [out for out in pool if out["intent"] != "START_CYCLE"]
    rng.shuffle(starts)
    n = len(starts)
    train_start = starts[: int(n * 0.78)]
    validation_start = starts[int(n * 0.78) : int(n * 0.89)]
    test_start = starts[int(n * 0.89) :]
    return train_start, validation_start, test_start, non_start


def index_by_intent(items):
    by_intent = defaultdict(list)
    for out in items:
        by_intent[out["intent"]].append(out)
    return dict(by_intent)


def choose_intent(rng):
    intents = list(INTENT_WEIGHTS)
    weights = [INTENT_WEIGHTS[i] for i in intents]
    return rng.choices(intents, weights=weights, k=1)[0]


def render_unique(out, split, rng):
    text = render(out, split, rng)
    text = canon(
        re.sub(
            r"^(Can you|Could you)\s+(how|what|which|where|when|is)\b",
            lambda m: m.group(2).capitalize(),
            text,
            flags=re.IGNORECASE,
        )
    )
    if rng.random() < 0.55:
        text = canon(text + rng.choice(EXTRA_SUFFIXES[split]))
    if rng.random() < 0.35 and text[-1] not in ".?!":
        text += rng.choice([".", "?"])
    return text


def generate_rows(start_scenarios, non_start_by_intent, split, count, seed, global_used_inputs):
    rng = random.Random(seed)
    rows = []
    used_inputs = set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError(f"could not generate enough unique rows for {split}; got {len(rows)}")
        intent = choose_intent(rng)
        if intent == "START_CYCLE":
            out = rng.choice(start_scenarios)
        else:
            bucket = non_start_by_intent.get(intent)
            if not bucket:
                continue
            out = rng.choice(bucket)
        text = render_unique(out, split, rng)
        key = text.casefold()
        if key in used_inputs or key in global_used_inputs:
            continue
        used_inputs.add(key)
        global_used_inputs.add(key)
        rows.append({"input": text, "output": out})
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def report(rows_by_split):
    lines = []
    input_sets = {k: {r["input"].casefold() for r in v} for k, v in rows_by_split.items()}
    output_sets = {k: {output_key(r["output"]) for r in v} for k, v in rows_by_split.items()}
    start_output_sets = {
        k: {output_key(r["output"]) for r in v if r["output"]["intent"] == "START_CYCLE"}
        for k, v in rows_by_split.items()
    }
    slotted_complex_sets = {
        k: {
            output_key(r["output"])
            for r in v
            if r["output"]["intent"] == "START_CYCLE"
            and sum(1 for value in r["output"].values() if value not in (None, "UNSPECIFIED", [], {})) > 2
        }
        for k, v in rows_by_split.items()
    }

    for split, rows in rows_by_split.items():
        lines.append(f"== {split} ==")
        lines.append(f"rows: {len(rows)}")
        lines.append(f"unique_inputs: {len(input_sets[split])}")
        lines.append(f"unique_outputs: {len(output_sets[split])}")
        lines.append("intent_counts:")
        for intent, count in Counter(r["output"]["intent"] for r in rows).most_common():
            lines.append(f"  {intent}: {count}")
        lines.append("")

    names = list(rows_by_split)
    lines.append("== leakage_checks ==")
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lines.append(f"{left} vs {right} exact_input_overlap: {len(input_sets[left] & input_sets[right])}")
            lines.append(f"{left} vs {right} exact_output_overlap_all: {len(output_sets[left] & output_sets[right])}")
            lines.append(f"{left} vs {right} start_cycle_output_overlap: {len(start_output_sets[left] & start_output_sets[right])}")
            lines.append(
                f"{left} vs {right} complex_start_output_overlap: {len(slotted_complex_sets[left] & slotted_complex_sets[right])}"
            )
    lines.append("")
    lines.append("Split policy:")
    lines.append("- Exact input overlap is forbidden across all splits.")
    lines.append("- START_CYCLE semantic scenarios are split before text generation, so their target JSONs do not overlap.")
    lines.append("- Train/validation/test use separate language-template families.")
    lines.append("- Atomic and finite single-slot labels can repeat across splits because they are the actual labels the model must learn.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_v5_leakage_controlled")
    parser.add_argument("--train", type=int, default=24000)
    parser.add_argument("--validation", type=int, default=3000)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_semantic_pool(args.seed, target=12000)
    train_start, validation_start, test_start, non_start = split_start_semantics(pool, args.seed + 1)
    non_start_by_intent = index_by_intent(non_start)

    global_used_inputs = set()
    rows = {
        "train": generate_rows(train_start, non_start_by_intent, "train", args.train, args.seed + 10, global_used_inputs),
        "validation": generate_rows(
            validation_start, non_start_by_intent, "validation", args.validation, args.seed + 20, global_used_inputs
        ),
        "test": generate_rows(test_start, non_start_by_intent, "test", args.test, args.seed + 30, global_used_inputs),
    }
    write_jsonl(out_dir / "washing_machine_synthetic_v5_train_24000.jsonl", rows["train"])
    write_jsonl(out_dir / "washing_machine_synthetic_v5_validation_3000.jsonl", rows["validation"])
    write_jsonl(out_dir / "washing_machine_synthetic_v5_test_3000.jsonl", rows["test"])
    text = report(rows)
    (out_dir / "washing_machine_synthetic_v5_leakage_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
