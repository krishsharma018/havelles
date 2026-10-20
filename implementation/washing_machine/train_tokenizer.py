"""
train_tokenizer.py — Phase 3, Task 2 & 3
=========================================
Train ByteLevel-BPE tokenizer candidates on TRAINING DATA ONLY.

Corpus:
  A. All 37,950 training inputs  (natural language)
  B. All 37,950 minimal JSON targets (serialized JSON)

Do NOT use validation or test text.

Outputs:
  implementation/washing_machine/tokenizer/vocab_<N>/tokenizer.json
  for N in [512, 768, 1024, 1536, 2048, 4096]

Special tokens (reserved ASCII bytes, no added_tokens in tokenizer.json):
  BOS = chr(0x02)  = ASCII STX
  SEP = chr(0x1F)  = ASCII US (Unit Separator)
  EOS = chr(0x03)  = ASCII ETX

These are valid ASCII (< 0x80), guaranteed single tokens (byte table),
and absent from any natural language or JSON text in the dataset.
"""

import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
PREPARED_MINIMAL_DIR = ROOT / "data_v5_leakage_controlled" / "prepared_minimal"
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"

TRAIN_PATH = PREPARED_MINIMAL_DIR / "train.jsonl"

# Candidate vocabulary sizes to evaluate
CANDIDATE_VOCAB_SIZES = [512, 768, 1024, 1536, 2048, 4096]

# Special tokens as reserved single-byte ASCII characters
# These are valid ASCII (< 0x80), absent from NL/JSON, and guaranteed single-token IDs.
BOS_CHAR = "\x02"  # ASCII STX  — Begin of Sequence
SEP_CHAR = "\x1f"  # ASCII US   — Input/Target Separator
EOS_CHAR = "\x03"  # ASCII ETX  — End of Sequence

SPECIAL_CHARS = {
    "BOS": BOS_CHAR,
    "SEP": SEP_CHAR,
    "EOS": EOS_CHAR,
}


def canonical_serialize_target(target):
    """Deterministic compact serialization of minimal target JSON."""
    return json.dumps(target, separators=(',', ':'), ensure_ascii=False)


def load_training_corpus():
    """
    Load all training examples and return corpus lines.
    Source: TRAINING SET ONLY. No validation or test data.
    Corpus includes:
      - All 37,950 natural language inputs
      - All 37,950 serialized minimal JSON targets
    """
    print(f"Loading training corpus from {TRAIN_PATH}")
    corpus_lines = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            inp = item["input"]
            tgt = canonical_serialize_target(item["target"])
            corpus_lines.append(inp)
            corpus_lines.append(tgt)

    print(f"  {len(corpus_lines):,} corpus lines loaded")
    print(f"  ({len(corpus_lines) // 2:,} input + {len(corpus_lines) // 2:,} target lines)")
    total_chars = sum(len(l) for l in corpus_lines)
    print(f"  Total corpus size: {total_chars:,} characters")
    return corpus_lines


def train_tokenizer_candidate(corpus_lines, vocab_size):
    """
    Train a ByteLevel BPE tokenizer compatible with the existing ESP32 runtime.
    Configuration must match generate_tokenizer_header.py requirements exactly:
      - model.type = "BPE"
      - pre_tokenizer.type = "ByteLevel", add_prefix_space=False, use_regex=True
      - NO added_tokens
      - NO truncation, padding, normalizer, post_processor
    """
    out_dir = TOKENIZER_DIR / f"vocab_{vocab_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tokenizer.json"

    if out_path.exists():
        print(f"  vocab={vocab_size}: already trained at {out_path}, loading.")
        return Tokenizer.from_file(str(out_path))

    print(f"  Training BPE tokenizer with vocab_size={vocab_size}...")

    # This exact configuration matches what generate_tokenizer_header.py requires
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # No special_tokens here — we use reserved ASCII bytes instead.
        # Adding special_tokens here creates added_tokens which BREAKS BTK1 packing.
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train_from_iterator(corpus_lines, trainer=trainer)

    # Save tokenizer.json (HuggingFace format, compatible with packer)
    tok.save(str(out_path))
    actual_vocab_size = tok.get_vocab_size()
    print(f"  Saved tokenizer.json: vocab_size={actual_vocab_size}")

    return tok


def verify_special_token_single_ids(tok):
    """
    Verify that BOS, SEP, EOS are single-token IDs in this tokenizer.
    They must be since they're single bytes mapped through the 256-byte table.
    """
    results = {}
    for name, char in SPECIAL_CHARS.items():
        ids = tok.encode(char, add_special_tokens=False).ids
        if len(ids) != 1:
            print(f"  WARNING: {name} ({repr(char)}) tokenized to {len(ids)} tokens: {ids}")
        results[name] = {
            "char": repr(char),
            "token_id": ids[0] if len(ids) == 1 else ids,
            "is_single_token": len(ids) == 1,
        }
    return results


def save_tokenizer_metadata(out_dir, tok, special_token_ids, vocab_size):
    """Save metadata.json describing the tokenizer configuration."""
    import hashlib

    tokenizer_path = out_dir / "tokenizer.json"
    sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    meta = {
        "algorithm": "ByteLevel-BPE",
        "library": "huggingface/tokenizers",
        "pre_tokenizer": "ByteLevel(add_prefix_space=False, use_regex=True)",
        "decoder": "ByteLevel",
        "added_tokens": [],
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tok.get_vocab_size(),
        "esp32_runtime_compatible": True,
        "btk1_format": "version_2",
        "special_tokens": special_token_ids,
        "special_token_encoding": "reserved_ascii_bytes_no_added_tokens",
        "tokenizer_sha256": sha256,
        "notes": [
            "No added_tokens — compatible with BTK1 packer (generate_tokenizer_header.py)",
            "Special tokens are reserved ASCII bytes: BOS=\\x02, SEP=\\x1f, EOS=\\x03",
            "These bytes are guaranteed single-token IDs via the 256-byte BTK1 table",
            "Non-ASCII input rejected by runtime (bpe_encode_ascii)",
            "Corpus: 37,950 NL inputs + 37,950 minimal JSON targets (TRAIN ONLY)",
        ],
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {meta_path}")
    return meta


def main():
    print("=" * 60)
    print("PHASE 3 — TOKENIZER TRAINING")
    print("=" * 60)

    if not TRAIN_PATH.exists():
        raise SystemExit(f"Training file not found: {TRAIN_PATH}")

    corpus_lines = load_training_corpus()
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for vocab_size in CANDIDATE_VOCAB_SIZES:
        print(f"\n--- vocab_size={vocab_size} ---")
        tok = train_tokenizer_candidate(corpus_lines, vocab_size)
        out_dir = TOKENIZER_DIR / f"vocab_{vocab_size}"

        special_ids = verify_special_token_single_ids(tok)
        for name, info in special_ids.items():
            print(f"  {name}: char={info['char']}, token_id={info['token_id']}, single={info['is_single_token']}")

        meta = save_tokenizer_metadata(out_dir, tok, special_ids, vocab_size)
        all_metadata[vocab_size] = meta

    # Save a summary of all candidate metadata
    summary_path = TOKENIZER_DIR / "candidates_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"\nAll candidates summary saved: {summary_path}")
    print("\nDONE: Tokenizer training complete.")


if __name__ == "__main__":
    main()
