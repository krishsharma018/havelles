r"""infer.py — Interactive Inference for Washing Machine SLM
============================================================
Runs interactive natural language -> minimal JSON inference using the trained
washing machine model checkpoint.

Usage:
  python E:\havells\implementation\washing_machine\infer.py
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from tokenizers import Tokenizer

# Ensure repository root is in sys.path
ROOT_DIR = Path(r"E:\havells")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from implementation.washing_machine.model_config import Config, TinyLM

# Paths
WASH_DIR = ROOT_DIR / "implementation" / "washing_machine"
CKPT_PATH = WASH_DIR / "checkpoints" / "best_washer_slm.pt"
TOK_PATH = WASH_DIR / "tokenizer" / "tokenizer.json"

# Special Tokens
BOS_ID = 190
SEP_ID = 219
EOS_ID = 191
MAX_NEW_TOKENS = 80


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def generate_response(
    model: TinyLM,
    tok: Tokenizer,
    user_input: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Tuple[str, List[int], bool, Optional[dict]]:
    """Encodes user input, performs greedy autoregressive generation, and decodes response."""
    model.eval()

    # Encode input prompt
    input_tokens = tok.encode(user_input, add_special_tokens=False).ids
    prompt_ids = [BOS_ID] + input_tokens + [SEP_ID]
    curr_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    max_len = model.cfg.seq_len
    allowed_gen = min(max_new_tokens, max_len - len(prompt_ids))

    gen_ids = []

    for _ in range(max(1, allowed_gen)):
        input_window = curr_tensor[:, -max_len:]
        logits, _ = model(input_window)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()

        if next_token_id == EOS_ID:
            break

        gen_ids.append(next_token_id)
        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        curr_tensor = torch.cat([curr_tensor, next_tensor], dim=1)

    # Decode generated response
    gen_text = tok.decode(gen_ids).strip()

    # Validate JSON
    is_valid_json = False
    parsed_json = None
    try:
        parsed_json = json.loads(gen_text)
        is_valid_json = isinstance(parsed_json, dict)
    except Exception:
        is_valid_json = False

    return gen_text, gen_ids, is_valid_json, parsed_json


def main():
    device = get_device()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")
    if not TOK_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {TOK_PATH}")

    # Load Checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # Verify Tokenizer SHA256 if available
    tok_bytes = TOK_PATH.read_bytes()
    tok_sha = hashlib.sha256(tok_bytes).hexdigest()
    ckpt_tok_sha = ckpt.get("tokenizer_sha256")

    if ckpt_tok_sha and tok_sha != ckpt_tok_sha:
        print(f"[Warning] Tokenizer SHA256 mismatch! Local: {tok_sha[:16]}... vs Checkpoint: {ckpt_tok_sha[:16]}...")

    # Load Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))

    # Reconstruct TinyLM Model
    cfg = Config(**ckpt["model_config"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    budget = model.param_budget()
    total_params = budget["total"]

    # Print Startup Summary
    print("=" * 65)
    print("=== Washing Machine SLM Interactive Inference ===")
    print("=" * 65)
    print(f"  Device:                 {device}")
    print(f"  Checkpoint:             {CKPT_PATH}")
    print(f"  Best Step:              {ckpt.get('step', 'N/A')}")
    print(f"  Model Parameter Count:  {total_params:,} (Core: {budget['core']:,}, Table: {budget['table']:,})")
    print(f"  Context Length:         {cfg.seq_len}")
    print(f"  Tokenizer Vocab Size:   {cfg.vocab_size}")
    print(f"  Max Generation Tokens:  {MAX_NEW_TOKENS}")
    print("=" * 65)
    print("Enter natural language commands. Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("\nUser Input > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting inference session.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting inference session.")
            break

        gen_text, gen_ids, is_valid_json, parsed_json = generate_response(
            model, tok, user_input, device, MAX_NEW_TOKENS
        )

        print("-" * 65)
        print(f"INPUT:       {user_input}")
        print(f"OUTPUT:      {gen_text}")
        print(f"TOKENS:      Count={len(gen_ids)} | IDs={gen_ids}")
        print(f"VALID_JSON:  {is_valid_json}")
        if is_valid_json and parsed_json is not None:
            print(f"PARSED JSON: {json.dumps(parsed_json, indent=2)}")
        print("-" * 65)


if __name__ == "__main__":
    main()
