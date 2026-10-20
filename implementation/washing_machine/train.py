"""train.py — Washing Machine SLM Training Script (Phase 4, Tasks 8, 9, 10)
==========================================================================
Trains the PLE Transformer on tokenized washing machine natural language -> minimal JSON
sequences using target-only loss masking.

Usage:
  Sanity check run (2 steps):
    python -m implementation.washing_machine.train --sanity-check

  Full training (requires project lead approval):
    python -m implementation.washing_machine.train --steps 4000 --batch-size 32
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure implementation module is importable
ROOT_DIR = Path(r"E:\havells")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from implementation.washing_machine.model_config import Config, TinyLM, get_recommended_config

# Directories
WASH_DIR = ROOT_DIR / "implementation" / "washing_machine"
SEQ_DIR = WASH_DIR / "sequences"
TOK_DIR = WASH_DIR / "tokenizer"
CHECKPOINT_DIR = WASH_DIR / "checkpoints"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WasherSequenceDataset(Dataset):
    """Dataset for pre-tokenized washing machine input -> target sequences."""

    def __init__(self, jsonl_path: Path, max_seq_len: int = 128):
        self.max_seq_len = max_seq_len
        self.records = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids = rec["combined_sequence_ids"]
                mask = rec["loss_mask"]
                
                # Truncate or Pad to max_seq_len
                if len(ids) > max_seq_len:
                    ids = ids[:max_seq_len]
                    mask = mask[:max_seq_len]
                elif len(ids) < max_seq_len:
                    pad_len = max_seq_len - len(ids)
                    ids = ids + [0] * pad_len  # Pad with 0
                    mask = mask + [0] * pad_len

                # Input is ids[:-1], Target is ids[1:], Mask is mask[1:]
                input_ids = ids[:-1]
                target_ids = ids[1:]
                loss_mask = mask[1:]

                self.records.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                    torch.tensor(loss_mask, dtype=torch.float32),
                ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.records[idx]


def get_tokenizer_sha256() -> str:
    tok_path = TOK_DIR / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer missing at {tok_path}")
    return hashlib.sha256(tok_path.read_bytes()).hexdigest()


def lr_schedule(step: int, total_steps: int, peak_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 * peak_lr + 0.9 * peak_lr * 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: TinyLM, dataloader: DataLoader, device: str, max_batches: int = 40) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_target_tokens = 0
    batches = 0

    for input_ids, target_ids, loss_mask in dataloader:
        if batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits, loss = model(input_ids, target_ids, loss_mask)
        total_loss += loss.item()

        # Compute accuracy on masked target tokens
        preds = logits.argmax(dim=-1)
        correct = (preds == target_ids) & (loss_mask == 1.0)
        total_correct += correct.sum().item()
        total_target_tokens += loss_mask.sum().item()

        batches += 1

    model.train()
    avg_loss = total_loss / max(1, batches)
    acc = total_correct / max(1.0, total_target_tokens)
    return avg_loss, acc


def run_training(args):
    device = get_device()
    print(f"[Train] Using device: {device}")

    tok_sha = get_tokenizer_sha256()
    print(f"[Train] Tokenizer SHA256: {tok_sha[:16]}...")

    # Load Model Configuration
    cfg = get_recommended_config()
    cfg.seq_len = args.seq_len
    model = TinyLM(cfg).to(device)
    budget = model.param_budget()
    print(f"[Train] Model instantiated | Total params: {budget['total']:,} | Core: {budget['core']:,} | Table: {budget['table']:,}")

    # Load Data
    train_path = SEQ_DIR / "train.jsonl"
    val_path = SEQ_DIR / "validation.jsonl"

    print(f"[Train] Loading training data from {train_path}...")
    train_dataset = WasherSequenceDataset(train_path, max_seq_len=args.seq_len)
    val_dataset = WasherSequenceDataset(val_path, max_seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"[Train] Dataset loaded | Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # Optimizer (decay on 2D matrices, no decay on norms/embeddings)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or "table" in n or "tok_emb" in n:
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.sanity_check:
        print("\n" + "=" * 60)
        print("=== SANITY TEST STARTING (Task 10) ===")
        print("=" * 60)

        model.train()
        train_iter = iter(train_loader)

        for step in range(1, 3):
            input_ids, target_ids, loss_mask = next(train_iter)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits, loss = model(input_ids, target_ids, loss_mask)
            print(f"Sanity Step {step}: Loss = {loss.item():.4f} | Logits shape = {logits.shape}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Test Evaluation
        val_loss, val_acc = evaluate(model, val_loader, device, max_batches=2)
        print(f"Sanity Evaluation: Val Loss = {val_loss:.4f} | Val Accuracy = {val_acc*100:.2f}%")

        # Test Checkpoint Save (Task 9)
        ckpt_path = CHECKPOINT_DIR / "sanity_checkpoint.pt"
        ckpt_payload = {
            "model_config": cfg.__dict__,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "tokenizer_vocab_size": cfg.vocab_size,
            "tokenizer_sha256": tok_sha,
            "context_length": cfg.seq_len,
            "dataset_version": "v5_prepared_minimal",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "seed": args.seed,
            "param_count": budget,
            "ple_config": {"ple_dim": cfg.ple_dim, "n_layers": cfg.n_layers},
            "output_schema_version": "minimal_v1",
            "sanity_passed": True,
        }
        torch.save(ckpt_payload, ckpt_path)
        print(f"Sanity Checkpoint saved to: {ckpt_path}")

        # Test Checkpoint Reload
        reloaded = torch.load(ckpt_path, map_location=device)
        test_model = TinyLM(Config(**reloaded["model_config"])).to(device)
        test_model.load_state_dict(reloaded["state_dict"])
        print(f"Sanity Checkpoint reload verified successfully!")

        print("\n" + "=" * 60)
        print("=== SANITY TEST PASSED CLEANLY! STOPPING AS PER INSTRUCTIONS. ===")
        print("=" * 60)
        return

    # FULL TRAINING RUN (Requires explicit project lead approval)
    print("\n" + "=" * 60)
    print("=== STARTING FULL TRAINING RUN ===")
    print("=" * 60)

    model.train()
    step = 0
    best_val_loss = float("inf")
    t0 = time.time()

    while step < args.steps:
        for input_ids, target_ids, loss_mask in train_loader:
            if step >= args.steps:
                break
            
            lr = lr_schedule(step, args.steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits, loss = model(input_ids, target_ids, loss_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.eval_every == 0 or step == args.steps - 1:
                val_loss, val_acc = evaluate(model, val_loader, device, max_batches=args.eval_iters)
                elapsed = time.time() - t0
                print(
                    f"Step {step:5d}/{args.steps} | LR {lr:.6f} | Train Loss {loss.item():.4f} | "
                    f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}% | Elapsed {elapsed:.1f}s"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CHECKPOINT_DIR / "best_washer_slm.pt"
                    torch.save({
                        "model_config": cfg.__dict__,
                        "state_dict": model.state_dict(),
                        "tokenizer_vocab_size": cfg.vocab_size,
                        "tokenizer_sha256": tok_sha,
                        "context_length": cfg.seq_len,
                        "dataset_version": "v5_prepared_minimal",
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                        "step": step,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "seed": args.seed,
                        "param_count": budget,
                        "ple_config": {"ple_dim": cfg.ple_dim, "n_layers": cfg.n_layers},
                        "output_schema_version": "minimal_v1",
                    }, best_path)

            step += 1


def main():
    ap = argparse.ArgumentParser(description="Train Washing Machine PLE SLM")
    ap.add_argument("--sanity-check", action="store_true", help="Perform 2-step sanity run only")
    ap.add_argument("--steps", type=int, default=4000, help="Total training steps")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--seq-len", type=int, default=128, help="Sequence context length")
    ap.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    ap.add_argument("--warmup", type=int, default=200, help="Warmup steps")
    ap.add_argument("--eval-every", type=int, default=250, help="Evaluation frequency")
    ap.add_argument("--eval-iters", type=int, default=40, help="Evaluation batch count")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run_training(args)


if __name__ == "__main__":
    main()
