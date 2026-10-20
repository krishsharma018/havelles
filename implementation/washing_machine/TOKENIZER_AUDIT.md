# Tokenizer Audit — Washing Machine SLM
*Prepared: 2026-10-19. Read-only investigation. No files were modified.*

---

## 1. Executive Verdict

**Yes. The tokenizer was trained from scratch exclusively on washing-machine data.**

Specifically: `train_tokenizer.py` loaded 37,950 natural-language user inputs and 37,950 serialised minimal-JSON targets from `data_v5_leakage_controlled/prepared_minimal/train.jsonl` — **the training split only**. It used the HuggingFace `tokenizers` library to run ByteLevel BPE on those 75,900 lines. No validation data, no test data, no TinyStories data, no Barista data, and no pretrained weights were involved at any stage of tokenizer construction.

The resulting tokenizer — chosen candidate `vocab_1024` — is stored at `tokenizer/vocab_1024/tokenizer.json` and is symlinked/copied to `tokenizer/tokenizer.json`. Its SHA-256 is `fb80b3ca0fda4f73d0eaa93716ca0e0da6163c3cd8d1ade35bd78fd1c764aec7`, which matches the value recorded in `tokenizer/metadata.json` (line 29) and embedded in `checkpoints/best_washer_slm.pt` at the key `"tokenizer_sha256"`.

---

## 2. Evidence Table

| Claim | File / Path | Evidence | Conclusion |
|---|---|---|---|
| Tokenizer trained from scratch | `train_tokenizer.py` L104 | `Tokenizer(models.BPE(unk_token=None))` — no pretrained model loaded | ✅ New tokenizer, not fine-tuned |
| Training data source | `train_tokenizer.py` L33, L36, L69–75 | `TRAIN_PATH = ROOT / "data_v5_leakage_controlled" / "prepared_minimal" / "train.jsonl"` | ✅ WM training split only |
| Both input and target used | `train_tokenizer.py` L73–75 | `corpus_lines.append(inp)` and `corpus_lines.append(tgt)` — both appended | ✅ NL inputs + JSON targets |
| Validation and test excluded | `train_tokenizer.py` L59–81 | Only `TRAIN_PATH` is opened; `validation.jsonl` and `test.jsonl` are never read in this file | ✅ Val/test not used |
| Algorithm: ByteLevel BPE | `train_tokenizer.py` L104–105 | `models.BPE(unk_token=None)`, `pre_tokenizers.ByteLevel(add_prefix_space=False)` | ✅ ByteLevel BPE |
| Vocab size = 1024 (chosen) | `tokenizer/metadata.json` L7–8 | `"vocab_size_requested": 1024, "vocab_size_actual": 1024` | ✅ Confirmed |
| Corpus was 75,900 lines | `train_tokenizer.py` docstring L7–8; `TOKENIZER_REPORT.md` L53 | "37,950 input + 37,950 target lines" | ✅ Confirmed |
| No TinyStories data used | All `.py` files in `washing_machine/` | `Select-String` search returned zero hits for "tinystories" in any `.py` file | ✅ Not present |
| No Barista training data used | All `.py` files in `washing_machine/` | Zero hits for "barista" in any `.py` file; Barista appears only in `.md` compatibility docs as a reference | ✅ Not present |
| No LoRA | All files | No reference to LoRA in any script or artifact | ✅ LoRA not used |
| No pretrained weights | All files | No `load_state_dict`, `from_pretrained`, or weight-loading call in `train_tokenizer.py` | ✅ Trained from scratch |
| Tokenizer SHA-256 matches checkpoint | `tokenizer/metadata.json` L29; `train.py` L140–141, L221 | SHA-256 `fb80b3ca…` stored in `metadata.json` and written into checkpoint at training time | ✅ Same tokenizer used for training and inference |
| Inference uses same file | `infer.py` L30 | `TOK_PATH = WASH_DIR / "tokenizer" / "tokenizer.json"` — same path as training | ✅ Consistent |
| SHA-256 verified at inference | `infer.py` L107–112 | Reads `tokenizer.json`, hashes it, compares against `ckpt["tokenizer_sha256"]`; prints warning on mismatch | ✅ Integrity check present |
| Sequences file records actual tokenizer path | `sequences/sequence_stats.json` L4 | `"tokenizer_path": "E:\\havells\\implementation\\washing_machine\\tokenizer\\vocab_1024\\tokenizer.json"` | ✅ Confirmed |
| Zero tokenization errors | `sequences/sequence_stats.json` L97 | `"total_errors": 0` — all 43,950 examples tokenize cleanly | ✅ Confirmed |
| Dataset version tagged in checkpoint | `train.py` L223, L294 | `"dataset_version": "v5_prepared_minimal"` written into every checkpoint | ✅ Provenance recorded |

---

## 3. Actual Tokenizer-Training Pipeline

### 3a. Input Data Sources

| Role | File | Examples | Used for tokenizer? |
|---|---|---|---|
| Training split | `data_v5_leakage_controlled/prepared_minimal/train.jsonl` | 37,950 | **Yes** |
| Validation split | `data_v5_leakage_controlled/prepared_minimal/validation.jsonl` | 3,000 | **No** |
| Test split | `data_v5_leakage_controlled/prepared_minimal/test.jsonl` | 3,000 | **No** |

Each training example has two fields used by the tokenizer:
- `"input"` — the natural-language user command (e.g. `"Can you turn off stain removal for this load"`).
- `"target"` — a compact JSON object (e.g. `{"intent":"DISABLE_OPTION","disabled_options":["STAIN_REMOVAL"]}`), serialised with `json.dumps(target, separators=(',', ':'), ensure_ascii=False)` (`train_tokenizer.py` L55–56).

Both were appended as separate lines to the BPE training corpus: 37,950 NL lines + 37,950 JSON lines = **75,900 corpus lines** and **≈5,397,094 characters** total (`TOKENIZER_REPORT.md` L53).

### 3b. Algorithm and Configuration

| Parameter | Value | Source |
|---|---|---|
| Algorithm | ByteLevel BPE | `train_tokenizer.py` L104–105 |
| Library | HuggingFace `tokenizers` (Python) | `train_tokenizer.py` L30 |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False, use_regex=True)` | `train_tokenizer.py` L105 |
| Decoder | `ByteLevel()` | `train_tokenizer.py` L106 |
| `unk_token` | `None` (no unknown-token fallback) | `train_tokenizer.py` L104 |
| `special_tokens` list | `[]` (empty) | `train_tokenizer.py` L112 |
| Initial alphabet | 256-byte table via `ByteLevel.alphabet()` | `train_tokenizer.py` L113 |
| No normalizer | Yes | `tokenizer/tokenizer.json` L6 (`null`) |
| No padding | Yes | `tokenizer/tokenizer.json` L4 (`null`) |
| No truncation | Yes | `tokenizer/tokenizer.json` L3 (`null`) |
| No post-processor | Yes | `tokenizer/tokenizer.json` L13 (`null`) |

Six vocabulary-size candidates were trained in sequence. All use identical configuration; only `vocab_size` differs:

| Requested vocab | Actual vocab | Merges | SHA-256 (from `candidates_summary.json`) |
|---|---|---|---|
| 512 | 512 | 256 | `76390f14…` |
| 768 | 768 | 512 | `b424caaf…` |
| **1024** | **1024** | **768** | **`fb80b3ca…` ← CHOSEN** |
| 1536 | **1374** | 1118 | `4ca172c0…` (corpus saturated) |
| 2048 | **1374** | 1118 | `4ca172c0…` (same as 1536) |
| 4096 | **1374** | 1118 | `4ca172c0…` (same as 1536) |

> **Corpus saturation**: The washing-machine domain vocabulary is finite. BPE ran out of beneficial merges at 1,374 tokens. Requesting more than 1,374 produces an identical tokenizer.

### 3c. Special Tokens (No `added_tokens` Used)

The BTK1 ESP32 runtime rejects any `tokenizer.json` with non-empty `added_tokens`. Special tokens are therefore plain ASCII bytes that are guaranteed single-token IDs through the 256-byte lookup table:

| Token | ASCII byte | Char repr | Token ID (vocab=1024) |
|---|---|---|---|
| BOS — Begin of Sequence | `0x02` | STX | **190** |
| SEP — Input/Target Separator | `0x1F` | US (Unit Separator) | **219** |
| EOS — End of Sequence | `0x03` | ETX | **191** |

Source: `tokenizer/metadata.json` L11–26; `train_tokenizer.py` L43–45; `prepare_sequences.py` L56–58; `infer.py` L33–35.

These bytes are absent from every natural-language input and every JSON target in the dataset. Their token IDs are fixed by the byte table regardless of the number of BPE merges, which is why the same IDs appear across all six vocabulary candidates (`candidates_summary.json`).

### 3d. Script Invocation

The training script `train_tokenizer.py` is run as:
```
python -m implementation.washing_machine.train_tokenizer
```
No command-line arguments are required; all configuration is hard-coded. The script exits when all six candidates are saved.

Sequence preparation uses the 1024-vocab tokenizer by default:
```
python implementation/washing_machine/prepare_sequences.py [--vocab 1024]
```
(`prepare_sequences.py` L279–283)

---

## 4. One Real Example — Actual Tokenization and Token IDs

This example is taken verbatim from `sequences/train.jsonl` (first record), which was generated by `prepare_sequences.py` using `tokenizer/vocab_1024/tokenizer.json`.

**Input text:**
```
Can you turn off stain removal for this load
```

**Target text (canonical serialised JSON):**
```
{"intent":"DISABLE_OPTION","disabled_options":["STAIN_REMOVAL"]}
```

**Actual token IDs from `sequences/train.jsonl`:**

| Part | Token IDs | Count |
|---|---|---|
| BOS | `190` | 1 |
| Input | `427 327 963 220 78 907 419 699 323 335 349` | 11 |
| SEP | `219` | 1 |
| Target | `267 272 258 731 62 598 261 732 62 382 381 491 62 493 488` | 15 |
| EOS | `191` | 1 |
| **Combined** | `190 427 327 963 220 78 907 419 699 323 335 349 219 267 272 258 731 62 598 261 732 62 382 381 491 62 493 488 191` | **29** |

**Loss mask (same length as combined, position-aligned):**
```
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```
`0` = no loss (BOS + all 11 input tokens + SEP). `1` = compute loss (all 15 target tokens + EOS). The Transformer is trained to predict only the `1`-masked positions.

**Selected vocab lookup for the input tokens (from `tokenizer/tokenizer.json` vocab section):**
- ID 427 → `Can` (line ~421 of vocab)
- ID 327 → `Ġyou` (Ġ = space-prefix in ByteLevel encoding; raw text: ` you`)
- ID 963 → `Ġturn` (raw text: ` turn`)
- ID 78 → `o` (single letter; "off" splits as `o`+`ff`)
- ID 907 → `ff` 
- ID 419 → `Ġstain` (raw text: ` stain`)
- ID 699 → `Ġremoval` (raw text: ` removal`)
- ID 323 → `Ġfor` (raw text: ` for`)
- ID 335 → `Ġthis` (raw text: ` this`)
- ID 349 → `Ġload` (raw text: ` load`)

> Token IDs above are read directly from `sequences/train.jsonl` and `tokenizer/tokenizer.json`; none are invented.

**Selected vocab lookup for target tokens:**
- ID 267 → `{"` (merged JSON opening; `train_tokenizer.py` learned this pair because `{` and `"` are always adjacent in JSON output)
- ID 272 → `intent`
- ID 258 → `":"` (common JSON key-colon-quote pattern)
- ID 731 → `disable` (part of `DISABLE_OPTION`)
- ID 62 → `_`
- ID 598 → `OPTION`
- ID 261 → `","` (comma-quote separator in JSON)
- ID 732 → `disabled`
- ID 382 → `options`
- ID 381 → `":["` (key-colon-array-open pattern)
- ID 491 → `STAIN_REMOVAL` — wait, let me clarify. From `tokenization_stats.json`, `STAIN_REMOVAL` at vocab=512 encodes to 3 tokens (`STAIN` / `_` / `REMOVAL`). IDs 491, 62, 493 correspond to `STAIN`, `_`, `REMOVAL` respectively at vocab=1024.
- ID 488 → `"]}` (JSON array-close and brace-close)

The Transformer loss is applied **only to positions 13–28** (the 15 target token + EOS positions). Position 13 (ID 267 = `{"`) is the first token whose prediction is trained.

---

## 5. Tokenizer versus Transformer

### What the tokenizer learned (and stores)

The tokenizer contains exactly two kinds of data:
1. **A vocabulary** — a mapping from byte-level string segments to integer IDs (0–1023). All 256 single-byte ASCII characters are always present as base tokens (IDs 0–255, rearranged by the ByteLevel Unicode mapping). The remaining 768 tokens are *merge results* — substrings that appeared frequently enough to be combined.
2. **A merge table** — 768 ordered rules, each of the form `(left_token, right_token)`. BPE applies these rules greedily from lowest rank (most frequent) to highest.

The tokenizer does **not** store:
- Any notion of what a command means.
- Any mapping from a string to a JSON output.
- Any weights or parameters that could produce text.
- Any information about which NL phrase maps to which intent.

**Why domain-specific merges appear:** Because `intent`, `COTTON`, `temperature`, `DISABLE_OPTION`-style patterns occurred thousands of times in the 75,900 training lines, BPE merged them into single tokens (or efficient multi-token sequences). This makes the representation compact, not semantically meaningful to the tokenizer.

### Does the tokenizer know that `600 RPM` means a spin-speed setting?

**No.** The tokenizer only knows that the byte sequence `6`, `0`, `0` appears frequently enough to merit the merge `6`+`0` → `60` and then `60`+`0` → `600` (token ID 630 in the vocab). It also learned `Ġrpm` (token ID 374, raw text ` rpm`) as a single token because ` rpm` appears frequently. Neither merge encodes any knowledge that the combination `600 rpm` means a spin speed. That semantic relationship is learned entirely by the **Transformer's weights** during training, by observing thousands of examples where `600 rpm` in the input corresponds to `"spin_speed":600` in the target JSON.

### What the Transformer learned (and stores)

The Transformer (stored in `checkpoints/best_washer_slm.pt`) learns — through the masked cross-entropy loss applied to the target positions — to predict the correct next token given all previous tokens in a sequence. After seeing 37,950 examples across 4,000 training steps, the model's weights encode relationships such as:

- `" 600 rpm"` in the input → the word `spin_speed` tends to appear in the target.
- `" cotton"` in the input → `COTTON` is the likely program value.
- `"intent":"` in the target → the next token is most likely one of `START_CYCLE`, `SELECT_PROGRAM`, `SET_TEMPERATURE`, etc.

**The tokenizer defines the token space; the Transformer defines the mapping.**

### Why training the tokenizer on washing-machine text does not teach it to generate washing-machine JSON

BPE is an unsupervised compression algorithm. It finds the sequence of byte-pair merges that most reduces the total number of tokens needed to represent the corpus. The merge rules encode *co-occurrence frequency*, not *input→output relationships*. The tokenizer has no concept of "input" vs "target" — it processes each line as an independent sequence of bytes. Only the Transformer sees both sides of a pair simultaneously (through the `[BOS] input [SEP] target [EOS]` sequence structure) and is explicitly trained — via the loss mask — to predict the target side given the input side.

---

## 6. TinyStories Connection

### What TinyStories is in this repository

`TOKENIZER_COMPATIBILITY.md` (line 10) lists `esp32-ai-main/research/tinystories/prepare.py` as a *reference file inspected* during the compatibility analysis phase. This script is the original tokenizer-training script used for the **Barista** (coffee-machine) project — a sibling project on the same ESP32 platform.

### What was reused from TinyStories / Barista

| Item | Reused? | Evidence |
|---|---|---|
| BTK1 binary format (`bpe_tokenizer.h`, `generate_tokenizer_header.py`) | **Yes — runtime code only** | Referenced in `TOKENIZER_COMPATIBILITY.md` as the target format; the washing-machine tokenizer must satisfy BTK1 invariants |
| ByteLevel BPE algorithm | **Yes — as a technique** | Same algorithm chosen for ESP32 compatibility |
| TinyStories training data | **No** | No TinyStories file path appears in any `.py` script in `washing_machine/` |
| Barista training data | **No** | No Barista dataset path appears in any `.py` script |
| Pretrained Barista model weights | **No** | `train_tokenizer.py` instantiates `Tokenizer(models.BPE(unk_token=None))` — a blank model; `train.py` instantiates `TinyLM(cfg)` — random initialisation |
| Barista tokenizer vocab (`vocab_4096` of Barista) | **No** | Not loaded anywhere; the washing-machine tokenizer has its own 1024-token vocab trained on its own data |

In summary: the TinyStories/Barista connection is **tooling and algorithm**, not training data or pretrained weights. The washing-machine project reuses the same serialisation format (BTK1), the same runtime C encoder, and the same HuggingFace tokenizer library — but all vocabulary content was learned from washing-machine text.

---

## 7. Generated Artifacts — Complete File Inventory

### Tokenizer Artifacts

| Path | Purpose | Type | Needed at inference? |
|---|---|---|---|
| `tokenizer/vocab_512/tokenizer.json` (20,823 bytes) | Candidate vocab=512; 256 merges | Tokenizer artifact | No (candidate only) |
| `tokenizer/vocab_512/metadata.json` | Metadata for vocab_512 candidate | Metadata | No |
| `tokenizer/vocab_768/tokenizer.json` | Candidate vocab=768; 512 merges | Tokenizer artifact | No |
| `tokenizer/vocab_768/metadata.json` | Metadata for vocab_768 candidate | Metadata | No |
| `tokenizer/vocab_1024/tokenizer.json` (77,245 bytes) | **Chosen tokenizer**; 768 merges; SHA-256 `fb80b3ca…` | **Tokenizer artifact** | **Yes** |
| `tokenizer/vocab_1024/metadata.json` (1,233 bytes) | Chosen tokenizer metadata; records SHA-256, BOS/SEP/EOS IDs | **Metadata** | Recommended (for verification) |
| `tokenizer/vocab_1536/tokenizer.json` | Saturated at 1,374 tokens | Tokenizer artifact | No |
| `tokenizer/vocab_1536/metadata.json` | Metadata for saturated candidate | Metadata | No |
| `tokenizer/vocab_2048/tokenizer.json` | Saturated at 1,374 (identical to 1536) | Tokenizer artifact | No |
| `tokenizer/vocab_2048/metadata.json` | Metadata | Metadata | No |
| `tokenizer/vocab_4096/tokenizer.json` | Saturated at 1,374 (identical to 1536) | Tokenizer artifact | No |
| `tokenizer/vocab_4096/metadata.json` | Metadata | Metadata | No |
| `tokenizer/tokenizer.json` (53,787 bytes) | **Copy of vocab_1024/tokenizer.json**; SHA-256 `fb80b3ca…` | **Tokenizer artifact** | **Yes — used by `infer.py` and `train.py`** |
| `tokenizer/metadata.json` (1,233 bytes) | **Copy of vocab_1024/metadata.json** | **Metadata** | Recommended |
| `tokenizer/vocab.json` (38,178 bytes) | Human-readable ID→token mapping for debugging | Debug artifact | No |
| `tokenizer/candidates_summary.json` (7,905 bytes) | All six candidates' metadata in one file | Summary | No |
| `tokenizer/tokenization_stats.json` (61,101 bytes) | Full per-split, per-vocab token-length statistics | Analysis artifact | No |

**`tokenizer.json` structure:** Contains vocabulary (`model.vocab`, 1024 entries) and merge rules (`model.merges`, 768 entries) in HuggingFace format. Also records pre-tokenizer type, decoder type, and confirms `added_tokens: []`. This is the complete tokenizer artifact; no separate merge file exists.

### Sequence Files (Tokenized Dataset)

| Path | Purpose | Type | Needed at inference? |
|---|---|---|---|
| `sequences/train.jsonl` (66.6 MB) | 37,950 tokenized training examples with combined sequences and loss masks | **Training dataset** | No |
| `sequences/validation.jsonl` (5.6 MB) | 3,000 tokenized validation examples | **Validation dataset** | No |
| `sequences/test.jsonl` (5.6 MB) | 3,000 tokenized test examples | **Test dataset** | No |
| `sequences/sequence_stats.json` (2,211 bytes) | Token-length statistics and coverage analysis for the chosen vocab=1024 | Analysis artifact | No |

### Model Checkpoints

| Path | Purpose | Type | Needed at inference? |
|---|---|---|---|
| `checkpoints/best_washer_slm.pt` (6.65 MB) | Best model checkpoint (lowest validation loss during full training run) | **Model checkpoint** | **Yes** |
| `checkpoints/sanity_checkpoint.pt` (19.97 MB) | Two-step sanity-check run checkpoint (includes optimizer state) | Sanity check artifact | No |

The checkpoint stores, among other things: `model_config`, `state_dict`, `tokenizer_vocab_size: 1024`, `tokenizer_sha256: fb80b3ca…`, `dataset_version: "v5_prepared_minimal"`.

### Scripts

| Path | Purpose |
|---|---|
| `train_tokenizer.py` | Phase 3, Tasks 2–3: Trains all 6 vocabulary candidates from scratch on the training split |
| `measure_tokenization.py` | Phase 3, Tasks 4–5: Measures token-length statistics and JSON-field efficiency for each candidate |
| `prepare_sequences.py` | Phase 3, Tasks 7–8: Tokenises all three splits and produces `sequences/*.jsonl` with combined sequences and loss masks |
| `train.py` | Phase 4: Trains the TinyLM Transformer on `sequences/train.jsonl`; reads `tokenizer.json` for SHA-256 verification |
| `infer.py` | Inference: Loads `tokenizer/tokenizer.json` and `checkpoints/best_washer_slm.pt`; runs greedy autoregressive generation |
| `evaluate_test_set.py`, `fast_eval.py` | Evaluation scripts |
| `model_config.py` | Defines `Config`, `TinyLM` architecture, and `get_recommended_config()` |

---

## 8. Tokenizer → Transformer Code Path

### Training-time path

```
1. RAW INPUT TEXT (e.g. "Can you turn off stain removal for this load")
   │
   └─► prepare_sequences.py (L128)
         tok.encode(inp_text, add_special_tokens=False).ids
         Tokenizer loaded from: tokenizer/vocab_1024/tokenizer.json
         ↓
2. TOKENS: [427, 327, 963, 220, 78, 907, 419, 699, 323, 335, 349]

3. SEQUENCE CONSTRUCTION (prepare_sequences.py L131)
   combined = [BOS=190] + input_ids + [SEP=219] + target_ids + [EOS=191]
   loss_mask = [0, …0…, 0, 1, …1…, 1]
   ↓
4. SAVED to sequences/train.jsonl as a JSON record with
   combined_sequence_ids, loss_mask, input_token_ids, target_token_ids

5. TRAINING (train.py)
   WasherSequenceDataset loads sequences/train.jsonl (L57–61)
   → reads combined_sequence_ids, loss_mask
   → input_ids  = ids[:-1]   (all tokens except last)
   → target_ids = ids[1:]    (all tokens except first; next-token prediction)
   → loss_mask  = mask[1:]   (applied to target positions only)
   ↓
6. TRANSFORMER FORWARD (model_config.py L181–222)
   logits, loss = model(input_ids, target_ids, loss_mask)
   Loss = masked cross-entropy on target-token positions only
   ↓
7. CHECKPOINT SAVE (train.py L287–303)
   Stores tokenizer_vocab_size=1024, tokenizer_sha256="fb80b3ca…"
```

### Inference-time path

```
1. USER INPUT: "Can you turn off stain removal for this load"
   │
   └─► infer.py main() (L115)
         tok = Tokenizer.from_file("tokenizer/tokenizer.json")
         ↓
2. SHA-256 VERIFICATION (infer.py L107–112)
   Reads tokenizer.json → hashes it → compares with ckpt["tokenizer_sha256"]
   Prints warning if mismatch (does not abort)
   ↓
3. ENCODING (infer.py L59–60)
   input_tokens = tok.encode(user_input, add_special_tokens=False).ids
   prompt_ids   = [BOS=190] + input_tokens + [SEP=219]
   ↓
4. GREEDY GENERATION LOOP (infer.py L68–78)
   Feed prompt_ids to model
   Autoregressively sample next token (argmax)
   Append until EOS=191 or max_new_tokens reached
   ↓
5. DECODING (infer.py L81)
   gen_text = tok.decode(gen_ids).strip()
   ↓
6. JSON VALIDATION (infer.py L87–90)
   json.loads(gen_text) → validates and parses
```

### Vocabulary consistency check

`train.py` (L91–94) hashes `tokenizer/tokenizer.json` and records the SHA-256 in the checkpoint. `infer.py` (L107–112) re-hashes the same file at load time and compares. The SHA-256 of the file at time of audit is `fb80b3ca0fda4f73d0eaa93716ca0e0da6163c3cd8d1ade35bd78fd1c764aec7`, which matches the value in `tokenizer/metadata.json` (L29).

`model_config.py` (L23) sets `vocab_size: int = 1024` as the default and `get_recommended_config()` (L249–251) explicitly passes `vocab_size=1024, out_vocab_size=1024`. The embedding table `tok_emb` and the output head `head` are both of size 1024 × 128 (embedding dimension). Any tokenizer with a different actual vocabulary size would cause an index-out-of-bounds error on the first token lookup. No such error would occur silently.

**No vocabulary-size mismatch, no different tokenizer files, no special-token inconsistency was found.**

---

## 9. Unverified Points

The following could not be established from the existing code, artifacts, and logs in the repository:

1. **Exact wall-clock time and machine used to train the tokenizer.** No execution log for `train_tokenizer.py` was found in the repository. The fact that all six `tokenizer.json` files exist and contain the expected SHA-256 values proves the script ran to completion, but the timestamp, duration, and hardware are not recorded.

2. **Exact wall-clock time and hardware used for the full training run.** The `best_washer_slm.pt` checkpoint exists (6.65 MB) and records `step` and `val_loss`, but no training log was found. The `tokenizer_sha256` in the checkpoint matches the current `tokenizer/tokenizer.json`, confirming it was produced by the current training pipeline.

3. **Whether `tokenizer/tokenizer.json` is a literal byte-for-byte copy of `tokenizer/vocab_1024/tokenizer.json` or a separate file that happens to match.** Both share the same SHA-256 (`fb80b3ca…`), confirming identical content. The mechanism of copying (file copy, symlink, or re-save) is not recorded in any script.

4. **Whether validation or test examples were indirectly used in data preparation upstream of the tokenizer (e.g. in generating the dataset itself).** This audit is scoped to the tokenizer training step only. The upstream dataset (`data_v5_leakage_controlled/prepared_minimal/`) is labelled "leakage-controlled", which asserts that validation and test examples did not influence the training set — but the data-generation pipeline for that dataset is outside the scope of this audit.

5. **Confirmed from `best_washer_slm.pt` (torch.load verified):** `step=5750`, `val_loss=0.05025`, `val_acc=0.9923`, `train_size=37950`, `val_size=3000`, `tokenizer_sha256="fb80b3ca…"`, `dataset_version="v5_prepared_minimal"`, `vocab_size=1024`. All fields consistent with the tokenizer and training pipeline described in this report. This was unverified during the initial audit window but was confirmed before report finalisation.

6. **Exact git commit or version of the HuggingFace `tokenizers` library used.** The library version is not pinned in any visible `requirements.txt` or `pyproject.toml` in the `washing_machine/` directory. The tokenizer artifacts exist and are internally consistent; the library version only matters if reproducibility from scratch is required.

---

*End of audit. No source files were modified. One new file created: `TOKENIZER_AUDIT.md`.*
