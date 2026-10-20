# Tokenizer Compatibility Analysis — Washing Machine SLM Phase 3

## Source Files Inspected

| File | Purpose |
|---|---|
| `esp32-ai-main/runtime/bpe_tokenizer.h` | Runtime C encoder (host + device) |
| `esp32-ai-main/firmware/esp32_barista/tools/generate_tokenizer_header.py` | BTK1 asset packer from tokenizer.json |
| `esp32-ai-main/firmware/esp32_barista/tools/verify_tokenizer.py` | Conformance checker |
| `esp32-ai-main/research/tinystories/prepare.py` | Tokenizer trainer (HuggingFace ByteLevel BPE) |
| `esp32-ai-main/src/model.py` | TinyLM model with Config |
| `esp32-ai-main/firmware/esp32_barista/esp32_barista.ino` | Barista runtime generation loop |

---

## 1. Tokenizer Algorithm

**ByteLevel BPE** (identical to GPT-2 / Hugging Face ByteLevel pre-tokenizer).

The runtime in `bpe_tokenizer.h` implements `bpe_encode_ascii()` which:

1. **Pre-tokenizes** the input ASCII text into pieces using the same regex rules as `ByteLevel(add_prefix_space=False, use_regex=True)`:
   - Contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
   - `space+letter*` sequences (word tokens, optionally space-prefixed)
   - `space+digit*` sequences (number tokens)
   - `space+non-whitespace-non-alpha-non-digit*` (punctuation tokens)
   - Runs of pure digits
   - Runs of pure letters
   - Runs of pure punctuation
   - Whitespace runs (consumed as prefix for next token)

2. **BPE merges** each piece independently using a ranked merge table (binary search over sorted `uint32 (left<<16 | right)` keys).

3. Returns token ids as `uint16_t[]`.

**CRITICAL CONSTRAINT**: The runtime **only accepts ASCII (bytes < 0x80)**. Non-ASCII input returns `BTK_ERR_NOT_ASCII (-3)`.

---

## 2. Vocabulary Representation

- Vocabulary is a dense range `0 .. active_vocab-1`.
- All 256 single-byte symbols are required (`byte_ids[]` table = 256 × uint16).
- The mapping uses the **GPT-2/HuggingFace `bytes_to_unicode()`** function — printable ASCII maps to itself; control chars and high bytes are lifted above 0xFF.
- Token IDs are `uint16_t`, so vocab is bounded at **65,536 max**.
- The runtime stores `active_vocab` and `merge_base` explicitly in the BTK1 header.

---

## 3. Merge Representation

**BTK1 Binary Asset Format** (`bpe_tokenizer.h`, `generate_tokenizer_header.py`):

```
Offset  Size   Field
0       4      Magic "BTK1"
4       4      Format version = 2
8       4      active_vocab (vocabulary size)
12      4      merge_count
16      4      Reserved = 0
20      4      merge_base (token id of rank-0 merge result)
24      28     Truncated SHA-256 of tokenizer.json (provenance, not checked at runtime)
52      512    256 × uint16 byte token ids (indexed by raw byte)
564     6*n    Merge entries: uint32 key (left<<16 | right), uint16 rank
               Sorted ascending by key for binary search.
```

**Invariants enforced by the packer**:
- All merge result IDs must equal `merge_base + rank` — strictly sequential.
- Merge table must be sorted by key.
- Vocab IDs must be a dense bijection onto `0..N-1`.

---

## 4. Special Token Handling

**The Barista runtime uses NO special token IDs in the BPE encoder asset.** Key facts:
- Offset 16 (reserved) is explicitly `0`, **NOT** an EOS token id.
- BOS/EOS in Barista are **output classes** in the word tables (`BARISTA_BOS`, `BARISTA_EOS`), NOT token IDs fed through the tokenizer.
- `added_tokens` in `tokenizer.json` cause the packer to **REJECT** the asset with an error: `"{n} added tokens are configured; the asset carries no table for them"`.

**Therefore**: For the washing machine SLM, special tokens `[BOS]`, `[SEP]`, `[EOS]` **cannot be implemented as `added_tokens` in tokenizer.json** if we want to use the existing BTK1 format directly.

### Options for Special Tokens:

**Option A (Recommended)**: Represent BOS/SEP/EOS as plain text strings (e.g. `"\x02"`, `"\x03"`, `"\x04"` — printable ASCII control-like). They will be encoded as single byte tokens from the 256-byte-symbol layer. Token IDs will be whatever `byte_ids[0x02]` etc. map to in the vocabulary.

**Option B**: Reserve three specific single-byte ASCII characters that are unlikely to appear in training text and use them as sentinels. Good candidates: ASCII SOH `\x01`, STX `\x02`, ETX `\x03` (bytes 1, 2, 3 — valid ASCII, not naturally occurring in NL text or JSON).

**Option C**: Prefix/suffix the input and target text with ASCII markers before tokenization (e.g. `"<s>"`, `"</s>"`, `"<sep>"`). These tokenize as multi-token subword sequences but are fully representable in the vocabulary.

**This analysis recommends Option A**: Use single reserved ASCII bytes `\x02` (BOS), `\x1f` (SEP), `\x03` (EOS). These are:
- Valid ASCII (< 0x80) — accepted by `bpe_encode_ascii()`.
- Single bytes → guaranteed single token IDs via the 256-byte lookup table.
- Not present in any natural language or JSON text in the dataset.
- Deterministic — their token IDs are fixed by the byte table, regardless of vocabulary size.

---

## 5. Tokenizer File Format

The runtime requires exactly **TWO files**:

| File | Description |
|---|---|
| `tokenizer.json` | HuggingFace tokenizers library format with `"model": {"type": "BPE", ...}` |
| Generated by `generate_tokenizer_header.py` → `tokenizer_encoder.h` | C header with the BTK1 asset as PROGMEM array |

The `tokenizer.json` **must** satisfy:
- `model.type == "BPE"`
- `pre_tokenizer.type == "ByteLevel"`, `add_prefix_space == False`, `use_regex == True`
- `added_tokens` must be **empty** or absent
- No `truncation`, `padding`, `normalizer`, `post_processor`
- No `dropout`, `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `byte_fallback`, `ignore_merges`
- All merge results form a sequential range `merge_base..merge_base+N-1`

---

## 6. Maximum Token ID Supported by Runtime

`BTK_MAX_TABLE_ENTRIES = 0x10000 = 65,536`

Token IDs are `uint16_t`. The runtime explicitly rejects:
- `active_vocab < 256` (too small)
- `active_vocab > 65,536` (too large for uint16)
- `merge_count > 65,536`
- `merge_base + merge_count > active_vocab`

**Maximum supported vocabulary size: 65,536**.

---

## 7. JSON Punctuation Support

Yes. JSON punctuation (`{`, `}`, `[`, `]`, `:`, `,`, `"`, etc.) are all valid ASCII (< 0x80) and are handled by the punctuation branch of `bpe_encode_ascii()`:

```c
} else if (!btk_ascii_space(s[pos]) && !btk_ascii_letter(s[pos]) && !btk_ascii_digit(s[pos])) {
    while (pos < len && !btk_ascii_space(s[pos]) && !btk_ascii_letter(s[pos]) && !btk_ascii_digit(s[pos])) pos++;
}
```

Consecutive punctuation characters (like `{"` or `":`) are grouped into one pre-token piece and then BPE-merged. With sufficient training data, `{"intent"` and common JSON patterns will be merged into single or few tokens.

**Important**: JSON keys like `"intent"` will be tokenized as `"` + `intent` (split at the quote boundary), because `"` is punctuation and `intent` is letters — different pre-token classes.

---

## 8. Arbitrary Reserved/Special Tokens

**No.** The BTK1 format has no mechanism for `added_tokens`. The packer rejects any `tokenizer.json` that specifies added tokens. Special tokens must be handled by choosing specific single bytes as sentinels (see section 4 above).

---

## 9. Custom Vocabulary Generated Outside This Project

**Yes**, with constraints:
- The tokenizer must be trained with HuggingFace `tokenizers` library using `models.BPE(unk_token=None)` and `pre_tokenizers.ByteLevel(add_prefix_space=False)`.
- The resulting `tokenizer.json` must be passed through `generate_tokenizer_header.py` to produce the BTK1 asset.
- The conformance check in `verify_tokenizer.py` verifies Python and C produce identical IDs.
- A custom vocabulary trained on washing machine data is **fully compatible** with the existing runtime, as long as the BTK1 invariants are satisfied.

---

## 10. Asymmetric Vocabulary: Input vs Output

From `esp32_barista.ino` and `llm.h`:

**Barista uses an asymmetric vocabulary** (`Vin=8057 input tokens`, `Vout=854 output classes`). This is specific to the Barista deployment. The washing machine SLM will use a **symmetric vocabulary** where:

- `input_vocab` = vocabulary size (N)
- `output_vocab` = vocabulary size (N) — model reads and writes the same token space

The `llm.h` header `uint32_t vio[2]` reads `input_vocab` and `output_vocab` from the model binary header. For our symmetric model: `vio[0] == vio[1] == N`.

The PLE model supports both:
- Tied head (`FLAG_TIED_HEAD`): output head = first `out_vocab` rows of `tok_emb`
- Untied head: separate tensor appended at end

For a symmetric washing machine model, we will use a **tied head** where input and output share the same vocabulary.

---

## Summary Table

| Property | Value |
|---|---|
| Algorithm | ByteLevel BPE |
| Pre-tokenizer | ByteLevel (add_prefix_space=False, use_regex=True) |
| Max Vocabulary | 65,536 (uint16 constraint) |
| Token ID type | uint16 |
| Special tokens via added_tokens | NOT SUPPORTED by BTK1 |
| Special token approach | Single reserved ASCII bytes (\x02, \x1f, \x03) |
| JSON punctuation support | Yes (punctuation pre-token class) |
| Non-ASCII support | No (bpe_encode_ascii rejects) |
| Custom vocabulary compatible | Yes, if BTK1 invariants are met |
| Input/output vocab | Symmetric (same space) for washing machine |
| Asset format | BTK1 binary (generated from tokenizer.json) |

---

## Compatibility Verdict

The existing ESP32 runtime (`bpe_tokenizer.h` + `generate_tokenizer_header.py`) is **fully compatible** with a custom washing machine domain tokenizer, provided:

1. The tokenizer is trained with HuggingFace `tokenizers.BPE` + `ByteLevel(add_prefix_space=False)`.
2. No `added_tokens` are used (special tokens = reserved ASCII bytes).
3. Vocabulary size ≤ 65,536.
4. The `tokenizer.json` is packed with `generate_tokenizer_header.py` to produce a BTK1 header.
5. The round-trip is verified via `verify_tokenizer.py`.

**No modifications to any existing runtime files are required.**
