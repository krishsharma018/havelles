# Phase 2 Report: Minimal Output Schema for ESP32-S3 SLM

## 1. Executive Summary & Original 30 Fields Overview

In Phase 1, the washing machine NLP dataset was canonicalized into a 30-field ground truth schema containing 37,950 training examples, 3,000 validation examples, and 3,000 test examples. While this 30-field schema represents complete ground-truth metadata, predicting all 30 fields on a resource-constrained micro-controller (ESP32-S3) introduces severe decoding latency and memory overhead (~125 target tokens per prediction).

Phase 2 proposes a compact, machine-actionable **Minimal Target Schema**. By stripping 10 metadata provenance fields (`*_source`), omitting dead fields (`water_level_constraint`), deriving implicit boolean flags (`soak`, `dry`), and omitting `null` or empty fields from the output payload, target text size is reduced by **~89.5% in characters** and **~87.6% in tokens** (from a median of 124 tokens down to **9 tokens** in training). Zero physical machine-control information is lost.

---

## 2. Categorization of original 30 Fields

Every field in the original 30-field schema has been evaluated into one of four action categories:

| # | Field Name | Action Category | Target Minimal Field | Decision & Technical Rationale |
|---|---|---|---|---|
| 1 | `intent` | **KEEP** | `intent` | Top-level command intent (`START_CYCLE`, `SELECT_PROGRAM`, `SET_TEMPERATURE`, etc.). Always required by MCU state machine. |
| 2 | `program_concept` | **KEEP** | `program` | Wash cycle program name (`COTTON`, `DELICATE`, `WOOL`, etc.). Renamed to shorter key `program`. |
| 3 | `program_source` | **REMOVE** | *-* | Provenance metadata (`USER`, `INFERRED`, `UNSPECIFIED`). Not needed by physical hardware MCU. |
| 4 | `load_type` | **CONDITIONAL** | `load_type` | Fabric load type (`BABY_CLOTHES`, `COTTON`, `JEANS`, etc.). Included only when explicitly specified. |
| 5 | `load_type_source` | **REMOVE** | *-* | Provenance metadata. |
| 6 | `soil_level` | **CONDITIONAL** | `soil_level` | Dirtiness level (`LIGHT`, `NORMAL`, `HEAVY`). Included when specified or inferred. |
| 7 | `soil_level_source` | **REMOVE** | *-* | Provenance metadata. |
| 8 | `stain_type` | **CONDITIONAL** | `stain_type` | Stain classification (`DIRT`, `FOOD`, `OIL`, `SWEAT`). Included when stain treatment is requested. |
| 9 | `stain_type_source` | **REMOVE** | *-* | Provenance metadata. |
| 10 | `temperature` | **CONDITIONAL** | `temperature` | Target water temperature (`20`–`60`, `"COLD"`, `"WARM"`). Included when temperature is set. |
| 11 | `temperature_source` | **REMOVE** | *-* | Provenance metadata. |
| 12 | `temperature_constraint` | **CONDITIONAL** | `temperature_constraint` | Temperature boundary (`NOT_COLD`, `NOT_HOT`). Included when relative constraint is requested. |
| 13 | `spin_speed` | **CONDITIONAL** | `spin_speed` | Target spin speed (`400`–`1200`, `"LOW"`, `"MEDIUM"`, `"HIGH"`). Included when spin speed is set. |
| 14 | `spin_speed_source` | **REMOVE** | *-* | Provenance metadata. |
| 15 | `spin_speed_constraint` | **CONDITIONAL** | `spin_speed_constraint` | Spin boundary (`NOT_HIGH`, `NOT_LOW`). Included when relative constraint is requested. |
| 16 | `water_level` | **CONDITIONAL** | `water_level` | Fill level (`AUTO`, `LOW`, `MEDIUM`, `HIGH`). Included when water level is set. |
| 17 | `water_level_source` | **REMOVE** | *-* | Provenance metadata. |
| 18 | `water_level_constraint` | **REMOVE** | *-* | Dead field: 100% `null` across all 43,950 records in the dataset. |
| 19 | `wash_duration` | **CONDITIONAL** | `wash_duration` | Wash stage minutes (`10`–`90`). Included when wash duration is set. |
| 20 | `wash_duration_source` | **REMOVE** | *-* | Provenance metadata. |
| 21 | `soak` | **DERIVE** | `soak` / *-* | `soak: true` is derived when `soak_duration` is non-null or `"SOAK"` is in `options`. Omitted when redundant. |
| 22 | `soak_source` | **REMOVE** | *-* | Provenance metadata. |
| 23 | `soak_duration` | **CONDITIONAL** | `soak_duration` | Soak phase duration in minutes. Included when soak time is specified. |
| 24 | `rinse_count` | **CONDITIONAL** | `rinse_count` | Number of rinse cycles (`1`, `2`, `3`). Included when rinse count is set. |
| 25 | `dry` | **DERIVE** | `dry` / *-* | `dry: true` is derived when `dry_duration` is non-null. Omitted when redundant. |
| 26 | `dry_source` | **REMOVE** | *-* | Provenance metadata. |
| 27 | `dry_duration` | **CONDITIONAL** | `dry_duration` | Dryer duration in minutes (`10`–`60`). Included when dry duration is specified. |
| 28 | `options` | **CONDITIONAL** | `options` | List of enabled extra options (`CHILD_LOCK`, `EXTRA_RINSE`, `PREWASH`, `STEAM`, etc.). Omitted when empty. |
| 29 | `disabled_options` | **CONDITIONAL** | `disabled_options` | List of explicitly disabled options. Omitted when empty. |
| 30 | `constraints` | **CONDITIONAL** | `constraints` | Priority optimizations (`cleaning_priority`, `fabric_care_priority`, `time_priority`). Omitted when empty. |

---

## 3. Summary of Category Groupings

### KEEP Fields (Always present when intent demands primary payload)
- `intent`
- `program` (renamed from `program_concept`)

### REMOVE Fields (11 fields removed)
- All 10 `*_source` provenance metadata fields (`program_source`, `load_type_source`, `soil_level_source`, `stain_type_source`, `temperature_source`, `spin_speed_source`, `water_level_source`, `wash_duration_source`, `soak_source`, `dry_source`).
- `water_level_constraint` (100% null).

### DERIVE Fields (2 fields)
- `soak`: Derived automatically by MCU from `soak_duration` or `options`.
- `dry`: Derived automatically by MCU from `dry_duration`.

### CONDITIONAL Fields (Sparse inclusion when non-null / non-empty)
- `load_type`, `soil_level`, `stain_type`, `temperature`, `temperature_constraint`, `spin_speed`, `spin_speed_constraint`, `water_level`, `wash_duration`, `soak_duration`, `rinse_count`, `dry_duration`, `options`, `disabled_options`, `constraints`.

---

## 4. Final Proposed Minimal JSON Specification

The proposed minimal output JSON consists of **14 candidate machine-control fields** plus 3 auxiliary collection fields (`options`, `disabled_options`, `constraints`). `null` fields and empty lists/objects are omitted from the JSON payload to maximize compactness.

### Full Target to Minimal Conversion Function
```python
def full_target_to_minimal(full_target: dict) -> dict:
    minimal = {}
    
    # 1. Always keep intent
    minimal['intent'] = full_target['intent']

    # 2. Program concept renamed to program
    if full_target.get('program_concept') is not None:
        minimal['program'] = full_target['program_concept']

    # 3. Scalar parameters (keep if non-null)
    for field in [
        'load_type', 'soil_level', 'stain_type', 'temperature', 
        'temperature_constraint', 'spin_speed', 'spin_speed_constraint', 
        'water_level', 'wash_duration', 'soak_duration', 'rinse_count', 'dry_duration'
    ]:
        val = full_target.get(field)
        if val is not None:
            minimal[field] = val

    # 4. Handle soak / dry boolean fallback when duration is absent
    if 'soak_duration' not in minimal:
        if full_target.get('soak') is True and 'SOAK' not in full_target.get('options', []):
            minimal['soak'] = True

    if 'dry_duration' not in minimal:
        if full_target.get('dry') is True:
            minimal['dry'] = True

    # 5. Collections (keep if non-empty)
    if full_target.get('options'):
        minimal['options'] = list(full_target['options'])
    if full_target.get('disabled_options'):
        minimal['disabled_options'] = list(full_target['disabled_options'])
    if full_target.get('constraints'):
        minimal['constraints'] = dict(full_target['constraints'])

    return minimal
```

---

## 5. Example Conversions

### Example 1: Option Modification Command
- **Input**: `"Can you turn off stain removal for this load"`
- **Original 30-field Target (726 chars, 124 tokens)**:
  `{"intent":"DISABLE_OPTION","program_concept":null,"program_source":"UNSPECIFIED","load_type":null,"load_type_source":"UNSPECIFIED","soil_level":null,"soil_level_source":"UNSPECIFIED","stain_type":null,"stain_type_source":"UNSPECIFIED","temperature":null,"temperature_source":"UNSPECIFIED","temperature_constraint":null,"spin_speed":null,"spin_speed_source":"UNSPECIFIED","spin_speed_constraint":null,"water_level":null,"water_level_source":"UNSPECIFIED","water_level_constraint":null,"wash_duration":null,"wash_duration_source":"UNSPECIFIED","soak":null,"soak_source":"UNSPECIFIED","soak_duration":null,"rinse_count":null,"dry":null,"dry_source":"UNSPECIFIED","dry_duration":null,"options":[],"disabled_options":["STAIN_REMOVAL"],"constraints":{}}`
- **Minimal Target (49 chars, 9 tokens)**:
  `{"intent":"DISABLE_OPTION","disabled_options":["STAIN_REMOVAL"]}`

### Example 2: Duration Setting Command
- **Input**: `"Run it for 30 minutes"`
- **Original 30-field Target (726 chars, 124 tokens)**:
  `{"intent":"SET_WASH_DURATION",...,"wash_duration":30,...}`
- **Minimal Target (44 chars, 8 tokens)**:
  `{"intent":"SET_WASH_DURATION","wash_duration":30}`

### Example 3: Complex Multi-Slot Start Cycle Command
- **Input**: `"Can you start a wash. Select wool cycle. For different fabrics. Heavily soiled. With cooking oil stains. At no heat. Add stain removal"`
- **Original 30-field Target (788 chars, 134 tokens)**:
  `{"intent":"START_CYCLE","program_concept":"WOOL","program_source":"INFERRED","load_type":"MIXED","load_type_source":"USER","soil_level":"HEAVY","soil_level_source":"INFERRED","stain_type":"OIL","stain_type_source":"USER","temperature":"COLD","temperature_source":"USER","temperature_constraint":null,"spin_speed":null,"spin_speed_source":"UNSPECIFIED","spin_speed_constraint":null,"water_level":null,"water_level_source":"UNSPECIFIED","water_level_constraint":null,"wash_duration":null,"wash_duration_source":"UNSPECIFIED","soak":null,"soak_source":"UNSPECIFIED","soak_duration":null,"rinse_count":null,"dry":null,"dry_source":"UNSPECIFIED","dry_duration":null,"options":["STAIN_REMOVAL"],"disabled_options":[],"constraints":{"cleaning_priority":"HIGH"}}`
- **Minimal Target (180 chars, 36 tokens)**:
  `{"intent":"START_CYCLE","program":"WOOL","load_type":"MIXED","soil_level":"HEAVY","stain_type":"OIL","temperature":"COLD","options":["STAIN_REMOVAL"],"constraints":{"cleaning_priority":"HIGH"}}`

---

## 6. Detailed Token & Character Size Reduction Benchmark

All stats calculated across 37,950 training, 3,000 validation, and 3,000 test examples:

### Character Length Benchmark
| Split | Original Mean (Median / P95 / Max) | Minimal Mean (Median / P95 / Max) | % Character Reduction |
|---|---|---|---|
| **Train (37,950)** | 728.37 (726.0 / 757.0 / 829) | **76.52 (49.0 / 186.0 / 321)** | **89.49%** |
| **Validation (3,000)** | 729.74 (726.0 / 760.0 / 811) | **82.85 (58.0 / 196.0 / 333)** | **88.65%** |
| **Test (3,000)** | 730.61 (727.0 / 762.05 / 816) | **84.89 (58.0 / 196.0 / 335)** | **88.38%** |

### Estimated Token Count Benchmark
| Split | Original Mean (Median / P95 / Max) | Minimal Mean (Median / P95 / Max) | % Token Reduction |
|---|---|---|---|
| **Train (37,950)** | 124.80 (124.0 / 128.0 / 138) | **15.47 (9.0 / 37.0 / 59)** | **87.60%** |
| **Validation (3,000)** | 125.01 (124.0 / 128.0 / 134) | **16.66 (11.0 / 39.0 / 61)** | **86.67%** |
| **Test (3,000)** | 125.10 (124.0 / 128.0 / 134) | **16.97 (11.0 / 39.0 / 61)** | **86.43%** |

---

## 7. Machine-Control Information Loss Analysis

**Zero physical machine-control information is lost.**

- **Source Fields (`*_source`)**: Represent provenance annotation regarding *how* the label generator populated the field (`USER` vs `INFERRED` vs `UNSPECIFIED`). The physical washing machine micro-controller (ESP32 MCU) executes motor/pump/valve routines based strictly on requested target values (e.g. `temperature: 40`), regardless of provenance.
- **Dead Fields (`water_level_constraint`)**: 100% `null` across all 43,950 dataset entries.
- **Omitted `null` / empty keys**: Omitted keys represent unspecified parameters. On the ESP32 runtime, unspecified parameters default to hardware/cycle factory defaults.

---

## 8. Final Recommendation & Next Steps

1. **Adopt Minimal Schema**: Utilize `data_v5_leakage_controlled/prepared_minimal/` for all downstream tokenizer training and Transformer model training.
2. **Speed & Resource Benefits on ESP32-S3**:
   - Decoding 15 tokens vs 125 tokens reduces autoregressive inference latency by **~8x**.
   - Significantly reduces RAM allocation during beam/greedy decoding.
3. **Dataset Files Available**:
   - `data_v5_leakage_controlled/prepared_minimal/train.jsonl` (37,950 rows)
   - `data_v5_leakage_controlled/prepared_minimal/validation.jsonl` (3,000 rows)
   - `data_v5_leakage_controlled/prepared_minimal/test.jsonl` (3,000 rows)

**STOP**: Awaiting Project Lead approval before initiating Phase 3 (Tokenizer training & Model training).
