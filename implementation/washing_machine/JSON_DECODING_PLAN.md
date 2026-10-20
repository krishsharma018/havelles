# JSON_DECODING_PLAN.md — Washer Minimal Schema Constrained Decoding Specification

## 1. Objective

When deployed on the ESP32-S3, the small language model generates JSON text token-by-token. To guarantee **100% valid JSON syntax** and **100% valid domain schema values**, an on-device **Constrained Decoder (Grammar Masker)** will filter logits at each generation step.

This document defines the formal grammar, state transitions, legal field orders, value sets, and token constraints required for the future C runtime implementation.

---

## 2. Minimal JSON Target Schema & Field Rules

The minimal washer schema contains **10 key fields**. Fields MUST be generated in a fixed, canonical field order to simplify state transitions and maximize token prediction efficiency.

### 2.1 Canonical Field Order

```
1. "intent"                 (REQUIRED — first key)
2. "program"                (OPTIONAL)
3. "load_type"              (OPTIONAL)
4. "soil_level"             (OPTIONAL)
5. "stain_type"             (OPTIONAL)
6. "temperature"            (OPTIONAL)
7. "spin_speed"             (OPTIONAL)
8. "wash_duration"          (OPTIONAL)
9. "disabled_options"       (OPTIONAL array)
10. "constraints"           (OPTIONAL object)
```

> **Omission Rule**: Null or non-specified attributes MUST be omitted entirely from the generated JSON object. They are never written as `"temperature": null` or `"program": ""`.

---

## 3. Allowed Domain Field Values

### 3.1 `intent` (Required String)
Allowed enum values:
- `"START_CYCLE"`
- `"PAUSE_CYCLE"`
- `"RESUME_CYCLE"`
- `"STOP_CYCLE"`
- `"SELECT_PROGRAM"`
- `"SET_TEMPERATURE"`
- `"SET_SPIN_SPEED"`
- `"SET_WASH_DURATION"`
- `"DISABLE_OPTION"`
- `"SET_CONSTRAINTS"`

### 3.2 `program` (String)
Allowed enum values:
- `"COTTON"`
- `"SYNTHETIC"`
- `"DELICATE"`
- `"WOOL"`
- `"QUICK"`
- `"HEAVY_DUTY"`
- `"BABY_CARE"`
- `"MIXED"`
- `"ECO"`
- `"RINSE_SPIN"`
- `"SPIN_ONLY"`
- `"DRUM_CLEAN"`

### 3.3 `load_type` (String)
Allowed enum values:
- `"COTTON"`
- `"SYNTHETIC"`
- `"DELICATE"`
- `"WOOL"`
- `"MIXED"`
- `"HEAVY"`
- `"BEDDING"`
- `"TOWELS"`
- `"SPORTSWEAR"`
- `"BABY_CLOTHES"`

### 3.4 `soil_level` (String)
Allowed enum values:
- `"LIGHT"`
- `"MEDIUM"`
- `"HEAVY"`
- `"MAX"`

### 3.5 `stain_type` (String)
Allowed enum values:
- `"BLOOD"`
- `"WINE"`
- `"GRASS"`
- `"OIL"`
- `"FOOD"`
- `"SWEAT"`
- `"DIRT"`
- `"MAKEUP"`

### 3.6 `temperature` (Integer)
Allowed integer values (Celsius):
- `20`, `30`, `40`, `60`, `90`, `0` (cold/tap)

### 3.7 `spin_speed` (Integer)
Allowed integer values (RPM):
- `400`, `600`, `800`, `1000`, `1200`, `1400`, `0` (no spin)

### 3.8 `wash_duration` (Integer)
Allowed integer values (Minutes):
- `15`, `30`, `45`, `60`, `90`, `120`

### 3.9 `disabled_options` (Array of Strings)
Allowed elements:
- `"PREWASH"`
- `"EXTRA_RINSE"`
- `"STEAM"`
- `"SOAK"`
- `"STAIN_REMOVAL"`
- `"EASY_IRON"`

### 3.10 `constraints` (Sub-Object)
Allowed key-value pairs:
- `"temperature_constraint"`: `"NOT_HOT"`, `"NOT_COLD"`, `"MAX_40"`, `"MAX_60"`
- `"spin_speed_constraint"`: `"NO_HIGH_SPIN"`, `"MAX_800"`

---

## 4. Finite State Machine (FSM) Decoding Transition States

```
[State 0: START] ─── token '{' ───> [State 1: EXPECT_KEY_OR_CLOSE]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
          "intent" key                                       token '}'
                  │                                               │
                  ▼                                               ▼
          [State 2: EXPECT_COLON]                          [State END / EOS]
                  │
                  ▼
          [State 3: EXPECT_INTENT_VAL]
                  │
                  ▼
          [State 4: EXPECT_COMMA_OR_CLOSE]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
    token ','           token '}'
        │                   │
        ▼                   ▼
[State 5: NEXT_KEY]    [State END / EOS]
```

### 4.1 State Rules:

1. **State 0 (START)**:
   - Only token `{` (ID 267/90) is allowed.

2. **State 1 (KEY_SELECTION)**:
   - Allowed key tokens according to canonical field order.
   - Once field $k_i$ is emitted or skipped, state transition moves to field $k_{j > i}$. Next key MUST follow canonical order.

3. **State 2 (VAL_SELECTION)**:
   - For string fields (`program`, `soil_level`, etc.): Only tokens corresponding to quotes `"` followed by allowed enum values.
   - For integer fields (`temperature`, `spin_speed`, etc.): Only tokens corresponding to allowed integer literals.
   - For array field (`disabled_options`): Token `[` followed by array elements separated by `,` ending with `]`.

4. **State 3 (END/EOS)**:
   - The `EOS` token (ID 191 / `\x03`) is strictly **FORBIDDEN** until the final closing brace `}` of the JSON object has been generated.
   - Immediately following `}`, `EOS` is enforced as the **ONLY** allowed token.

---

## 5. Token Bitmasking Mechanism for C-Runtime

On ESP32-S3:
1. Maintain an array `uint32_t allowed_mask[1024 / 32]` (32 uint32s = 128 bytes).
2. Before calling `top_k` / sampling, the FSM state sets bits for legal token IDs at the current step and zeroes out forbidden tokens.
3. For forbidden tokens: `logits[i] = -1e9f`.
4. This guarantees **zero syntax errors** and **zero schema violations** at runtime.
