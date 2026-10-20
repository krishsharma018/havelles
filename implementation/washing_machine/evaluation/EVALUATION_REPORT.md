# Phase 6 Test-Set Evaluation Report: Washing Machine SLM

## 1. Checkpoint Identification

- **Evaluated Checkpoint**: `best_washer_slm.pt`
- **Best Training Step**: `5750` (out of 10,000 steps)
- **Validation Loss at Best Step**: `0.05025`
- **Validation Accuracy at Best Step**: `99.23%`

## 2. Executive Evaluation Summary

| Metric | Value |
|---|---|
| **Total Test Examples** | `3,000` |
| **Valid JSON Syntax %** | **`97.37%`** (`2921/3000`) |
| **Exact JSON Match %** | **`62.13%`** (`1864/3000`) |
| **Intent Accuracy %** | **`85.30%`** (`2559/3000`) |
| **Malformed JSON Count** | `79` (`2.63%`) |
| **Avg Generated Length** | `23.29` tokens |
| **Max Generated Length** | `71` tokens |
| **Evaluation Runtime** | `661.2` seconds |

## 3. Field-Level Accuracy Breakdown

| Field Name | Total Expected | Correctly Matched | Field Accuracy |
|---|---|---|---|
| `intent` | `3,000` | `2,559` | **`85.30%`** |
| `program` | `1,328` | `829` | **`62.42%`** |
| `load_type` | `577` | `504` | **`87.35%`** |
| `soil_level` | `571` | `484` | **`84.76%`** |
| `stain_type` | `341` | `324` | **`95.01%`** |
| `temperature` | `780` | `736` | **`94.36%`** |
| `spin_speed` | `803` | `635` | **`79.08%`** |
| `wash_duration` | `607` | `587` | **`96.71%`** |
| `disabled_options` | `116` | `116` | **`100.00%`** |
| `constraints` | `684` | `578` | **`84.50%`** |

## 4. Per-Intent Accuracy Breakdown

| Intent Name | Total Test Examples | Intent Accuracy | Exact JSON Match |
|---|---|---|---|
| `CANCEL` | `31` | `12.90%` | **`12.90%`** |
| `DISABLE_OPTION` | `116` | `100.00%` | **`100.00%`** |
| `ENABLE_OPTION` | `100` | `100.00%` | **`100.00%`** |
| `PAUSE` | `32` | `34.38%` | **`34.38%`** |
| `QUERY_PROGRAM` | `32` | `68.75%` | **`68.75%`** |
| `QUERY_REMAINING_TIME` | `33` | `57.58%` | **`57.58%`** |
| `QUERY_STATUS` | `35` | `65.71%` | **`65.71%`** |
| `RESUME` | `26` | `73.08%` | **`73.08%`** |
| `SELECT_PROGRAM` | `211` | `48.82%` | **`48.82%`** |
| `SET_DRY` | `111` | `99.10%` | **`99.10%`** |
| `SET_RINSE` | `79` | `100.00%` | **`100.00%`** |
| `SET_SOAK` | `112` | `96.43%` | **`95.54%`** |
| `SET_SPIN_SPEED` | `200` | `55.50%` | **`54.50%`** |
| `SET_TEMPERATURE` | `147` | `89.80%` | **`82.99%`** |
| `SET_WASH_DURATION` | `175` | `99.43%` | **`99.43%`** |
| `SET_WATER_LEVEL` | `136` | `96.32%` | **`96.32%`** |
| `START_CYCLE` | `1,393` | `92.32%` | **`43.36%`** |
| `STOP` | `31` | `35.48%` | **`35.48%`** |

## 5. Top 20 Failure Examples Analysis

Total failures (non-exact matches): `1136` out of `3000`

### Failure Example 1 [MISSING_program]
- **Input**: `Yaar run the machine. Delicate cycle should work. Just freshen them up. Don't use hot water haan`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "DELICATE", "soil_level": "LIGHT", "temperature_constraint": "NOT_HOT", "constraints": {"time_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","soil_level":"LIGHT","temperature_constraint":"NOT_HOT","constraints":{"time_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "soil_level": "LIGHT", "temperature_constraint": "NOT_HOT", "constraints": {"time_priority": "HIGH"}}`

### Failure Example 2 [INTENT_MISMATCH]
- **Input**: `Yaar fast spin spin is fine boss`
- **Expected JSON**: `{"intent": "SET_SPIN_SPEED", "spin_speed": "HIGH"}`
- **Generated Output**: `{"intent":"QUERY_SPIN_SPEED","spin_speed":"HIGH"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_SPIN_SPEED", "spin_speed": "HIGH"}`

### Failure Example 3 [INTENT_MISMATCH]
- **Input**: `Um start it back up boss?`
- **Expected JSON**: `{"intent": "RESUME"}`
- **Generated Output**: `{"intent":"QUERY_PROGRAM"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_PROGRAM"}`

### Failure Example 4 [INTENT_MISMATCH]
- **Input**: `I want daily wash in the machine`
- **Expected JSON**: `{"intent": "SELECT_PROGRAM", "program": "NORMAL"}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"NORMAL"}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "NORMAL"}`

### Failure Example 5 [MISMATCH_program]
- **Input**: `Listen, take care of these clothes. Spin only should work. Not too dirty. For 25 minutes in the machine`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "SPIN", "soil_level": "LIGHT", "wash_duration": 25, "constraints": {"time_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"SYNTHETIC","soil_level":"LIGHT","wash_duration":25,"constraints":{"time_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "SYNTHETIC", "soil_level": "LIGHT", "wash_duration": 25, "constraints": {"time_priority": "HIGH"}}`

### Failure Example 6 [EXTRA_soil_level,MISMATCH_constraints,MISMATCH_program]
- **Input**: `Do the wash. Mild cycle should work. For silk clothes. Not too fancy, 20 degrees. Don't spin too hard`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "GENTLE", "load_type": "DELICATE", "temperature": 20, "spin_speed_constraint": "NOT_HIGH", "constraints": {"fabric_care_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"SYNTHETIC","load_type":"DELICATE","soil_level":"LIGHT","temperature":20,"spin_speed_constraint":"NOT_HIGH","constraints":{"fabric_care_priority":"HIGH","time_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "SYNTHETIC", "load_type": "DELICATE", "soil_level": "LIGHT", "temperature": 20, "spin_speed_constraint": "NOT_HIGH", "constraints": {"fabric_care_priority": "HIGH", "time_priority": "HIGH"}}`

### Failure Example 7 [MISMATCH_constraints,EXTRA_program]
- **Input**: `Yaar take care of these clothes. Not too fancy, thirty degrees. Don't spin too hard. For 30 minutes. Dry for 45 minutes for the laundry?`
- **Expected JSON**: `{"intent": "START_CYCLE", "temperature": 30, "spin_speed_constraint": "NOT_HIGH", "wash_duration": 30, "dry_duration": 45, "constraints": {"fabric_care_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"NORMAL","temperature":30,"spin_speed_constraint":"NOT_HIGH","wash_duration":30,"dry_duration":45,"constraints":{"fabric_care_priority":"HIGH","time_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "NORMAL", "temperature": 30, "spin_speed_constraint": "NOT_HIGH", "wash_duration": 30, "dry_duration": 45, "constraints": {"fabric_care_priority": "HIGH", "time_priority": "HIGH"}}`

### Failure Example 8 [INTENT_MISMATCH]
- **Input**: `Where has it reached haan`
- **Expected JSON**: `{"intent": "QUERY_STATUS"}`
- **Generated Output**: `{"intent":"QUERY_PROGRAM"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_PROGRAM"}`

### Failure Example 9 [EXTRA_water_level]
- **Input**: `Take care of these clothes. I want woollens program. For sweaters. With oil stains. Keep the water side 20 celsius. Don't spin too hard. Add stain removal. Dry for 10 minutes`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "WOOL", "load_type": "WOOL", "stain_type": "OIL", "temperature": 20, "spin_speed_constraint": "NOT_HIGH", "dry_duration": 10, "options": ["STAIN_REMOVAL"], "constraints": {"cleaning_priority": "HIGH", "fabric_care_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"WOOL","load_type":"WOOL","stain_type":"OIL","temperature":20,"spin_speed_constraint":"NOT_HIGH","water_level":"MEDIUM","dry_duration":10,"options":["STAIN_REMOVAL"],"constraints":{"cleaning_priority":"HIGH","fabric_care_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "WOOL", "load_type": "WOOL", "stain_type": "OIL", "temperature": 20, "spin_speed_constraint": "NOT_HIGH", "water_level": "MEDIUM", "dry_duration": 10, "options": ["STAIN_REMOVAL"], "constraints": {"cleaning_priority": "HIGH", "fabric_care_priority": "HIGH"}}`

### Failure Example 10 [INTENT_MISMATCH]
- **Input**: `Um six hundred rpm spin is fine`
- **Expected JSON**: `{"intent": "SET_SPIN_SPEED", "spin_speed": 600}`
- **Generated Output**: `{"intent":"QUERY_SPIN_SPEED","spin_speed":600}`
- **Parsed Generated JSON**: `{"intent": "QUERY_SPIN_SPEED", "spin_speed": 600}`

### Failure Example 11 [INTENT_MISMATCH]
- **Input**: `Um synthetics program should work in the machine`
- **Expected JSON**: `{"intent": "SELECT_PROGRAM", "program": "SYNTHETIC"}`
- **Generated Output**: `{"intent":"QUERY_REMAINING_TIME"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_REMAINING_TIME"}`

### Failure Example 12 [JSON_SYNTAX_ERROR]
- **Input**: `Uh do the wash. Careful cycle should work. For light sarees. With sauce marks. Spin side twelve hundred rpm. Add stain mode.`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "GENTLE", "load_type": "DELICATE", "stain_type": "FOOD", "spin_speed": 1200, "options": ["STAIN_REMOVAL"], "constraints": {"cleaning_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"QUERY_REMAINING_TIME,"load_type":"DELICATE","stain_type":"FOOD","spin_speed":1200,"options":["STAIN_REMOVAL"],"constraints":{"cleaning_priority":"HIGH"}}`
- **Parsed Generated JSON**: `None`

### Failure Example 13 [MISMATCH_spin_speed]
- **Input**: `Run the machine. Go with just spin. Lightly used. Not too fancy, 30 degrees. Six hundred rpm spin is fine. For 40 minutes only`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "SPIN", "soil_level": "LIGHT", "temperature": 30, "spin_speed": 600, "wash_duration": 40, "constraints": {"time_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"SPIN","soil_level":"LIGHT","temperature":30,"spin_speed":400,"wash_duration":40,"constraints":{"time_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "SPIN", "soil_level": "LIGHT", "temperature": 30, "spin_speed": 400, "wash_duration": 40, "constraints": {"time_priority": "HIGH"}}`

### Failure Example 14 [INTENT_MISMATCH]
- **Input**: `Yaar take care of these clothes. I want just spin. For bedsheets. Muddy. Water around thirty degrees. Dry after`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "SPIN", "load_type": "COTTON", "soil_level": "HEAVY", "temperature": 30, "dry": true, "constraints": {"cleaning_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"SPIN","program":"SPIN","load_type":"COTTON","soil_level":"HEAVY","temperature":30,"dry":true,"constraints":{"cleaning_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "SPIN", "program": "SPIN", "load_type": "COTTON", "soil_level": "HEAVY", "temperature": 30, "dry": true, "constraints": {"cleaning_priority": "HIGH"}}`

### Failure Example 15 [INTENT_MISMATCH]
- **Input**: `Yaar short cycle should work in the machine?`
- **Expected JSON**: `{"intent": "SELECT_PROGRAM", "program": "QUICK"}`
- **Generated Output**: `{"intent":"QUERY_REMAINING_TIME"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_REMAINING_TIME"}`

### Failure Example 16 [INTENT_MISMATCH]
- **Input**: `Uh run the machine. Go with program for cotton clothes. Heavily soiled. Water around no heat?`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "COTTON", "soil_level": "HEAVY", "temperature": "COLD", "constraints": {"cleaning_priority": "HIGH"}}`
- **Generated Output**: `{"intent":"SET_TEMPERATURE","program":"COTTON","soil_level":"HEAVY","temperature":"COLD","constraints":{"cleaning_priority":"HIGH"}}`
- **Parsed Generated JSON**: `{"intent": "SET_TEMPERATURE", "program": "COTTON", "soil_level": "HEAVY", "temperature": "COLD", "constraints": {"cleaning_priority": "HIGH"}}`

### Failure Example 17 [INTENT_MISMATCH]
- **Input**: `Uh medium spin spin is fine for the laundry`
- **Expected JSON**: `{"intent": "SET_SPIN_SPEED", "spin_speed": "MEDIUM"}`
- **Generated Output**: `{"intent":"QUERY_SPIN_SPEED","spin_speed":"MEDIUM"}`
- **Parsed Generated JSON**: `{"intent": "QUERY_SPIN_SPEED", "spin_speed": "MEDIUM"}`

### Failure Example 18 [MISMATCH_program]
- **Input**: `Uh do the wash. Go with cotton wash. For bedsheets. Use medium water level`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "COTTON", "load_type": "COTTON", "water_level": "MEDIUM"}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"SYNTHETIC","load_type":"COTTON","water_level":"MEDIUM"}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "SYNTHETIC", "load_type": "COTTON", "water_level": "MEDIUM"}`

### Failure Example 19 [EXTRA_wash_duration,MISMATCH_program]
- **Input**: `Get this load done. Strong wash should work. For mixed clothes only.`
- **Expected JSON**: `{"intent": "START_CYCLE", "program": "HEAVY", "load_type": "MIXED"}`
- **Generated Output**: `{"intent":"START_CYCLE","program":"SYNTHETIC","load_type":"MIXED","wash_duration":90}`
- **Parsed Generated JSON**: `{"intent": "START_CYCLE", "program": "SYNTHETIC", "load_type": "MIXED", "wash_duration": 90}`

### Failure Example 20 [INTENT_MISMATCH]
- **Input**: `Uh end this washing job boss?`
- **Expected JSON**: `{"intent": "STOP"}`
- **Generated Output**: `{"intent":"CANCEL"}`
- **Parsed Generated JSON**: `{"intent": "CANCEL"}`

