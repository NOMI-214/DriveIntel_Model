"""
data_loader.py — Load and normalize all DTC JSON files into a unified DataFrame.

Handles six distinct schema families found across the dataset:

  A. Standard vehicle files  (Toyota, Hyundai, KIA, …)
       {dtcs: [{dtc_code, description, possible_causes, steps, …}]}

  B. Audi combined file  (audi_all_dtc_combined.json)
       {vehicles: [{vehicle, model, p_codes: {codes: [...]}, u_codes: {codes: [...]}}]}
       inner record: {code, category, description, malfunction_criteria, diagnostic_procedure}

  C. Audi per-vehicle file  (audi_dtc_codes.json)
       {vehicles: [{vehicle, source_file, dtc_codes: [{code, description, diagnostic_procedure}]}]}

  D. Pakistan generic dataset  (pakistan_dtc_dataset.json)
       {dtc_codes: [{code, type, system, description, diagnostic_procedure, pak_vehicles}]}

  E. Pakistan complete vehicle edition  (pakistan_vehicles_complete_dtc.json)
       {categories: [{category, vehicles: [{make, model, dtc_codes: [{code, system, description, …}]}]}]}

  F. Plain list of DTC records  (some legacy files)
       [{code/dtc_code, description, …}]

Each DTC record → {dtc_code, description, possible_causes, steps_text,
                   system, code_prefix, code_number, vehicle_make, vehicle_model}
"""

import json
import os
import re
import glob
from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent.parent / "Data"

# ── file-name → (make, model) parser ──────────────────────────────────────
_FILE_RE = re.compile(
    r"^(?P<make>[A-Za-z]+)_(?P<model>.+?)(?:_(?:P|B|C|U|L|DTC|dtc|Codes?|ALL).*)?_dtc\.json$",
    re.IGNORECASE,
)


def _parse_filename(fname: str):
    fname = os.path.basename(fname)
    m = _FILE_RE.match(fname)
    if m:
        return m.group("make"), m.group("model").replace("_", " ")
    parts = fname.replace("_dtc.json", "").replace(".json", "").split("_")
    return parts[0] if parts else "Unknown", " ".join(parts[1:]) if len(parts) > 1 else "Unknown"


# ── DTC record normaliser (common path) ───────────────────────────────────

def _normalise_record(rec: dict, make: str, model: str) -> dict | None:
    """Return a flat dict or None if the record has no usable code."""
    code = (rec.get("dtc_code") or rec.get("code") or "").strip().upper()
    if not code or len(code) < 2:
        return None

    # Description — various field names across schemas
    desc = (rec.get("description") or rec.get("fault_text")
            or rec.get("details") or rec.get("malfunction_criteria") or "")
    if isinstance(desc, list):
        desc = " ".join(str(x) for x in desc)
    desc = str(desc).strip()

    # Possible causes (list or string)
    causes = (rec.get("possible_causes") or rec.get("conditions")
              or rec.get("pak_vehicles") or [])
    if isinstance(causes, list):
        causes = "; ".join(str(c) for c in causes if c)
    causes = str(causes).strip()

    # Repair / diagnostic steps — field names differ by schema
    steps_raw = (rec.get("steps") or rec.get("diagnostic_procedure")
                 or rec.get("repair_steps") or [])
    steps_text = ""
    if isinstance(steps_raw, list):
        parts = []
        for s in steps_raw:
            if isinstance(s, dict):
                parts.append(s.get("instruction") or s.get("action") or s.get("step") or "")
            else:
                parts.append(str(s))
        steps_text = " | ".join(p for p in parts if p)
    else:
        steps_text = str(steps_raw)

    # System / subsystem tag
    system = (rec.get("system") or rec.get("category") or rec.get("fault_category") or "").strip()

    # Code decomposition
    prefix = code[0] if code[0].isalpha() else "X"
    num_part = re.sub(r"[^0-9]", "", code)
    code_number = int(num_part) if num_part else -1

    return {
        "dtc_code":       code,
        "description":    desc,
        "possible_causes": causes,
        "steps_text":     steps_text,
        "system":         system,
        "code_prefix":    prefix,
        "code_number":    code_number,
        "vehicle_make":   make,
        "vehicle_model":  model,
    }


# ── Per-schema extractors ──────────────────────────────────────────────────

def _extract_standard(raw: dict, make: str, model: str) -> list[dict]:
    """Schema A — standard vehicle files with top-level `dtcs` list."""
    records = []
    for item in raw.get("dtcs", []):
        if isinstance(item, dict):
            r = _normalise_record(item, make, model)
            if r:
                records.append(r)
    return records


def _extract_audi_combined(raw: dict) -> list[dict]:
    """Schema B — audi_all_dtc_combined.json: vehicles[].{p_codes,u_codes}.codes[]"""
    records = []
    for veh in raw.get("vehicles", []):
        # Try to get make/model from the vehicle string
        veh_str = veh.get("vehicle", "")
        veh_make = "Audi"
        veh_model = veh.get("model", veh_str.split()[2] if len(veh_str.split()) > 2 else "Unknown")

        for code_group in ("p_codes", "u_codes"):
            group = veh.get(code_group, {})
            for item in group.get("codes", []):
                if isinstance(item, dict):
                    r = _normalise_record(item, veh_make, veh_model)
                    if r:
                        records.append(r)
    return records


def _extract_audi_per_vehicle(raw: dict) -> list[dict]:
    """Schema C — audi_dtc_codes.json: vehicles[].dtc_codes[]"""
    records = []
    for veh in raw.get("vehicles", []):
        veh_str = veh.get("vehicle", "")
        parts = veh_str.split()
        veh_make = "Audi"
        veh_model = parts[2] if len(parts) > 2 else "Unknown"

        for item in veh.get("dtc_codes", []):
            if isinstance(item, dict):
                r = _normalise_record(item, veh_make, veh_model)
                if r:
                    records.append(r)
    return records


def _extract_pakistan_generic(raw: dict) -> list[dict]:
    """Schema D — pakistan_dtc_dataset.json: top-level dtc_codes[]"""
    records = []
    for item in raw.get("dtc_codes", []):
        if isinstance(item, dict):
            # pak_vehicles field tells us which vehicles use this code
            pak_vehs = item.get("pak_vehicles", "")
            # Use first vehicle as model context, rest go into causes
            make = "Generic"
            model = pak_vehs.split(",")[0].strip() if pak_vehs else "OBD-II Generic"
            r = _normalise_record(item, make, model)
            if r:
                records.append(r)
    return records


def _extract_pakistan_complete(raw: dict) -> list[dict]:
    """Schema E — pakistan_vehicles_complete_dtc.json: categories[].vehicles[].dtc_codes[]"""
    records = []
    for cat in raw.get("categories", []):
        for veh in cat.get("vehicles", []):
            veh_make  = veh.get("make",  "Unknown")
            veh_model = veh.get("model", "Unknown")
            for item in veh.get("dtc_codes", []):
                if isinstance(item, dict):
                    r = _normalise_record(item, veh_make, veh_model)
                    if r:
                        records.append(r)
    return records


def _extract_plain_list(raw: list, make: str, model: str) -> list[dict]:
    """Schema F — plain list of DTC records."""
    records = []
    for item in raw:
        if isinstance(item, dict):
            r = _normalise_record(item, make, model)
            if r:
                records.append(r)
    return records


# ── Schema detector ────────────────────────────────────────────────────────

def _detect_and_extract(raw, fname: str, make: str, model: str) -> list[dict]:
    """Route each file to the correct extractor based on its structure."""
    bname = os.path.basename(fname).lower()

    if bname == "audi_all_dtc_combined.json":
        return _extract_audi_combined(raw)

    if bname == "audi_dtc_codes.json":
        return _extract_audi_per_vehicle(raw)

    if bname == "pakistan_dtc_dataset.json":
        return _extract_pakistan_generic(raw)

    if bname == "pakistan_vehicles_complete_dtc.json":
        return _extract_pakistan_complete(raw)

    # Generic fallback: inspect top-level structure
    if isinstance(raw, list):
        return _extract_plain_list(raw, make, model)

    if isinstance(raw, dict):
        # Standard schema A
        if "dtcs" in raw:
            return _extract_standard(raw, make, model)

        # Schema C pattern (any file with top-level `vehicles` list)
        if "vehicles" in raw:
            records = []
            for veh in raw["vehicles"]:
                if not isinstance(veh, dict):
                    continue
                vm = veh.get("make", make)
                vmod = veh.get("model", model)
                for item in (veh.get("dtc_codes") or veh.get("dtcs") or []):
                    r = _normalise_record(item, vm, vmod)
                    if r:
                        records.append(r)
            return records

        # Schema D pattern (top-level `dtc_codes` list)
        if "dtc_codes" in raw:
            return _extract_pakistan_generic(raw)

    return []


# ── main loader ────────────────────────────────────────────────────────────

def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.json")))
    records = []
    file_stats = []

    for fpath in files:
        make, model = _parse_filename(fpath)
        try:
            with open(fpath, encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as e:
            print(f"[WARN] Could not read {os.path.basename(fpath)}: {e}")
            continue

        extracted = _detect_and_extract(raw, fpath, make, model)
        file_stats.append((os.path.basename(fpath), len(extracted)))
        records.extend(extracted)

    df = pd.DataFrame(records)

    # De-duplicate on (dtc_code, description) keeping first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=["dtc_code", "description"]).reset_index(drop=True)
    after = len(df)

    print(f"[data_loader] Loaded {before:,} raw records → {after:,} unique from {len(files)} files")
    print(f"  Code prefixes: { df['code_prefix'].value_counts().to_dict() }")

    # Show new-file contributions
    new_files = ["audi_all_dtc_combined.json", "audi_dtc_codes.json",
                 "pakistan_dtc_dataset.json", "pakistan_vehicles_complete_dtc.json"]
    for fname, count in file_stats:
        if fname in new_files:
            print(f"  {fname}: {count} records extracted")

    return df


if __name__ == "__main__":
    df = load_all()
    print(df.head(3).to_string())
