"""
session_inference.py — Session-level inference orchestrator.

Wraps the three model calls and applies post-inference rules that the
individual models cannot see because they process DTCs one-at-a-time:

  Rule 1 — Multi-code compound escalation
      ≥2 HIGH faults across ≥2 distinct safety-critical systems → VERY_CRITICAL
      ≥3 HIGH/VERY_CRITICAL faults total → VERY_CRITICAL
      (single-code models can't see the compound risk picture)

  Rule 2 — Sensor noise suppression for no-fault sessions
      If zero active DTCs AND >6 channels flagged MARGINAL/ABNORMAL,
      sensor anomaly scores are likely dominated by LSTM calibration noise.
      Scores are scaled down so a truly healthy vehicle isn't alarmed.

  Rule 3 — Predictive maintenance urgency
      If any sensor trend is "increasing" toward alert_hi and days_to_alert ≤ 7,
      bump urgency one step even without a DTC.

  Rule 4 — Safety-critical sensor + DTC cross-correlation
      Coolant ABNORMAL + any engine/ignition DTC → HIGH minimum on that DTC.
      Battery ABNORMAL + network/hybrid DTC → HIGH minimum.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from dtc_classifier  import predict_dtc
from sensor_anomaly  import predict_sensor_anomaly, SENSOR_PARAMS
from health_scorer   import predict_health_score

MODELS_DIR = Path(__file__).parent.parent / "models"

SEVERITY_LABELS   = ["LOW", "MEDIUM", "HIGH", "VERY_CRITICAL"]
URGENCY_MAP       = {0: "routine", 1: "within_week", 2: "within_week", 3: "immediate"}
USER_ACTION_MAP   = {
    "LOW":          "Monitor — no immediate action required",
    "MEDIUM":       "Schedule service within 2 weeks",
    "HIGH":         "Book workshop urgently within 3 days",
    "VERY_CRITICAL": "STOP DRIVING — contact mechanic immediately",
}

# Systems whose simultaneous failure constitutes a compound safety risk
_SAFETY_CRITICAL_SYSTEMS = {"engine", "abs_brakes", "airbag", "steering",
                             "transmission", "hybrid", "ignition"}

# ── Code-specific severity overrides (applied after ML classification) ─────────
# CAP: these codes must never exceed the given severity int (0=LOW … 3=VERY_CRITICAL).
# Use for codes whose typical presentation is less urgent than the keyword pattern
# or code-range rule alone would suggest.
_CODE_SEVERITY_CAP: dict[str, int] = {
    # Catalyst efficiency below threshold — converter degraded, not imminent failure
    "P0420": 1, "P0421": 1, "P0422": 1,
    "P0430": 1, "P0431": 1, "P0432": 1,
    # O2 / lambda sensor circuit faults — sensor degradation, schedule service
    "P0130": 1, "P0131": 1, "P0132": 1, "P0133": 1, "P0134": 1,
    "P0135": 1, "P0136": 1, "P0137": 1, "P0138": 1, "P0139": 1,
    "P0140": 1, "P0141": 1, "P0150": 1, "P0151": 1, "P0156": 1,
    # Clutch position / AGS actuator — mechanical wear, book service (not safety)
    "P0810": 1, "P0811": 1, "P0812": 1, "P0813": 1, "P0814": 1, "P0815": 1,
    # Toyota / Lexus OEM VVT system malfunction — performance issue, not immediate
    "P1349": 1, "P1346": 1, "P1348": 1,
}

# FLOOR: these codes must be at least the given severity int.
# Use for codes where the ML model under-ranks a genuine risk.
_CODE_SEVERITY_FLOOR: dict[str, int] = {
    # Camshaft/intake VVT slow response — timing chain / actuator failure risk
    "P000A": 2, "P000B": 2, "P000C": 2, "P000D": 2,
    # Camshaft position actuator circuit — VVT solenoid failure (engine damage risk)
    "P0010": 2, "P0011": 2, "P0012": 2, "P0013": 2, "P0014": 2, "P0015": 2,
    "P0020": 2, "P0021": 2, "P0022": 2, "P0023": 2, "P0024": 2, "P0025": 2,
}


# PID hard thresholds for cross-correlation (only alert when truly out of spec)
_PID_ALERT = {
    "coolant_temp":    {"hi": 105,  "lo": 55},
    "battery_voltage": {"hi": 15.0, "lo": 11.8},
    "fuel_trim_short": {"hi": 20,   "lo": -20},
    "fuel_trim_long":  {"hi": 18,   "lo": -18},
    "rpm":             {"hi": 4000, "lo": 400},
    "maf":             {"hi": 22,   "lo": 1.2},
    "o2_voltage":      {"hi": 0.95, "lo": 0.08},
}


def _pid_out_of_spec(ch: str, val: float) -> bool:
    """True only when raw PID value exceeds hard engineering thresholds."""
    spec = _PID_ALERT.get(ch)
    if spec is None:
        return False
    return val > spec["hi"] or val < spec["lo"]


def _apply_compound_escalation(dtc_results: list[dict]) -> list[dict]:
    """
    Rule 1: Escalate session severity when multiple safety-critical systems fail.
    Returns dtc_results with adjusted severity_int / severity_label where warranted.
    """
    if not dtc_results:
        return dtc_results

    high_or_vc = [r for r in dtc_results if r["severity_int"] >= 2]
    safety_systems_hit = {r["category_label"] for r in high_or_vc
                          if r["category_label"] in _SAFETY_CRITICAL_SYSTEMS}

    # Rule 1a: ≥2 distinct safety-critical systems with HIGH+ faults → VERY_CRITICAL
    # Rule 1b: ≥3 HIGH+ faults total → VERY_CRITICAL
    should_escalate = len(safety_systems_hit) >= 2 or len(high_or_vc) >= 3

    if should_escalate:
        for r in dtc_results:
            if r["severity_int"] >= 2:   # upgrade HIGH → VERY_CRITICAL
                r["severity_int"]   = 3
                r["severity_label"] = "VERY_CRITICAL"
                r["escalated"]      = True

    return dtc_results


def _apply_pid_overlay(sensor_results: dict, raw_pids: dict) -> dict:
    """
    Direct PID engineering check: if a raw reading is within normal spec,
    force status to NORMAL regardless of LSTM score. This prevents flat-snapshot
    false positives where the LSTM flags constant-value inputs as anomalous.

    Only overrides from ABNORMAL/MARGINAL → NORMAL.
    Never downgrades a genuine ABNORMAL reading that is also PID-confirmed.
    """
    for ch, v in sensor_results.items():
        raw_val = raw_pids.get(ch)
        if raw_val is None:
            continue
        p = SENSOR_PARAMS.get(ch, {})
        if not p:
            continue

        # Within ±1.5× std of mean → clearly normal; trust PID over LSTM
        in_normal_range = (p["min"] * 0.9 <= raw_val <= p["max"] * 1.1
                           and abs(raw_val - p["mean"]) <= 2.5 * p["std"])

        if in_normal_range and v["status"] != "NORMAL":
            v["status"]        = "NORMAL"
            v["anomaly_score"] = round(v["anomaly_score"] * 0.3, 4)
            v["days_to_alert"] = None

    return sensor_results


def _apply_sensor_suppression(sensor_results: dict, has_dtcs: bool) -> dict:
    """
    Rule 2: When no DTCs are active and most channels are flagged, the LSTM
    is likely seeing normal operating noise. Scale down anomaly scores so a
    healthy vehicle isn't needlessly alarmed.
    """
    if has_dtcs:
        return sensor_results   # DTCs present — sensor alerts are meaningful

    n_flagged = sum(1 for v in sensor_results.values() if v["status"] != "NORMAL")
    n_total   = len(sensor_results)

    # If >60% of channels flagged with no DTCs → likely calibration noise
    if n_flagged / n_total > 0.60:
        suppression = 0.55   # scale all anomaly scores down
        for ch, v in sensor_results.items():
            v["anomaly_score"] = round(v["anomaly_score"] * suppression, 4)
            # Re-evaluate status at new score
            if v["anomaly_score"] > 0.85:
                v["status"] = "ABNORMAL"
            elif v["anomaly_score"] > 0.50:
                v["status"] = "MARGINAL"
                v["days_to_alert"] = 14
            else:
                v["status"]        = "NORMAL"
                v["days_to_alert"] = None

    return sensor_results


def _apply_cross_correlation(dtc_results: list[dict],
                             sensor_results: dict,
                             raw_pids: dict) -> list[dict]:
    """
    Rule 4: Cross-correlate sensor readings with DTC systems to upgrade severity.

    Guards:
    - Only fires when the RAW PID value is outside hard engineering thresholds
      (not just when the LSTM marks it ABNORMAL — avoids false positives on flat
       snapshot sequences).
    - Does NOT upgrade a DTC that already explains the sensor reading
      (e.g. P0171 System Too Lean + high fuel trims = same root cause, no escalation).
    """
    # DTC codes that already explain elevated fuel trim readings (no escalation needed)
    _LEAN_RICH_CODES = {"P0171", "P0172", "P0174", "P0175", "P0087", "P0088"}
    # DTC codes that explain coolant / temp readings
    _COOLANT_CODES   = {"P0116", "P0117", "P0118", "P0119", "P0128", "P0217"}

    coolant_oos = _pid_out_of_spec("coolant_temp",    raw_pids.get("coolant_temp", 90))
    battery_oos = _pid_out_of_spec("battery_voltage", raw_pids.get("battery_voltage", 13.8))
    ft_oos      = (_pid_out_of_spec("fuel_trim_short", raw_pids.get("fuel_trim_short", 0))
                   and _pid_out_of_spec("fuel_trim_long", raw_pids.get("fuel_trim_long", 0)))

    dtc_codes_present = {r["dtc_code"] for r in dtc_results}

    for r in dtc_results:
        cat  = r.get("category_label", "")
        sev  = r["severity_int"]
        code = r["dtc_code"]

        # Coolant overtemp + engine/ignition DTC that is NOT a coolant code
        if (coolant_oos and cat in ("engine", "ignition", "hybrid")
                and code not in _COOLANT_CODES and sev < 2):
            r["severity_int"]   = 2
            r["severity_label"] = "HIGH"
            r["cross_corr"]     = f"coolant {raw_pids.get('coolant_temp')}°C + {cat} fault"

        # Low battery + network/hybrid DTC (battery_voltage only, not "engine")
        if (battery_oos and cat in ("network", "hybrid") and sev < 2):
            r["severity_int"]   = 2
            r["severity_label"] = "HIGH"
            r["cross_corr"]     = f"battery {raw_pids.get('battery_voltage')}V + {cat} fault"

        # High fuel trims + fuel/emissions DTC that is NOT the lean/rich code itself
        if (ft_oos and cat in ("fuel", "emissions")
                and code not in _LEAN_RICH_CODES and sev < 2):
            r["severity_int"]   = 2
            r["severity_label"] = "HIGH"
            r["cross_corr"]     = f"fuel trims {raw_pids.get('fuel_trim_short')}%/{raw_pids.get('fuel_trim_long')}% + {cat} fault"

    return dtc_results


def _compute_urgency(highest_sev_int: int, sensor_results: dict) -> str:
    """Rule 3: Escalate urgency if sensor trend is worsening fast."""
    base_urgency = URGENCY_MAP[highest_sev_int]

    # Any increasing-trend channel with days_to_alert ≤ 7 → bump urgency
    for v in sensor_results.values():
        if (v.get("trend") == "increasing"
                and v.get("days_to_alert") is not None
                and v["days_to_alert"] <= 7
                and base_urgency == "routine"):
            base_urgency = "within_week"
            break

    return base_urgency


def run_session(
    session: dict,
    save_dir: Path = MODELS_DIR,
) -> dict:
    """
    Full pipeline for one session.json dict.

    session keys used: session_id, vehicle, dtc_codes, sensors, location

    Returns the complete DriveIntel output JSON (matches output_schema.json).
    """
    vehicle    = session.get("vehicle", {})
    location   = session.get("location", {"lat": 0.0, "lng": 0.0})
    mileage_km = vehicle.get("mileage_km", 80_000)

    # ── 1. Build DTC records ──────────────────────────────────────────────────
    import re as _re
    raw_dtcs = session.get("dtc_codes", [])
    dtc_records = []
    for d in raw_dtcs:
        code   = (d.get("code") or d.get("dtc_code") or "").strip().upper()
        prefix = code[0] if code and code[0].isalpha() else "X"
        nums   = _re.sub(r"[^0-9]", "", code)
        dtc_records.append({
            "dtc_code":        code,
            "description":     d.get("description", ""),
            "possible_causes": d.get("possible_causes", ""),
            "code_prefix":     prefix,
            "code_number":     int(nums) if nums else -1,
        })

    # ── 2. DTC classification ─────────────────────────────────────────────────
    dtc_results = predict_dtc(dtc_records, save_dir=save_dir) if dtc_records else []

    # ── 2b. Apply code-specific severity caps / floors ────────────────────────
    for r in dtc_results:
        code  = r["dtc_code"]
        cap   = _CODE_SEVERITY_CAP.get(code)
        floor = _CODE_SEVERITY_FLOOR.get(code)
        if cap is not None and r["severity_int"] > cap:
            r["severity_int"]   = cap
            r["severity_label"] = SEVERITY_LABELS[cap]
        if floor is not None and r["severity_int"] < floor:
            r["severity_int"]   = floor
            r["severity_label"] = SEVERITY_LABELS[floor]

    # ── 3. Build sensor timeseries (30-reading window from snapshot if no history)
    sensors_snap = session.get("sensors", {})
    sensor_ts    = {}
    from sensor_anomaly import SEQ_LEN, SENSOR_CHANNELS
    import numpy as _np
    _rng = _np.random.default_rng(42)
    for ch in SENSOR_CHANNELS:
        snap = sensors_snap.get(ch, {})
        val  = snap.get("value", 0.0) if isinstance(snap, dict) else float(snap)
        # Repeat snapshot with realistic jitter for 30 readings
        jitter = abs(val) * 0.015 + 0.1
        readings = [{"value": float(val + _rng.normal(0, jitter))} for _ in range(SEQ_LEN)]
        sensor_ts[ch] = readings

    # ── 4. Sensor anomaly detection ───────────────────────────────────────────
    sensor_results = predict_sensor_anomaly(sensor_ts, save_dir=save_dir)

    # ── 5. Post-inference rules ───────────────────────────────────────────────
    # 5a. PID overlay — trust raw values over LSTM for within-spec readings
    raw_pids = {ch: sensors_snap.get(ch, {}).get("value", SENSOR_PARAMS.get(ch, {}).get("mean", 0))
                if isinstance(sensors_snap.get(ch, {}), dict)
                else float(sensors_snap.get(ch, 0))
                for ch in sensor_results}
    sensor_results = _apply_pid_overlay(sensor_results, raw_pids)
    # 5b. Suppression — if still >60% flagged with no DTCs, scale down (LSTM noise)
    sensor_results = _apply_sensor_suppression(sensor_results, has_dtcs=bool(dtc_results))
    # 5c. Cross-correlation — only fires when raw PIDs confirm out-of-spec
    dtc_results    = _apply_cross_correlation(dtc_results, sensor_results, raw_pids)
    # 5d. Compound escalation — multi-system high-severity combination
    dtc_results    = _apply_compound_escalation(dtc_results)

    # ── 6. Health score ───────────────────────────────────────────────────────
    health = predict_health_score(dtc_results, sensor_results,
                                  mileage_km=mileage_km, save_dir=save_dir)

    # ── 7. Session-level summary ──────────────────────────────────────────────
    highest_sev_int = max((r["severity_int"] for r in dtc_results), default=0)
    highest_sev     = SEVERITY_LABELS[highest_sev_int]
    fault_cats      = list({r["category_label"] for r in dtc_results})
    urgency         = _compute_urgency(highest_sev_int, sensor_results)

    days_to_alert = None
    for v in sensor_results.values():
        if v.get("days_to_alert") is not None:
            days_to_alert = min(days_to_alert or 999, v["days_to_alert"])

    # ── 8. Assemble output ────────────────────────────────────────────────────
    return {
        "session_id":          session.get("session_id", "unknown"),
        "model_version":       _load_bundle_id(save_dir),
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle":             vehicle,

        "dtc_analysis": [{
            "dtc_code":       r["dtc_code"],
            "description":    r.get("description", ""),
            "severity":       r["severity_int"],
            "severity_label": r["severity_label"],
            "severity_why":   r.get("cross_corr") or r.get("escalated") and "compound escalation" or "",
            "category":       r["category_label"],
            "confidence":     round(max(r["severity_probs"]), 4),
            "severity_probs": [round(p, 4) for p in r["severity_probs"]],
        } for r in dtc_results],

        "sensor_analysis": sensor_results,

        "health": {
            "score":         health["score"],
            "status":        health["status"],
            "days_to_alert": days_to_alert,
            "summary": (
                f"Vehicle health is {health['status'].lower()} "
                f"({health['score']:.0f}/100)."
                + (f" Predictive alert in {days_to_alert} day(s)." if days_to_alert else "")
            ),
        },

        "severity_summary": {
            "highest_severity": highest_sev,
            "active_dtcs":      len(dtc_results),
            "safety_systems":   sorted(fault_cats),
            "justification": (
                f"{len(dtc_results)} active fault(s) across {len(fault_cats)} system(s). "
                f"Highest individual severity: {highest_sev}."
            ),
            "user_action": USER_ACTION_MAP[highest_sev],
        },

        "workshop_query": {
            "lat":              location.get("lat", 0.0),
            "lng":              location.get("lng", 0.0),
            "radius_km":        15,
            "fault_categories": fault_cats,
            "severity_level":   highest_sev,
            "urgency":          urgency,
            "filters":          {"min_rating": 3.5, "open_now": True},
        },
    }


def _load_bundle_id(save_dir: Path) -> str:
    try:
        with open(save_dir / "model_meta.json") as f:
            return json.load(f).get("bundle_id", "unknown")
    except Exception:
        return "unknown"
