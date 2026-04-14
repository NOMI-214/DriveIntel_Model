"""
train_pipeline.py — DriveIntel end-to-end training orchestrator.

Runs in order:
  1. Load + normalise all 169 DTC JSON files
  2. Rule-based severity + category labeling
  3. Train DTC Classifier   (PyTorch MLP, multi-task)
  4. Train Sensor Anomaly Detector (LSTM autoencoder)
  5. Train Health Scorer    (Shallow MLP regression)
  6. Export all to ONNX + INT8 quantization
  7. Write model_meta.json

Usage:
  python src/train_pipeline.py               # full run (default)
  python src/train_pipeline.py --fast        # reduced epochs/data for testing
  python src/train_pipeline.py --skip-sensor # skip LSTM (slow on CPU)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader      import load_all
from labeling         import label_dataframe, SEVERITY_LABELS, FAULT_CATEGORIES
from dtc_classifier   import train_dtc_classifier
from sensor_anomaly   import train_sensor_anomaly
from health_scorer    import train_health_scorer
from quantize_export  import quantize_all, write_model_meta


MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy drift check
# ══════════════════════════════════════════════════════════════════════════════

def drift_check(results: dict, baseline: dict | None, max_drop_pct: float = 2.0) -> bool:
    """
    Compare new training metrics against a saved baseline.
    Returns True if the new model passes (no significant regression).
    """
    if baseline is None:
        print("[drift_check] No baseline found — new model accepted as initial.")
        return True

    checks = [
        ("dtc_val_acc",   results.get("dtc_val_acc",   0), baseline.get("dtc_val_acc",   0)),
        ("sensor_auc",    results.get("sensor_auc",    0), baseline.get("sensor_auc",    0)),
        ("health_val_r2", results.get("health_val_r2", 0), baseline.get("health_val_r2", 0)),
    ]

    all_pass = True
    for name, new_val, old_val in checks:
        drop = (old_val - new_val) * 100  # positive = regression
        status = "PASS" if drop <= max_drop_pct else "FAIL"
        print(f"  [{status}] {name}: {old_val:.4f} → {new_val:.4f}  (Δ={-drop:+.2f}%)")
        if status == "FAIL":
            all_pass = False

    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace):
    t0 = time.time()
    print("=" * 70)
    print("  DriveIntel — Model Training Pipeline")
    print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mode    : {'FAST' if args.fast else 'FULL'}")
    print("=" * 70)

    # ── Load existing baseline metrics (for drift check) ─────────────────────
    meta_path = MODELS_DIR / "model_meta.json"
    baseline = None
    if meta_path.exists():
        with open(meta_path) as f:
            old_meta = json.load(f)
        baseline = old_meta.get("training_metrics")
        print(f"\n[Pipeline] Loaded baseline metrics from {meta_path.name}")

    # ── Step 1+2: Data loading + labeling ────────────────────────────────────
    print("\n[Step 1/6] Loading & labeling DTC data…")
    df = load_all(ROOT_DIR / "Data")
    df = label_dataframe(df)

    print(f"\n  Dataset summary:")
    print(f"    Total records  : {len(df):,}")
    print(f"    Severity dist  : { df['severity_label'].value_counts().to_dict() }")
    print(f"    Category dist  : { df['category_label'].value_counts().to_dict() }")

    training_results = {}

    # ── Step 3: DTC Classifier ───────────────────────────────────────────────
    print("\n[Step 2/6] Training DTC Classifier…")
    dtc_cfg = dict(
        epochs        = 15 if args.fast else 40,
        batch_size    = 256,
        lr            = 1e-3,
        hidden        = 512,
        tfidf_features= 1000 if args.fast else 3000,
        save_dir      = MODELS_DIR,
    )
    dtc_out = train_dtc_classifier(df, **dtc_cfg)
    training_results["dtc_val_acc"] = dtc_out["best_val_acc"]

    # ── Step 4: Sensor Anomaly Detector ──────────────────────────────────────
    if not args.skip_sensor:
        print("\n[Step 3/6] Training Sensor Anomaly Detector…")
        sensor_cfg = dict(
            n_normal    = 2000  if args.fast else 8000,
            n_anomalous = 1000  if args.fast else 4000,
            epochs      = 10    if args.fast else 30,
            batch_size  = 64,
            latent_dim  = 64,
            lr          = 1e-3,
            save_dir    = MODELS_DIR,
        )
        sensor_out = train_sensor_anomaly(**sensor_cfg)
        training_results["sensor_auc"] = sensor_out["auc"]
    else:
        print("\n[Step 3/6] Sensor Anomaly Detector — SKIPPED")
        training_results["sensor_auc"] = baseline.get("sensor_auc", 0) if baseline else 0

    # ── Step 5: Health Scorer ─────────────────────────────────────────────────
    print("\n[Step 4/6] Training Health Scorer…")
    health_cfg = dict(
        n_sessions = 10000 if args.fast else 50000,
        epochs     = 20    if args.fast else 60,
        batch_size = 512,
        hidden     = 128,
        lr         = 1e-3,
        save_dir   = MODELS_DIR,
    )
    health_out = train_health_scorer(**health_cfg)
    training_results["health_val_mae"] = health_out["val_mae"]
    training_results["health_val_r2"]  = health_out["val_r2"]

    # ── Step 6: Accuracy drift check ─────────────────────────────────────────
    print("\n[Step 5/6] Accuracy drift check…")
    passed = drift_check(training_results, baseline)
    if not passed:
        print("\n  [WARNING] Drift check FAILED — new bundle NOT promoted.")
        print("  Investigate training data quality before redeploying.")
    else:
        print("  Drift check PASSED — proceeding to export.")

    # ── Step 7: ONNX export + INT8 quantization ───────────────────────────────
    print("\n[Step 6/6] ONNX export + INT8 quantization…")
    meta_models = quantize_all(MODELS_DIR)

    # Bundle size check
    total_mb = sum(m["size_mb"] for m in meta_models.values())
    print(f"\n  Total INT8 bundle size: {total_mb:.3f} MB  (target ≤ 5 MB)")
    if total_mb > 5.0:
        print("  [WARNING] Bundle exceeds 5 MB target — consider pruning or further quantization.")

    # ── Write model_meta.json ─────────────────────────────────────────────────
    write_model_meta(
        meta_models        = meta_models,
        models_dir         = MODELS_DIR,
        training_results   = {
            **training_results,
            "drift_check_passed": passed,
            "total_bundle_mb":    round(total_mb, 3),
            "training_records":   len(df),
        },
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"\n  Model artefacts in: {MODELS_DIR}")
    for f in sorted(MODELS_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<40} {size:7.1f} KB")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Inference demo (end-to-end session JSON → output JSON)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference_demo():
    """
    Demonstrate the full inference pipeline on a synthetic session.json.
    """
    import re
    import pandas as pd
    import numpy as np
    from dtc_classifier  import predict_dtc
    from sensor_anomaly  import predict_sensor_anomaly, SENSOR_PARAMS
    from health_scorer   import predict_health_score

    print("\n" + "=" * 70)
    print("  DriveIntel — Inference Demo")
    print("=" * 70)

    # Fake session
    session = {
        "session_id": "demo-session-001",
        "vehicle": {"make": "Toyota", "model": "Corolla", "year": 2018,
                    "engine": "1.8L 4-cyl", "mileage_km": 87400},
        "dtc_codes": [
            {"code": "P0420", "status": "confirmed"},
            {"code": "P0300", "status": "confirmed"},
        ],
        "sensors": {
            "rpm":             {"value": 850},
            "coolant_temp":    {"value": 105},   # elevated
            "o2_voltage":      {"value": 0.45},
            "maf":             {"value": 4.2},
            "fuel_trim_short": {"value": 18.5},  # abnormal
            "fuel_trim_long":  {"value": 12.2},  # abnormal
            "throttle_pos":    {"value": 14},
            "battery_voltage": {"value": 12.1},  # low
            "intake_air_temp": {"value": 28},
            "vehicle_speed":   {"value": 0},
        },
        "location": {"lat": 33.6844, "lng": 73.0479},
    }

    # ── Build DTC records for classifier ─────────────────────────────────────
    # Use labeling rules to get prefix/number for the codes
    import re as _re
    dtc_records = []
    for dtc in session["dtc_codes"]:
        code = dtc["code"].upper()
        prefix = code[0]
        num_s = _re.sub(r"[^0-9]", "", code)
        dtc_records.append({
            "dtc_code":       code,
            "description":    "",
            "possible_causes":"",
            "code_prefix":    prefix,
            "code_number":    int(num_s) if num_s else -1,
        })

    print("\n[1] DTC Classification…")
    dtc_results = predict_dtc(dtc_records, save_dir=MODELS_DIR)
    for r in dtc_results:
        print(f"  {r['dtc_code']} → severity={r['severity_label']}  "
              f"category={r['category_label']}  "
              f"conf={max(r['severity_probs']):.3f}")

    # ── Build fake timeseries from current sensor snapshot ────────────────────
    print("\n[2] Sensor Anomaly Detection…")
    sensor_ts = {}
    for ch, reading in session["sensors"].items():
        val = reading["value"]
        # Pad to SEQ_LEN with slight noise
        from sensor_anomaly import SEQ_LEN
        import numpy as np
        ts = [{"value": val + np.random.normal(0, abs(val) * 0.02)} for _ in range(SEQ_LEN)]
        sensor_ts[ch] = ts

    sensor_results = predict_sensor_anomaly(sensor_ts, save_dir=MODELS_DIR)
    for ch, res in sensor_results.items():
        if res["status"] != "NORMAL":
            print(f"  {ch:<20} {res['status']:<10} score={res['anomaly_score']:.3f}  "
                  f"days_to_alert={res['days_to_alert']}")

    # ── Health score ──────────────────────────────────────────────────────────
    print("\n[3] Health Score…")
    health = predict_health_score(
        dtc_results, sensor_results,
        mileage_km=session["vehicle"]["mileage_km"],
        save_dir=MODELS_DIR,
    )
    print(f"  Score: {health['score']} / 100  ({health['status']})")

    # ── Assemble output JSON ──────────────────────────────────────────────────
    highest_sev_int = max(r["severity_int"] for r in dtc_results) if dtc_results else 0
    sev_labels = ["LOW", "MEDIUM", "HIGH", "VERY_CRITICAL"]
    urgency_map = {0: "routine", 1: "within_week", 2: "within_week", 3: "immediate"}
    user_action_map = {
        "LOW": "Monitor", "MEDIUM": "Schedule service",
        "HIGH": "Book urgently", "VERY_CRITICAL": "Stop driving, call now",
    }
    fault_cats = list({r["category_label"] for r in dtc_results})
    highest_sev_label = sev_labels[highest_sev_int]

    output = {
        "session_id":  session["session_id"],
        "model_version": _load_version(),
        "inference_timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "dtc_analysis": [
            {
                "dtc_code":      r["dtc_code"],
                "description":   r.get("description", ""),
                "severity":      r["severity_int"],
                "severity_label":r["severity_label"],
                "category":      r["category_label"],
                "confidence":    round(max(r["severity_probs"]), 4),
                "severity_probs":r["severity_probs"],
            }
            for r in dtc_results
        ],
        "sensor_analysis": sensor_results,
        "health": {
            "score":   health["score"],
            "status":  health["status"],
            "summary": f"Vehicle health is {health['status'].lower()} at {health['score']}/100.",
        },
        "severity_summary": {
            "highest_severity": highest_sev_label,
            "justification":    f"Highest active fault severity across {len(dtc_results)} DTC(s).",
            "user_action":      user_action_map[highest_sev_label],
        },
        "workshop_query": {
            "lat":              session["location"]["lat"],
            "lng":              session["location"]["lng"],
            "radius_km":        15,
            "fault_categories": fault_cats,
            "severity_level":   highest_sev_label,
            "urgency":          urgency_map[highest_sev_int],
            "filters":          {"min_rating": 3.5, "open_now": True},
        },
    }

    out_path = MODELS_DIR / "demo_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full output written → {out_path.name}")
    print(json.dumps(output, indent=2))


def _load_version() -> str:
    meta_path = MODELS_DIR / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f).get("bundle_id", "unknown")
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriveIntel training pipeline")
    parser.add_argument("--fast",        action="store_true", help="Reduced epochs for quick test")
    parser.add_argument("--skip-sensor", action="store_true", help="Skip LSTM sensor model")
    parser.add_argument("--demo",        action="store_true", help="Run inference demo after training")
    args = parser.parse_args()

    run(args)

    if args.demo:
        run_inference_demo()
