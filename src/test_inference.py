"""
test_inference.py — DriveIntel comprehensive inference test suite (v2).

Uses session_inference.run_session() which applies all 4 post-inference rules:
  • Compound escalation  • Sensor suppression  • Cross-correlation  • Urgency boost

All 13 test cases + deployment readiness summary.
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from session_inference import run_session
from labeling import SEVERITY_LABELS

MODELS_DIR = Path(__file__).parent.parent / "models"

# ══════════════════════════════════════════════════════════════════════════════
# Test case builder
# ══════════════════════════════════════════════════════════════════════════════

def _session(session_id, make, model, year, engine, mileage_km,
             dtcs, sensors, location=None, expected_severity=None):
    """Build a minimal session dict and run full inference."""
    sess = {
        "session_id": session_id,
        "vehicle": {"make": make, "model": model, "year": year,
                    "engine": engine, "mileage_km": mileage_km},
        "dtc_codes": [{"code": c, "description": d, "possible_causes": p}
                      for c, d, p in dtcs],
        "sensors": {ch: {"value": v} for ch, v in sensors.items()},
        "location": location or {"lat": 33.6844, "lng": 73.0479},
    }
    t0 = time.perf_counter()
    out = run_session(sess)
    out["_elapsed_ms"]         = round((time.perf_counter() - t0) * 1000, 1)
    out["_expected_severity"]  = expected_severity
    out["_severity_match"]     = (expected_severity is None or
                                  out["severity_summary"]["highest_severity"] == expected_severity)
    return out


# Normal PID readings (baseline idle values)
_NORMAL = {
    "rpm":             800,
    "coolant_temp":    90,
    "o2_voltage":      0.45,
    "maf":             4.5,
    "fuel_trim_short": 1.5,
    "fuel_trim_long":  1.0,
    "throttle_pos":    14,
    "battery_voltage": 13.8,
    "intake_air_temp": 28,
    "vehicle_speed":   0,
}


def _pids(**overrides):
    """Return _NORMAL sensors with overridden values."""
    s = dict(_NORMAL)
    s.update(overrides)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 13 Test cases
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    results = []

    # ── 1. Generic SAE — catalyst + O2 sensor ────────────────────────────────
    results.append(_session(
        "T01", "Honda", "Civic 1.5T", 2016, "1.5L VTEC Turbo", 92_000,
        dtcs=[
            ("P0420", "Catalyst System Efficiency Below Threshold Bank 1",
             "Catalytic converter degraded; O2 sensor out of range"),
            ("P0136", "O2 Sensor Circuit Malfunction Bank 1 Sensor 2",
             "Oxygen sensor heater; wiring; ECM"),
        ],
        sensors=_pids(o2_voltage=0.12, fuel_trim_short=18.5, fuel_trim_long=14.0),
        expected_severity="MEDIUM",
    ))

    # ── 2. Generic SAE — random misfire + overheating ─────────────────────────
    results.append(_session(
        "T02", "Toyota", "Corolla", 2018, "1.8L 2ZR-FE", 87_400,
        dtcs=[
            ("P0300", "Random/Multiple Cylinder Misfire Detected",
             "Worn spark plugs; injector fault; low compression"),
            ("P0301", "Cylinder 1 Misfire Detected",
             "Spark plug; ignition coil; injector"),
        ],
        sensors=_pids(coolant_temp=112, rpm=550, maf=2.1),
        expected_severity="VERY_CRITICAL",
    ))

    # ── 3. OEM Toyota P1 — VVT system ────────────────────────────────────────
    results.append(_session(
        "T03", "Toyota", "RAV4", 2020, "2.5L A25A-FKS", 45_000,
        dtcs=[
            ("P1349", "Variable Valve Timing System Malfunction Bank 1",
             "VVT oil control valve sticking; low oil pressure; sludge"),
        ],
        sensors=_pids(rpm=720),
        expected_severity="MEDIUM",
    ))

    # ── 4. OEM Audi P0 — camshaft timing + MAF ───────────────────────────────
    results.append(_session(
        "T04", "Audi", "A4 B9", 2019, "2.0L TFSI EA888", 63_000,
        dtcs=[
            ("P000A", "Intake Camshaft Position Slow Response Bank 1",
             "VVT oil control valve; camshaft sensor; timing chain stretch"),
            ("P0101", "Mass Air Flow Sensor Range/Performance",
             "Dirty MAF sensor; air leak; vacuum hose"),
        ],
        sensors=_pids(maf=1.8, fuel_trim_short=19.0, intake_air_temp=52),
        expected_severity="HIGH",
    ))

    # ── 5. Body — airbag/SRS ──────────────────────────────────────────────────
    results.append(_session(
        "T05", "Hyundai", "Elantra", 2017, "2.0L Nu MPI", 78_000,
        dtcs=[
            ("B0100", "Driver Airbag Circuit Open",
             "Airbag module; clock spring; harness"),
            ("B0102", "Passenger Airbag Circuit Short to Ground",
             "SRS module; connector corrosion"),
        ],
        sensors=_pids(),   # sensors nominal — fault is electrical only
        expected_severity="VERY_CRITICAL",
    ))

    # ── 6. Chassis — ABS wheel speed sensors ──────────────────────────────────
    results.append(_session(
        "T06", "KIA", "Sportage", 2021, "2.0L MPI Theta II", 38_000,
        dtcs=[
            ("C0035", "Right Front Wheel Speed Sensor Circuit",
             "Wheel speed sensor; tone ring; wiring; bearing"),
            ("C0040", "Right Rear Wheel Speed Sensor Circuit",
             "Wheel speed sensor; reluctor ring damaged"),
        ],
        sensors=_pids(vehicle_speed=45),
        expected_severity="HIGH",
    ))

    # ── 7. Network — CAN bus loss + low battery ───────────────────────────────
    results.append(_session(
        "T07", "BMW", "328i", 2014, "2.0L N20B20 Turbo", 110_000,
        dtcs=[
            ("U0100", "Lost Communication With ECM/PCM",
             "CAN bus wiring; ECM power supply; ground fault"),
            ("U0073", "Control Module Communication Bus Off",
             "CAN bus terminal resistance; module failure"),
        ],
        sensors=_pids(battery_voltage=11.4),
        expected_severity="HIGH",
    ))

    # ── 8. Pakistan market — Suzuki Alto AGS + dusty MAF ─────────────────────
    results.append(_session(
        "T08", "Suzuki", "Alto AGS", 2022, "K10C 0.66L VVT", 25_000,
        dtcs=[
            ("P0810", "Clutch Position Control Error",
             "AGS clutch actuator; low clutch fluid; city traffic wear"),
            ("P0101", "MAF Sensor Range/Performance",
             "Dusty MAF — common in Karachi/Lahore; air filter clogged"),
        ],
        sensors=_pids(maf=1.6, fuel_trim_short=12.0),
        expected_severity="MEDIUM",
    ))

    # ── 9. Predictive — battery drain, zero DTCs ──────────────────────────────
    results.append(_session(
        "T09", "Toyota", "Prius PHV", 2019, "1.8L 2ZR-FXE Hybrid", 105_000,
        dtcs=[],
        sensors=_pids(battery_voltage=11.8, coolant_temp=97),
        expected_severity="LOW",
    ))

    # ── 10. Predictive — LTFT creeping lean (P0171 early stage) ───────────────
    results.append(_session(
        "T10", "Mitsubishi", "Lancer", 2015, "2.0L 4B11", 130_000,
        dtcs=[
            ("P0171", "System Too Lean Bank 1",
             "Vacuum leak; dirty MAF; fuel pressure low; O2 sensor bias"),
        ],
        sensors=_pids(fuel_trim_short=21.0, fuel_trim_long=18.0,
                      maf=2.3, o2_voltage=0.10),
        expected_severity="MEDIUM",
    ))

    # ── 11. Predictive — cooling system degrading ─────────────────────────────
    results.append(_session(
        "T11", "Lexus", "LX570", 2016, "5.7L 3UR-FE V8", 175_000,
        dtcs=[
            ("P0116", "Engine Coolant Temperature Circuit Range/Performance",
             "Coolant temp sensor; thermostat stuck open; low coolant level"),
        ],
        sensors=_pids(coolant_temp=108, maf=12.0),
        expected_severity="HIGH",
    ))

    # ── 12. Multi-code critical — compound escalation test ────────────────────
    results.append(_session(
        "T12", "Toyota", "Land Cruiser", 2015, "4.5L 1VD-FTV V8 Diesel", 210_000,
        dtcs=[
            ("P0700", "Transmission Control System Malfunction",
             "TCM fault; solenoid failure; ATF contamination"),
            ("P0087", "Fuel Rail/System Pressure Too Low",
             "Fuel pump wear; clogged filter; pressure regulator"),
            ("P0562", "System Voltage Low",
             "Weak battery; alternator failing; parasitic drain"),
            ("C0200", "ABS Right Front Wheel Speed Sensor Circuit Failure",
             "Sensor; tone ring; wiring corroded"),
        ],
        sensors=_pids(battery_voltage=11.2, fuel_trim_short=22.0,
                      maf=3.5, coolant_temp=106, rpm=650),
        expected_severity="VERY_CRITICAL",
    ))

    # ── 13. Healthy baseline — zero faults, all PIDs nominal ─────────────────
    results.append(_session(
        "T13", "Toyota", "Yaris", 2023, "1.5L 3NR-VE", 8_000,
        dtcs=[],
        sensors=_pids(),   # all factory-nominal values
        expected_severity="LOW",
    ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Printer
# ══════════════════════════════════════════════════════════════════════════════

def _print(r: dict):
    SEP  = "─" * 72
    ICON = {"LOW": "✅", "MEDIUM": "🟡", "HIGH": "🟠", "VERY_CRITICAL": "🔴"}
    ss   = r["severity_summary"]
    h    = r["health"]
    veh  = r["vehicle"]
    sev  = ss["highest_severity"]
    exp  = r["_expected_severity"]
    match_tag = "" if r["_severity_match"] else f"  ⚠ EXPECTED {exp}"

    print(f"\n{'═'*72}")
    print(f"  {r.get('_name', r['session_id'])}")
    print(f"  {veh.get('year','')} {veh.get('make','')} {veh.get('model','')}  "
          f"| {veh.get('engine','')}  | {veh.get('mileage_km',0):,} km")
    print(f"{'═'*72}")

    # DTC table
    dtcs = r.get("dtc_analysis", [])
    if dtcs:
        print(f"  {'CODE':<8} {'SEVERITY':<14} {'CATEGORY':<14} {'CONF':>6}  {'NOTE'}")
        print(f"  {SEP}")
        for d in dtcs:
            note = d.get("severity_why", "")[:30]
            print(f"  {d['dtc_code']:<8} {d['severity_label']:<14} {d['category']:<14} "
                  f"{d['confidence']:>5.1%}  {note}")
    else:
        print("  No active DTC codes")

    # Sensor summary — only non-NORMAL channels
    sa = r.get("sensor_analysis", {})
    flagged = {ch: v for ch, v in sa.items() if v["status"] != "NORMAL"}
    print(f"\n  {SEP}")
    print(f"  SENSORS  {len(flagged)}/{len(sa)} flagged", end="")
    if h.get("days_to_alert"):
        print(f"  |  ⏰ Alert in {h['days_to_alert']} day(s)", end="")
    print()
    for ch, v in flagged.items():
        icon  = "⚠" if v["status"] == "ABNORMAL" else "△"
        trend = v.get("trend", "stable")
        days  = f"→ {v['days_to_alert']}d" if v.get("days_to_alert") else ""
        print(f"    {icon} {ch:<22} {v['status']:<10} "
              f"score={v['anomaly_score']:.3f}  trend={trend}  {days}")

    # Health bar + verdict
    bar_filled = int(h["score"] / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"\n  {SEP}")
    print(f"  HEALTH  [{bar}]  {h['score']:5.1f}/100  ({h['status']})")
    print(f"  {ICON.get(sev,'❓')} SEVERITY : {sev}{match_tag}")
    print(f"  ACTION   : {ss['user_action']}")
    print(f"  URGENCY  : {r['workshop_query']['urgency'].upper()}")
    print(f"  ⚡ {r['_elapsed_ms']} ms")


def print_summary(results: list[dict]):
    SEP = "─" * 72
    print(f"\n\n{'═'*72}")
    print("  DEPLOYMENT READINESS — SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'#':<4} {'Test':<44} {'SEV':<15} {'HEALTH':>6}  {'TIME':>7}  RESULT")
    print(f"  {SEP}")
    passed = 0
    for i, r in enumerate(results, 1):
        sev    = r["severity_summary"]["highest_severity"]
        health = r["health"]["score"]
        ms     = r["_elapsed_ms"]
        match  = r["_severity_match"]
        if match:
            passed += 1
        ok_tag = "✓" if match else f"✗ (exp {r['_expected_severity']})"
        name   = r.get("_name", r["session_id"])[:44]
        print(f"  {i:<4} {name:<44} {sev:<15} {health:>5.1f}  {ms:>6.0f}ms  {ok_tag}")

    avg_ms = sum(r["_elapsed_ms"] for r in results) / len(results)
    print(f"  {SEP}")
    print(f"  PASS: {passed}/{len(results)}  ({passed/len(results)*100:.0f}%)  |  "
          f"Avg inference: {avg_ms:.0f}ms")
    print(f"{'═'*72}")

    # Deployment verdict
    print()
    blockers = [r for r in results if not r["_severity_match"]]
    healthy  = next((r for r in results if r["session_id"] == "T13"), None)

    print("  DEPLOYMENT VERDICT")
    print(f"  {'─'*40}")
    items = [
        ("Severity accuracy (test suite)",
         f"{passed}/{len(results)} ({passed/len(results)*100:.0f}%)",
         passed / len(results) >= 0.85),
        ("Avg inference latency",
         f"{avg_ms:.0f}ms",
         avg_ms < 100),
        ("Healthy baseline health score",
         f"{healthy['health']['score']:.0f}/100" if healthy else "N/A",
         healthy and healthy["health"]["score"] >= 60),
        ("Healthy baseline sensors flagged",
         f"{len([v for v in healthy['sensor_analysis'].values() if v['status']!='NORMAL'])}/10" if healthy else "N/A",
         healthy and len([v for v in healthy["sensor_analysis"].values() if v["status"] != "NORMAL"]) <= 3),
        ("Critical faults correctly flagged",
         "VERY_CRITICAL on misfire/airbag/compound",
         all(r["severity_summary"]["highest_severity"] == "VERY_CRITICAL"
             for r in results if r["session_id"] in ("T02", "T05", "T12"))),
    ]
    all_green = True
    for label, val, ok in items:
        icon = "✅" if ok else "❌"
        if not ok:
            all_green = False
        print(f"  {icon}  {label:<40} {val}")

    print()
    if all_green:
        print("  ✅  READY FOR DEPLOYMENT")
    else:
        print("  ❌  NOT YET DEPLOYMENT-READY — resolve ❌ items above")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     DriveIntel — Deployment Readiness Test Suite (v2)              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = run_all()

    # Attach display names
    names = [
        "Generic SAE: P0420 Catalyst + O2 (Honda Civic)",
        "Generic SAE: P0300 Random Misfire + Overheating",
        "OEM Toyota P1349: VVT Oil Control Valve",
        "OEM Audi P000A: Camshaft Timing + MAF Sensor",
        "Body B0100/B0102: Airbag/SRS Open Circuit",
        "Chassis C0035/C0040: ABS Wheel Speed Sensors",
        "Network U0100/U0073: CAN Bus Loss + Low Battery",
        "Pakistan: Suzuki Alto AGS Clutch + Dusty MAF",
        "Predictive: Battery Drain (no DTCs)",
        "Predictive: Fuel System LTFT Creeping Lean",
        "Predictive: Cooling System Degrading (P0116)",
        "Multi-Code CRITICAL: Trans + Engine + ABS Storm",
        "Healthy Baseline: No faults, all PIDs nominal",
    ]
    for r, name in zip(results, names):
        r["_name"] = name

    for r in results:
        _print(r)

    print_summary(results)

    # Save
    out_path = MODELS_DIR / "test_results.json"
    with open(out_path, "w") as f:
        json.dump([{
            "id":               r["session_id"],
            "name":             r.get("_name", ""),
            "vehicle":          r["vehicle"],
            "highest_severity": r["severity_summary"]["highest_severity"],
            "expected":         r["_expected_severity"],
            "match":            r["_severity_match"],
            "health_score":     r["health"]["score"],
            "health_status":    r["health"]["status"],
            "days_to_alert":    r["health"].get("days_to_alert"),
            "urgency":          r["workshop_query"]["urgency"],
            "inference_ms":     r["_elapsed_ms"],
            "dtcs": [{
                "code":       d["dtc_code"],
                "severity":   d["severity_label"],
                "category":   d["category"],
                "confidence": d["confidence"],
                "note":       d.get("severity_why", ""),
            } for d in r.get("dtc_analysis", [])],
            "flagged_sensors": [ch for ch, v in r.get("sensor_analysis", {}).items()
                                if v["status"] != "NORMAL"],
        } for r in results], f, indent=2)
    print(f"  Results saved → {out_path.name}")
