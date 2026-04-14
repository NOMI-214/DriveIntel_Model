"""
labeling.py — Rule-based severity and fault-category assignment for DTC records.

Severity: 0=LOW  1=MEDIUM  2=HIGH  3=VERY_CRITICAL
Category: engine | emissions | fuel | ignition | transmission | abs_brakes |
          suspension | steering | body | airbag | climate | lighting |
          security | network | hybrid | unknown
"""

import re
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Severity rules
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "VERY_CRITICAL"]
SEVERITY_TO_INT = {s: i for i, s in enumerate(SEVERITY_LABELS)}

# ── keyword patterns applied to description+causes text (case-insensitive) ──
_VERY_CRITICAL_KW = re.compile(
    r"airbag|air.bag|srs|inflat|restraint|seat.belt|pretension|"
    r"fire|overh[ae]at|thermal runaway|coolant.high|over.temp|"
    r"brake.fail|brake.fluid.low|no.brake|loss.of.brake|"
    r"steer.fail|power.steer.loss|"
    r"engine.shut.?down|engine.stall.*immedi|engine.damage|engine.seizure|"
    r"fuel.leak|fuel.fire|combustion.fail",
    re.IGNORECASE,
)

_HIGH_KW = re.compile(
    r"misfire|no.start|engine.stall|rough.idle|"
    r"catalyst.fail|catalyst.*efficiency.*low|"        # NOT bare "catalytic converter"
    r"abs.fail|abs.malfunction|brake.switch.fail|"
    r"transmission.fail|trans.slip|gear.fail|"
    r"loss.of.power|power.loss|no.power|"
    r"oil.pressure|low.oil|knock.sensor|"
    r"crankshaft.pos.*fail|camshaft.pos.*fail|"
    r"camshaft.adjustment.*fail|vvt.*fail|vvt.*stuck|"  # VVT/camshaft specific
    r"throttle.body.fail|throttle.stuck|"
    r"high.coolant|coolant.temp.high|overheat|"
    r"coolant.temp.*circuit|thermostat.stuck|"          # cooling system faults
    r"fuel.injector.fail|fuel.pressure.low|"
    r"hybrid.system.fail|battery.fail|main.relay",
    re.IGNORECASE,
)

_LOW_KW = re.compile(
    r"interior.light|door.mirror|window.motor|wiper.blade|"
    r"horn.circuit|acc.switch|ambient.light|courtesy.light|"
    r"radio|audio|infotainment|navigation",
    re.IGNORECASE,
)


# ── code-number range rules per prefix ───────────────────────────────────────

def _severity_from_P(num: int) -> int:
    """Return severity int for P-prefix DTC by SAE code range."""
    if 300 <= num <= 399:    # Misfires
        return 2  # HIGH (escalated to VERY_CRITICAL by keyword check later)
    if 600 <= num <= 699:    # ECM/PCM faults
        return 2
    if 700 <= num <= 899:    # Transmission / drive-train
        return 2
    if 200 <= num <= 299:    # Fuel injectors
        return 2
    if 100 <= num <= 199:    # MAF/MAP/IAT sensors
        return 1
    if 400 <= num <= 499:    # Emission control (EGR, EVAP, etc.)
        return 1
    if 420 <= num <= 430:    # Catalytic efficiency
        return 1
    if 500 <= num <= 599:    # Speed / idle control
        return 1
    if 900 <= num <= 999:    # Fuel/air metering
        return 1
    if 1000 <= num <= 1999:  # OEM-P1 (mixed)
        return 1
    if 2000 <= num <= 2999:  # P2 – extended emissions
        return 1
    if 3000 <= num <= 3999:  # P3 – powertrain combined
        return 2
    # A00+ hybrid/EV
    if num >= 0xA00:
        return 2
    return 1  # default MEDIUM


def _severity_from_B(num: int) -> int:
    if 0 <= num <= 199:      # Airbag / SRS
        return 3
    if 200 <= num <= 299:    # Seat belts / restraints
        return 3
    if 300 <= num <= 399:    # Body security / immobilizer
        return 1
    if 1000 <= num <= 1999:  # OEM body
        return 1
    return 0  # climate/lighting/accessories = LOW


def _severity_from_C(num: int) -> int:
    if 0 <= num <= 299:      # ABS / ESC / traction control
        return 2
    if 300 <= num <= 499:    # Suspension
        return 1
    if 500 <= num <= 699:    # Steering
        return 2
    if 1000 <= num <= 1999:  # OEM chassis
        return 1
    return 1


def _severity_from_U(num: int) -> int:
    if 0 <= num <= 99:       # Bus-off / master module loss (CAN, LIN)
        return 2
    if 100 <= num <= 299:    # Individual module loss
        return 1
    if 300 <= num <= 499:    # Data errors
        return 1
    return 0


_PREFIX_FN = {"P": _severity_from_P, "B": _severity_from_B,
              "C": _severity_from_C, "U": _severity_from_U}


def assign_severity(row: pd.Series) -> int:
    prefix = str(row["code_prefix"]).upper()
    num = int(row["code_number"]) if row["code_number"] >= 0 else 0
    text = f"{row['description']} {row['possible_causes']}".strip()

    # Start from code-range baseline
    fn = _PREFIX_FN.get(prefix)
    base = fn(num) if fn else 1

    # Keyword overrides (can only upgrade, not downgrade)
    if _VERY_CRITICAL_KW.search(text):
        base = max(base, 3)
    elif _HIGH_KW.search(text):
        base = max(base, 2)
    elif _LOW_KW.search(text):
        base = min(base, 0)

    # P0300-P0309: random/multiple cylinder misfire → VERY CRITICAL
    # P0310-P0399: single cylinder misfire → HIGH
    if prefix == "P" and 300 <= num <= 309:
        base = max(base, 3)
    elif prefix == "P" and 310 <= num <= 399:
        base = max(base, 2)

    # P0700-P0899 transmission (HIGH baseline), but P0800+ control system → MEDIUM
    if prefix == "P" and 800 <= num <= 899:
        base = min(base, 2)   # cap at HIGH, not VERY_CRITICAL

    # P0810 clutch position control → MEDIUM (mechanical wear, not immediate failure)
    if prefix == "P" and 810 <= num <= 819:
        base = min(base, 1)

    # P0116/P0117/P0118 coolant temp circuit → HIGH (thermostat fault = overheating risk)
    if prefix == "P" and 116 <= num <= 119:
        base = max(base, 2)

    return base


# ══════════════════════════════════════════════════════════════════════════════
# Fault-category rules
# ══════════════════════════════════════════════════════════════════════════════

FAULT_CATEGORIES = [
    "engine", "emissions", "fuel", "ignition", "transmission",
    "abs_brakes", "suspension", "steering", "body", "airbag",
    "climate", "lighting", "security", "network", "hybrid", "unknown",
]
CAT_TO_INT = {c: i for i, c in enumerate(FAULT_CATEGORIES)}


_CAT_KW = {
    "airbag":       re.compile(r"airbag|air.bag|srs|inflat|restraint|pretension|seat.belt", re.I),
    "abs_brakes":   re.compile(r"\babs\b|anti.lock|brake.switch|brake.press|brake.fluid|ebd|esc|traction.ctrl", re.I),
    "suspension":   re.compile(r"suspension|shock|strut|damper|ride.height|air.spring", re.I),
    "steering":     re.compile(r"steer|eps|power.steer|rack.and.pinion", re.I),
    "transmission": re.compile(r"transmission|trans\b|gear|clutch|torque.conv|shift|atf|cvt\b|dct\b", re.I),
    "hybrid":       re.compile(r"hybrid|hv.battery|inverter|motor.generator|mhev|phev|bev|ev.battery", re.I),
    "fuel":         re.compile(r"fuel.inject|fuel.press|fuel.pump|fuel.trim|injector|evap\b|canister|purge", re.I),
    "emissions":    re.compile(r"catalyst|catalytic|o2.sensor|oxygen.sensor|egr|lambda|nox|particulate|dpf\b|gpf\b", re.I),
    "ignition":     re.compile(r"ignition|coil|spark.plug|misfire|timing.adv|knock.sensor|crankshaft.pos|camshaft.pos", re.I),
    "engine":       re.compile(r"engine|coolant|oil.press|maf\b|map.sensor|throttle|idle|rpm|turbo|vvt|vtec|timing.chain", re.I),
    "climate":      re.compile(r"a/c|air.cond|hvac|blower|refrigerant|heater.core|defroster|fan.motor", re.I),
    "lighting":     re.compile(r"headlamp|tail.lamp|brake.light|turn.signal|fog.lamp|led.driver|daytime", re.I),
    "security":     re.compile(r"immobilizer|transponder|smart.key|keyless|theft|alarm|bcm.auth", re.I),
    "network":      re.compile(r"\bcan\b|lin.bus|most.bus|flexray|bus.off|network.comm|module.comm|lost.comm", re.I),
    "body":         re.compile(r"door|window|mirror|wiper|horn|seat|sunroof|moonroof|lock.actuator|body.ctrl", re.I),
}


def _cat_from_code(prefix: str, num: int) -> str:
    if prefix == "P":
        if 300 <= num <= 399: return "ignition"
        if 200 <= num <= 299: return "fuel"
        if 400 <= num <= 499: return "emissions"
        if 420 <= num <= 430: return "emissions"
        if 700 <= num <= 899: return "transmission"
        if num >= 0xA00:       return "hybrid"
        return "engine"
    if prefix == "B":
        if 0 <= num <= 299:   return "airbag"
        if 300 <= num <= 399: return "security"
        return "body"
    if prefix == "C":
        if 0 <= num <= 299:   return "abs_brakes"
        if 300 <= num <= 499: return "suspension"
        return "steering"
    if prefix == "U":
        return "network"
    return "unknown"


def assign_category(row: pd.Series) -> int:
    text = f"{row['description']} {row['possible_causes']} {row['system']}".strip()
    prefix = str(row["code_prefix"]).upper()
    num = int(row["code_number"]) if row["code_number"] >= 0 else 0

    # Keyword search over ordered categories (priority order)
    for cat in ["airbag", "abs_brakes", "hybrid", "transmission", "suspension",
                "steering", "fuel", "emissions", "ignition", "engine",
                "network", "security", "climate", "lighting", "body"]:
        if _CAT_KW[cat].search(text):
            return CAT_TO_INT[cat]

    # Fallback: code-range heuristic
    fallback = _cat_from_code(prefix, num)
    return CAT_TO_INT[fallback]


# ══════════════════════════════════════════════════════════════════════════════
# Apply to DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity"] = df.apply(assign_severity, axis=1)
    df["category"] = df.apply(assign_category, axis=1)
    df["severity_label"] = df["severity"].map(lambda x: SEVERITY_LABELS[x])
    df["category_label"] = df["category"].map(lambda x: FAULT_CATEGORIES[x])
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_all
    df = load_all()
    df = label_dataframe(df)
    print(df[["dtc_code", "severity_label", "category_label"]].value_counts(
        ["severity_label", "category_label"]
    ).to_string())
