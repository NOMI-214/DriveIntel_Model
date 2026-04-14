"""
health_scorer.py — Shallow MLP: aggregated features → 0–100 health score.

Input features (22-dim):
  - highest_severity        : int 0–3
  - n_dtcs                  : total active fault count (clipped 0–20)
  - n_very_critical         : count of VERY CRITICAL faults
  - n_high                  : count of HIGH faults
  - n_medium                : count of MEDIUM faults
  - n_sensor_abnormal       : count of ABNORMAL sensors
  - n_sensor_marginal       : count of MARGINAL sensors
  - max_anomaly_score       : max per-channel anomaly score across all sensors
  - mean_anomaly_score      : mean anomaly score
  - battery_voltage_norm    : normalised battery voltage (0–1)
  - coolant_temp_norm       : normalised coolant temp
  - fuel_trim_norm          : normalised |fuel_trim_short| + |fuel_trim_long|
  - maf_norm                : normalised MAF reading
  - rpm_norm                : normalised RPM
  - o2_norm                 : normalised O2 voltage
  - n_categories_affected   : distinct fault categories (0–16)
  - has_misfire             : bool (0/1)
  - has_catalyst_fault      : bool
  - has_abs_fault           : bool
  - has_airbag_fault        : bool
  - has_network_fault       : bool
  - mileage_band            : 0=<50k  1=50-100k  2=100-150k  3=>150k km

Training is done on synthetic sessions generated from the rule-labelled DTC
data; the ground-truth score is computed analytically so the MLP learns to
reproduce it from the feature vector (and generalises to unseen combinations).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_FEATURES = 22
RNG = np.random.default_rng(0)


# ══════════════════════════════════════════════════════════════════════════════
# Analytic score formula (ground truth for training)
# ══════════════════════════════════════════════════════════════════════════════

def compute_health_score(features: np.ndarray) -> float:
    """
    Deterministic health score from raw feature vector.
    Returns float in [0, 100].
    """
    (highest_sev, n_dtcs, n_vc, n_hi, n_med, n_ab, n_marg,
     max_anom, mean_anom, bat_norm, cool_norm, ft_norm, maf_norm,
     rpm_norm, o2_norm, n_cats, has_misfire, has_cat, has_abs,
     has_air, has_net, mileage_band) = features

    score = 100.0

    # Severity deductions (capped so a single very-critical fault ≠ instant 0)
    sev_deduct = n_vc * 20.0 + n_hi * 10.0 + n_med * 4.0
    sev_deduct += (n_dtcs - n_vc - n_hi - n_med) * 1.5   # LOW faults
    score -= min(sev_deduct, 60.0)   # cap at -60 from DTCs alone

    # Sensor deductions — cap so 10 ABNORMAL sensors ≠ -80
    # If ALL 10 channels flagged with no DTCs it is calibration noise — weight halved
    n_total_sensors = 10
    calibration_noise = (n_dtcs == 0 and (n_ab + n_marg) >= 8)
    sensor_weight = 0.4 if calibration_noise else 1.0
    sensor_deduct = (n_ab * 4.0 + n_marg * 1.5 + max_anom * 10.0 + mean_anom * 4.0) * sensor_weight
    score -= min(sensor_deduct, 35.0)   # cap at -35 from sensors

    # Specific subsystem deductions
    score -= has_misfire * 8.0
    score -= has_cat     * 5.0
    score -= has_abs     * 4.0
    score -= has_air     * 15.0   # airbag = significant safety risk
    score -= has_net     * 4.0

    # Out-of-range sensor penalties (capped individually)
    score -= min(abs(bat_norm  - 0.5) * 8.0,  8.0)
    score -= min(abs(cool_norm - 0.5) * 6.0,  6.0)
    score -= min(ft_norm * 10.0,              10.0)

    # Mileage ageing discount
    score -= mileage_band * 2.0

    # Category breadth penalty
    score -= max(0, n_cats - 3) * 1.5

    return float(np.clip(score, 0.0, 100.0))


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic session generator
# ══════════════════════════════════════════════════════════════════════════════

def _random_session() -> tuple[np.ndarray, float]:
    """Return (feature_vector, health_score)."""
    highest_sev    = RNG.integers(0, 4)
    n_dtcs         = RNG.integers(0, 15)
    n_vc           = RNG.integers(0, min(n_dtcs + 1, 4))
    n_hi           = RNG.integers(0, min(n_dtcs - n_vc + 1, 6))
    n_med          = RNG.integers(0, min(n_dtcs - n_vc - n_hi + 1, 8))

    n_ab           = RNG.integers(0, 6)
    n_marg         = RNG.integers(0, 5)
    max_anom       = float(RNG.uniform(0, 1))
    mean_anom      = float(RNG.uniform(0, max_anom))

    bat_norm       = float(RNG.uniform(0, 1))   # 0=low, 1=high, 0.5=good
    cool_norm      = float(RNG.uniform(0, 1))
    ft_norm        = float(RNG.uniform(0, 1))   # 0=ideal, 1=very deviated
    maf_norm       = float(RNG.uniform(0, 1))
    rpm_norm       = float(RNG.uniform(0, 1))
    o2_norm        = float(RNG.uniform(0, 1))

    n_cats         = RNG.integers(0, 10)
    has_misfire    = float(RNG.integers(0, 2))
    has_cat        = float(RNG.integers(0, 2))
    has_abs        = float(RNG.integers(0, 2))
    has_air        = float(RNG.integers(0, 2))
    has_net        = float(RNG.integers(0, 2))
    mileage_band   = float(RNG.integers(0, 4))

    feat = np.array([
        highest_sev, n_dtcs, n_vc, n_hi, n_med, n_ab, n_marg,
        max_anom, mean_anom, bat_norm, cool_norm, ft_norm, maf_norm,
        rpm_norm, o2_norm, n_cats, has_misfire, has_cat, has_abs,
        has_air, has_net, mileage_band,
    ], dtype=np.float32)

    score = compute_health_score(feat)
    return feat, score


def generate_health_dataset(n: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    print(f"[health_scorer] Generating {n:,} synthetic sessions…")
    feats, scores = [], []
    for _ in range(n):
        f, s = _random_session()
        feats.append(f)
        scores.append(s)
    return np.array(feats, dtype=np.float32), np.array(scores, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Feature scaling helpers
# ══════════════════════════════════════════════════════════════════════════════

def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(0)
    std  = X.std(0) + 1e-6
    return mean, std


def scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# Dataset + Model
# ══════════════════════════════════════════════════════════════════════════════

class HealthDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)   # (N,1) for MSE

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HealthScorer(nn.Module):
    """Shallow MLP regressor: features → single score in [0, 100]."""

    def __init__(self, input_dim: int = N_FEATURES, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),   # output in (0, 1); multiply by 100 at inference
        )

    def forward(self, x):
        return self.net(x) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_health_scorer(
    n_sessions: int = 50000,
    epochs: int = 60,
    batch_size: int = 512,
    hidden: int = 128,
    lr: float = 1e-3,
    save_dir: Path = MODELS_DIR,
) -> dict:

    print("\n=== Health Scorer Training ===")
    X, y = generate_health_dataset(n_sessions)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42)

    sc_mean, sc_std = fit_scaler(X_tr)
    X_tr_s = scale(X_tr, sc_mean, sc_std)
    X_va_s = scale(X_va, sc_mean, sc_std)

    tr_set = HealthDataset(X_tr_s, y_tr)
    va_set = HealthDataset(X_va_s, y_va)
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device : {device}")

    model = HealthScorer(input_dim=N_FEATURES, hidden=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.HuberLoss(delta=5.0)

    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(Xb)
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xb, yb in va_loader:
                p = model(Xb.to(device)).cpu().numpy()
                preds.extend(p.flatten())
                trues.extend(yb.numpy().flatten())
        val_mae = mean_absolute_error(trues, preds)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | tr_loss={tr_loss/len(X_tr):.4f} | val_MAE={val_mae:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    model = model.cpu()

    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in va_loader:
            preds.extend(model(Xb).numpy().flatten())
            trues.extend(yb.numpy().flatten())
    final_mae = mean_absolute_error(trues, preds)
    final_r2  = r2_score(trues, preds)
    print(f"\n  Final val MAE={final_mae:.3f}  R²={final_r2:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    ckpt_path = save_dir / "health_scorer.pt"
    torch.save({
        "model_state": best_state,
        "hidden":      hidden,
        "sc_mean":     sc_mean.tolist(),
        "sc_std":      sc_std.tolist(),
        "val_mae":     final_mae,
        "val_r2":      final_r2,
    }, ckpt_path)

    # ONNX export
    onnx_path = save_dir / "health_scorer.onnx"
    dummy = torch.zeros(1, N_FEATURES)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["health_features"],
        output_names=["health_score"],
        dynamic_axes={"health_features": {0: "batch"}, "health_score": {0: "batch"}},
        opset_version=17,
    )

    print(f"  Saved: {ckpt_path.name}, {onnx_path.name}")

    return {
        "model":     model,
        "sc_mean":   sc_mean,
        "sc_std":    sc_std,
        "val_mae":   final_mae,
        "val_r2":    final_r2,
        "onnx_path": str(onnx_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Feature builder (session JSON → feature vector)
# ══════════════════════════════════════════════════════════════════════════════

def build_health_features(
    dtc_results: list[dict],
    sensor_results: dict,
    mileage_km: int = 80000,
) -> np.ndarray:
    """
    dtc_results : output of predict_dtc()   [{severity_int, category_label, …}]
    sensor_results: output of predict_sensor_anomaly()  {ch: {status, anomaly_score}}
    """
    from labeling import SEVERITY_TO_INT

    severities  = [r.get("severity_int", 0) for r in dtc_results]
    categories  = [r.get("category_label", "unknown") for r in dtc_results]
    n_dtcs      = min(len(dtc_results), 20)
    n_vc        = sum(1 for s in severities if s == 3)
    n_hi        = sum(1 for s in severities if s == 2)
    n_med       = sum(1 for s in severities if s == 1)
    highest_sev = max(severities, default=0)
    n_cats      = len(set(categories))

    has_misfire = float(any("ignition" in c or "misfire" in (r.get("description","")).lower()
                            for r, c in zip(dtc_results, categories)))
    has_cat     = float(any(c == "emissions" for c in categories))
    has_abs     = float(any(c == "abs_brakes" for c in categories))
    has_air     = float(any(c == "airbag"     for c in categories))
    has_net     = float(any(c == "network"    for c in categories))

    # Sensor aggregates
    anomaly_scores = [v["anomaly_score"] for v in sensor_results.values()]
    n_ab   = sum(1 for v in sensor_results.values() if v["status"] == "ABNORMAL")
    n_marg = sum(1 for v in sensor_results.values() if v["status"] == "MARGINAL")
    max_anom  = max(anomaly_scores, default=0.0)
    mean_anom = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0

    # Normalise key sensors to 0–1 using SENSOR_PARAMS ranges
    from sensor_anomaly import SENSOR_PARAMS

    def _norm(ch, val):
        p = SENSOR_PARAMS[ch]
        return float(np.clip((val - p["min"]) / (p["max"] - p["min"] + 1e-6), 0.0, 1.0))

    def _sr(ch):
        return sensor_results.get(ch, {}).get("anomaly_score", 0.0)

    bat_norm  = _sr("battery_voltage")
    cool_norm = _sr("coolant_temp")
    ft_norm   = (_sr("fuel_trim_short") + _sr("fuel_trim_long")) / 2.0
    maf_norm  = _sr("maf")
    rpm_norm  = _sr("rpm")
    o2_norm   = _sr("o2_voltage")

    mileage_band = min(int(mileage_km // 50000), 3)

    return np.array([
        highest_sev, n_dtcs, n_vc, n_hi, n_med, n_ab, n_marg,
        max_anom, mean_anom, bat_norm, cool_norm, ft_norm, maf_norm,
        rpm_norm, o2_norm, n_cats, has_misfire, has_cat, has_abs,
        has_air, has_net, mileage_band,
    ], dtype=np.float32)


def predict_health_score(
    dtc_results: list[dict],
    sensor_results: dict,
    mileage_km: int = 80000,
    save_dir: Path = MODELS_DIR,
) -> dict:
    ckpt = torch.load(save_dir / "health_scorer.pt", weights_only=False)
    model = HealthScorer(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sc_mean = np.array(ckpt["sc_mean"], dtype=np.float32)
    sc_std  = np.array(ckpt["sc_std"],  dtype=np.float32)

    feat = build_health_features(dtc_results, sensor_results, mileage_km)
    feat_s = scale(feat[None], sc_mean, sc_std)

    with torch.no_grad():
        score = model(torch.from_numpy(feat_s)).item()

    status = (
        "CRITICAL" if score < 30 else
        "POOR"     if score < 50 else
        "FAIR"     if score < 70 else
        "GOOD"     if score < 88 else
        "EXCELLENT"
    )
    return {"score": round(score, 1), "status": status}


if __name__ == "__main__":
    train_health_scorer()
