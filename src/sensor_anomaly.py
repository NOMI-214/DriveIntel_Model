"""
sensor_anomaly.py — LSTM Autoencoder for sensor timeseries anomaly detection.

Each of the 10 OBD-II sensor channels is modelled independently with a
shared architecture. A rolling 30-day window (sampled at ~4 readings/day)
gives sequence length = 120.

Since no real sensor timeseries data exists yet, we generate physics-informed
synthetic data:
  • NORMAL  : sensor oscillates around its typical idle/driving value with
              realistic Gaussian noise and slow drift.
  • ANOMALOUS: one of {spike, drift-out, stuck-low, stuck-high} patterns.

The LSTM autoencoder learns to reconstruct normal sequences. At inference time,
reconstruction error per channel = anomaly score (0–1 after normalisation).

Outputs (per session):
  {channel: {status, anomaly_score, days_to_alert, trend}}
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Sensor Definitions
# ══════════════════════════════════════════════════════════════════════════════

SENSOR_CHANNELS = [
    "rpm", "coolant_temp", "o2_voltage", "maf",
    "fuel_trim_short", "fuel_trim_long", "throttle_pos",
    "battery_voltage", "intake_air_temp", "vehicle_speed",
]

# Typical idle/city values and normal ranges for synthetic generation
SENSOR_PARAMS = {
    "rpm":              {"mean": 800,  "std": 80,   "min": 500,  "max": 1200,  "unit": "rpm",  "alert_hi": 4500, "alert_lo": 400},
    "coolant_temp":     {"mean": 90,   "std": 3,    "min": 80,   "max": 100,   "unit": "°C",   "alert_hi": 110,  "alert_lo": 50},
    "o2_voltage":       {"mean": 0.45, "std": 0.15, "min": 0.1,  "max": 0.9,   "unit": "V",    "alert_hi": 1.0,  "alert_lo": 0.05},
    "maf":              {"mean": 4.5,  "std": 0.8,  "min": 2.0,  "max": 8.0,   "unit": "g/s",  "alert_hi": 25.0, "alert_lo": 1.0},
    "fuel_trim_short":  {"mean": 1.5,  "std": 2.0,  "min": -10,  "max": 10,    "unit": "%",    "alert_hi": 25,   "alert_lo": -25},
    "fuel_trim_long":   {"mean": 1.0,  "std": 1.5,  "min": -10,  "max": 10,    "unit": "%",    "alert_hi": 20,   "alert_lo": -20},
    "throttle_pos":     {"mean": 14,   "std": 4,    "min": 5,    "max": 30,    "unit": "%",    "alert_hi": 95,   "alert_lo": 0},
    "battery_voltage":  {"mean": 13.8, "std": 0.2,  "min": 13.2, "max": 14.8,  "unit": "V",    "alert_hi": 15.0, "alert_lo": 11.5},
    "intake_air_temp":  {"mean": 30,   "std": 5,    "min": 10,   "max": 50,    "unit": "°C",   "alert_hi": 70,   "alert_lo": -10},
    "vehicle_speed":    {"mean": 30,   "std": 20,   "min": 0,    "max": 80,    "unit": "km/h", "alert_hi": 200,  "alert_lo": 0},
}

SEQ_LEN   = 120   # ~30 days at 4 readings/day
N_SENSORS = len(SENSOR_CHANNELS)
RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generator
# ══════════════════════════════════════════════════════════════════════════════

def _normal_sequence(params: dict, length: int = SEQ_LEN) -> np.ndarray:
    """
    Realistic normal timeseries: mix of idle, city, and highway segments
    with gradual warm-up, noise, and mild sensor-specific variation.
    Avoids perfect sine waves so the autoencoder generalises to real OBD snapshots.
    """
    seq = np.zeros(length, dtype=np.float32)
    mean, std = params["mean"], params["std"]

    # Random driving segment lengths
    seg_len = length // 4
    base_vals = RNG.normal(mean, std * 0.15, 4)   # 4 segments with slightly different baselines
    for i in range(4):
        seg = base_vals[i] + RNG.normal(0, std * 0.4, seg_len)
        seq[i*seg_len:(i+1)*seg_len] = seg

    # Smooth with a small window to avoid unrealistically jagged transitions
    kernel = np.ones(5) / 5
    seq = np.convolve(seq, kernel, mode='same')
    seq = np.clip(seq, params["min"] * 0.85, params["max"] * 1.15)
    return seq.astype(np.float32)


def _anomalous_sequence(params: dict, kind: str, length: int = SEQ_LEN) -> np.ndarray:
    seq = _normal_sequence(params, length)
    start = RNG.integers(length // 4, length // 2)

    if kind == "spike":
        n_spikes = RNG.integers(3, 10)
        idx = RNG.integers(start, length, n_spikes)
        spike_mag = params["std"] * RNG.uniform(5, 15)
        seq[idx] += spike_mag * RNG.choice([-1, 1])

    elif kind == "drift_high":
        drift = np.linspace(0, (params["alert_hi"] - params["mean"]) * 0.9, length - start)
        seq[start:] += drift

    elif kind == "drift_low":
        drift = np.linspace(0, (params["mean"] - params["alert_lo"]) * 0.9, length - start)
        seq[start:] -= drift

    elif kind == "stuck":
        stuck_val = params["mean"] * RNG.uniform(0.2, 0.5)
        seq[start:] = stuck_val + RNG.normal(0, params["std"] * 0.05, length - start)

    return seq.astype(np.float32)


def generate_synthetic_dataset(
    n_normal: int = 8000,
    n_anomalous: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where:
      X : (N, SEQ_LEN, N_SENSORS)   float32
      y : (N,)                       0=normal, 1=anomalous
    """
    print(f"[sensor_anomaly] Generating {n_normal} normal + {n_anomalous} anomalous sequences…")
    kinds = ["spike", "drift_high", "drift_low", "stuck"]
    normal_seqs, anomalous_seqs = [], []

    for _ in range(n_normal):
        sample = np.stack(
            [_normal_sequence(SENSOR_PARAMS[ch]) for ch in SENSOR_CHANNELS], axis=1
        )  # (SEQ_LEN, N_SENSORS)
        normal_seqs.append(sample)

    for _ in range(n_anomalous):
        # Inject anomaly into 1–3 random channels
        sample = np.stack(
            [_normal_sequence(SENSOR_PARAMS[ch]) for ch in SENSOR_CHANNELS], axis=1
        )
        n_aff = RNG.integers(1, 4)
        aff_ch = RNG.choice(N_SENSORS, n_aff, replace=False)
        for ch_idx in aff_ch:
            ch_name = SENSOR_CHANNELS[ch_idx]
            kind = kinds[RNG.integers(0, len(kinds))]
            sample[:, ch_idx] = _anomalous_sequence(SENSOR_PARAMS[ch_name], kind)
        anomalous_seqs.append(sample)

    X = np.array(normal_seqs + anomalous_seqs, dtype=np.float32)
    y = np.array([0] * n_normal + [1] * n_anomalous, dtype=np.int64)

    # Shuffle
    idx = RNG.permutation(len(X))
    return X[idx], y[idx]


# Per-channel normalisation (fitted on training normals only)
def fit_normaliser(X_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) of shape (N_SENSORS,) fitted on normal sequences."""
    flat = X_normal.reshape(-1, N_SENSORS)
    return flat.mean(0), flat.std(0) + 1e-6


def normalise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, T, C)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, latent_dim, n_layers, batch_first=True,
                            dropout=0.2, bidirectional=False)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]   # (batch, latent_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, n_features: int, seq_len: int, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim, latent_dim, n_layers, batch_first=True, dropout=0.2)
        self.out_proj = nn.Linear(latent_dim, n_features)

    def forward(self, z):
        # Repeat latent vector across time steps
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, latent)
        out, _ = self.lstm(z_seq)
        return self.out_proj(out)   # (B, T, n_features)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = N_SENSORS, latent_dim: int = 64,
                 seq_len: int = SEQ_LEN, n_layers: int = 2):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, latent_dim, n_layers)
        self.decoder = LSTMDecoder(latent_dim, n_features, seq_len, n_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean reconstruction error across all channels & timesteps."""
        with torch.no_grad():
            recon = self.forward(x)
            err = (recon - x).pow(2).mean(dim=(1, 2))   # (B,)
        return err


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_sensor_anomaly(
    n_normal: int = 8000,
    n_anomalous: int = 4000,
    epochs: int = 30,
    batch_size: int = 64,
    latent_dim: int = 64,
    lr: float = 1e-3,
    save_dir: Path = MODELS_DIR,
) -> dict:

    print("\n=== Sensor Anomaly Detector Training ===")
    X, y = generate_synthetic_dataset(n_normal, n_anomalous)

    # Train on NORMAL sequences only (unsupervised autoencoder)
    X_norm = X[y == 0]
    X_anom = X[y == 1]

    X_tr_n, X_va_n = train_test_split(X_norm, test_size=0.15, random_state=42)

    # Normaliser fitted on training normals
    norm_mean, norm_std = fit_normaliser(X_tr_n)
    X_tr_n_scaled = normalise(X_tr_n, norm_mean, norm_std)
    X_va_n_scaled  = normalise(X_va_n, norm_mean, norm_std)
    X_va_a_scaled  = normalise(X_anom[:len(X_va_n)], norm_mean, norm_std)

    tr_set = SensorDataset(X_tr_n_scaled, np.zeros(len(X_tr_n_scaled)))
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device : {device}")

    model = LSTMAutoencoder(n_features=N_SENSORS, latent_dim=latent_dim, seq_len=SEQ_LEN).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, _ in tr_loader:
            Xb = Xb.to(device)
            optim.zero_grad()
            recon = model(Xb)
            loss = criterion(recon, Xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * len(Xb)

        # Validation reconstruction loss on normal samples
        model.eval()
        with torch.no_grad():
            va_n_t = torch.from_numpy(X_va_n_scaled).to(device)
            va_loss = criterion(model(va_n_t), va_n_t).item()
        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | tr_loss={tr_loss/len(X_tr_n):.5f} | "
                  f"va_loss={va_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    model = model.cpu()

    # ── AUC evaluation ────────────────────────────────────────────────────────
    with torch.no_grad():
        s_n = model.anomaly_score(torch.from_numpy(X_va_n_scaled)).numpy()
        s_a = model.anomaly_score(torch.from_numpy(X_va_a_scaled)).numpy()

    scores = np.concatenate([s_n, s_a])
    labels = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    auc = roc_auc_score(labels, scores)
    print(f"\n  Validation AUC (normal vs anomalous): {auc:.4f}")

    # Global threshold = 99th pct of normal reconstruction errors (reduces false positives)
    threshold = float(np.percentile(s_n, 99))
    print(f"  Anomaly threshold (99th pct on val-normal): {threshold:.5f}")

    # Per-channel thresholds: 99th pct of per-channel MSE on normal val sequences
    with torch.no_grad():
        Xn_t  = torch.from_numpy(X_va_n_scaled)
        recon_n = model(Xn_t).numpy()
        ch_mse_normal = ((recon_n - X_va_n_scaled) ** 2).mean(axis=1)  # (N, C)
        ch_thresholds = np.percentile(ch_mse_normal, 99, axis=0)        # (C,)
    print(f"  Per-channel thresholds (99th pct): {dict(zip(SENSOR_CHANNELS, ch_thresholds.round(4)))}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    ckpt_path = save_dir / "sensor_anomaly.pt"
    torch.save({
        "model_state":    best_state,
        "norm_mean":      norm_mean.tolist(),
        "norm_std":       norm_std.tolist(),
        "threshold":      threshold,
        "ch_thresholds":  ch_thresholds.tolist(),
        "latent_dim":     latent_dim,
        "seq_len":        SEQ_LEN,
        "n_sensors":      N_SENSORS,
        "channels":       SENSOR_CHANNELS,
        "auc":            auc,
    }, ckpt_path)

    # ONNX export
    onnx_path = save_dir / "sensor_anomaly.onnx"
    dummy = torch.zeros(1, SEQ_LEN, N_SENSORS)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["sensor_sequence"],
        output_names=["reconstructed_sequence"],
        dynamic_axes={"sensor_sequence": {0: "batch"}, "reconstructed_sequence": {0: "batch"}},
        opset_version=17,
    )

    print(f"  Saved: {ckpt_path.name}, {onnx_path.name}")

    return {
        "model": model,
        "norm_mean": norm_mean,
        "norm_std":  norm_std,
        "threshold": threshold,
        "auc":       auc,
        "onnx_path": str(onnx_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def predict_sensor_anomaly(
    sensor_timeseries: dict,   # {channel_name: [{"ts": ..., "value": v}, ...]}
    save_dir: Path = MODELS_DIR,
) -> dict:
    """
    Input: dict of channel → list of readings (from sensor_timeseries.json).
    Output: {channel: {status, anomaly_score, days_to_alert, trend}}
    """
    ckpt = torch.load(save_dir / "sensor_anomaly.pt", weights_only=False)
    model = LSTMAutoencoder(
        n_features=ckpt["n_sensors"],
        latent_dim=ckpt["latent_dim"],
        seq_len=ckpt["seq_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm_mean     = np.array(ckpt["norm_mean"],     dtype=np.float32)
    norm_std      = np.array(ckpt["norm_std"],      dtype=np.float32)
    threshold     = ckpt["threshold"]
    # Per-channel thresholds (fallback: global for older checkpoints)
    ch_thresholds = np.array(ckpt.get("ch_thresholds", [threshold] * N_SENSORS), dtype=np.float32)
    channels      = ckpt["channels"]

    # Build (1, SEQ_LEN, N_SENSORS) tensor from provided readings
    seq = np.zeros((SEQ_LEN, N_SENSORS), dtype=np.float32)
    for ci, ch in enumerate(channels):
        readings = sensor_timeseries.get(ch, [])
        vals = [r["value"] if isinstance(r, dict) else float(r) for r in readings]
        vals = vals[-SEQ_LEN:]
        if vals:
            pad = SEQ_LEN - len(vals)
            seq[:, ci] = np.pad(vals, (pad, 0), mode="edge")
        else:
            seq[:, ci] = SENSOR_PARAMS[ch]["mean"]

    X_scaled = normalise(seq, norm_mean, norm_std)
    Xt = torch.from_numpy(X_scaled).unsqueeze(0)   # (1, T, C)

    with torch.no_grad():
        recon = model(Xt).squeeze(0).numpy()        # (T, C)
        ch_errors = (recon - X_scaled) ** 2         # (T, C)
        ch_mse = ch_errors.mean(0)                  # (C,)

    # Normalise per-channel MSE against that channel's 99th-pct threshold → 0-1 score
    results = {}
    for ci, ch in enumerate(channels):
        raw_mse   = float(ch_mse[ci])
        ch_thresh = float(ch_thresholds[ci]) + 1e-9
        norm_score = min(raw_mse / ch_thresh, 1.0)

        trend = "stable"
        recent_half = X_scaled[SEQ_LEN // 2:, ci]
        early_half  = X_scaled[:SEQ_LEN // 2, ci]
        delta = recent_half.mean() - early_half.mean()
        if delta > 0.4:
            trend = "increasing"
        elif delta < -0.4:
            trend = "decreasing"

        # Status thresholds: ABNORMAL if score > 0.85, MARGINAL if > 0.50
        status = "NORMAL"
        days_to_alert = None
        if norm_score > 0.85:
            status = "ABNORMAL"
            days_to_alert = 3
        elif norm_score > 0.50:
            status = "MARGINAL"
            days_to_alert = 14

        results[ch] = {
            "status":        status,
            "anomaly_score": round(norm_score, 4),
            "days_to_alert": days_to_alert,
            "trend":         trend,
        }

    return results


if __name__ == "__main__":
    train_sensor_anomaly()
