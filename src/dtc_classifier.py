"""
dtc_classifier.py — DTC Severity + Category Classifier

Architecture: TF-IDF text features + tabular one-hot → MLP (PyTorch)
Inputs : DTC code embedding (prefix + bucketed range) + TF-IDF of description
Outputs: severity (4-class) + fault_category (16-class)

Training uses the rule-labeled ground truth from labeling.py.
Multi-task loss: CrossEntropy(severity) + CrossEntropy(category)
"""

import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pathlib import Path

from labeling import SEVERITY_LABELS, FAULT_CATEGORIES, SEVERITY_TO_INT, CAT_TO_INT

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

CODE_PREFIXES = ["P", "B", "C", "U", "X"]   # X = other

# Bucketed code number ranges  →  one-hot
CODE_BUCKETS = [
    (0,    99),
    (100,  199),
    (200,  299),
    (300,  399),
    (400,  499),
    (500,  599),
    (600,  699),
    (700,  899),
    (900,  999),
    (1000, 1999),
    (2000, 2999),
    (3000, 9999),
]
N_BUCKETS = len(CODE_BUCKETS)
N_PREFIX  = len(CODE_PREFIXES)


def _bucket_index(num: int) -> int:
    for i, (lo, hi) in enumerate(CODE_BUCKETS):
        if lo <= num <= hi:
            return i
    return N_BUCKETS - 1


def build_tabular_features(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode prefix (5) + bucket (12) → 17-dim vector."""
    rows = []
    for _, r in df.iterrows():
        p = str(r["code_prefix"]).upper()
        pid = CODE_PREFIXES.index(p) if p in CODE_PREFIXES else CODE_PREFIXES.index("X")
        p_vec = [0] * N_PREFIX
        p_vec[pid] = 1

        num = int(r["code_number"]) if r["code_number"] >= 0 else 0
        b_vec = [0] * N_BUCKETS
        b_vec[_bucket_index(num)] = 1

        rows.append(p_vec + b_vec)
    return np.array(rows, dtype=np.float32)


class DTCFeaturiser:
    """Fits TF-IDF on training data; transforms new records at inference time."""

    def __init__(self, max_features: int = 3000):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.max_features = max_features
        self.n_tab = N_PREFIX + N_BUCKETS   # 17

    def fit_transform(self, df: pd.DataFrame):
        text = (df["description"] + " " + df["possible_causes"]).fillna("").values
        tab  = build_tabular_features(df)
        tfidf_mat = self.tfidf.fit_transform(text).toarray().astype(np.float32)
        return np.hstack([tab, tfidf_mat])

    def transform(self, df: pd.DataFrame):
        text = (df["description"] + " " + df["possible_causes"]).fillna("").values
        tab  = build_tabular_features(df)
        tfidf_mat = self.tfidf.transform(text).toarray().astype(np.float32)
        return np.hstack([tab, tfidf_mat])

    def input_dim(self) -> int:
        return self.n_tab + self.max_features

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "DTCFeaturiser":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset + Model
# ══════════════════════════════════════════════════════════════════════════════

class DTCDataset(Dataset):
    def __init__(self, X: np.ndarray, sev: np.ndarray, cat: np.ndarray):
        self.X   = torch.from_numpy(X)
        self.sev = torch.from_numpy(sev).long()
        self.cat = torch.from_numpy(cat).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.sev[idx], self.cat[idx]


class DTCClassifier(nn.Module):
    """
    Shared MLP backbone with two classification heads:
      - severity_head: 4 classes (LOW / MEDIUM / HIGH / VERY_CRITICAL)
      - category_head: 16 classes (fault categories)
    """

    def __init__(self, input_dim: int, hidden: int = 512,
                 n_severity: int = 4, n_category: int = 16, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
        )
        self.severity_head = nn.Linear(hidden // 4, n_severity)
        self.category_head = nn.Linear(hidden // 4, n_category)

    def forward(self, x):
        feat = self.backbone(x)
        return self.severity_head(feat), self.category_head(feat)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_dtc_classifier(
    df: pd.DataFrame,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 512,
    tfidf_features: int = 3000,
    save_dir: Path = MODELS_DIR,
) -> dict:

    print("\n=== DTC Classifier Training ===")
    featuriser = DTCFeaturiser(max_features=tfidf_features)

    X = featuriser.fit_transform(df)
    y_sev = df["severity"].values.astype(np.int64)
    y_cat = df["category"].values.astype(np.int64)

    X_tr, X_va, ys_tr, ys_va, yc_tr, yc_va = train_test_split(
        X, y_sev, y_cat, test_size=0.15, random_state=42, stratify=y_sev
    )

    tr_set = DTCDataset(X_tr, ys_tr, yc_tr)
    va_set = DTCDataset(X_va, ys_va, yc_va)
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device : {device}")

    model = DTCClassifier(
        input_dim=featuriser.input_dim(),
        hidden=hidden,
        n_severity=len(SEVERITY_LABELS),
        n_category=len(FAULT_CATEGORIES),
    ).to(device)

    # Class weights for imbalanced severity distribution
    sev_counts = np.bincount(ys_tr, minlength=len(SEVERITY_LABELS)).astype(np.float32)
    sev_weights = torch.tensor(1.0 / (sev_counts + 1e-6), device=device)
    sev_weights /= sev_weights.sum()

    cat_counts = np.bincount(yc_tr, minlength=len(FAULT_CATEGORIES)).astype(np.float32)
    cat_weights = torch.tensor(1.0 / (cat_counts + 1e-6), device=device)
    cat_weights /= cat_weights.sum()

    loss_sev = nn.CrossEntropyLoss(weight=sev_weights * len(SEVERITY_LABELS))
    loss_cat = nn.CrossEntropyLoss(weight=cat_weights * len(FAULT_CATEGORIES))

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for Xb, sb, cb in tr_loader:
            Xb, sb, cb = Xb.to(device), sb.to(device), cb.to(device)
            optim.zero_grad()
            ls, lc = model(Xb)
            loss = loss_sev(ls, sb) + 0.5 * loss_cat(lc, cb)
            loss.backward()
            optim.step()
            tr_loss += loss.item() * len(Xb)
            tr_correct += (ls.argmax(1) == sb).sum().item()
            tr_total += len(Xb)
        scheduler.step()

        # Validation
        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for Xb, sb, cb in va_loader:
                Xb, sb = Xb.to(device), sb.to(device)
                ls, _ = model(Xb)
                va_correct += (ls.argmax(1) == sb).sum().item()
                va_total += len(Xb)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={tr_loss/tr_total:.4f} | "
                  f"tr_acc={tr_acc:.4f} | va_acc={va_acc:.4f}")

    # ── Restore best weights & evaluate ──────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    all_true_s, all_pred_s = [], []
    all_true_c, all_pred_c = [], []
    with torch.no_grad():
        for Xb, sb, cb in va_loader:
            ls, lc = model(Xb.to(device))
            all_true_s.extend(sb.numpy())
            all_pred_s.extend(ls.argmax(1).cpu().numpy())
            all_true_c.extend(cb.numpy())
            all_pred_c.extend(lc.argmax(1).cpu().numpy())

    print("\n── Severity Classification Report ──")
    print(classification_report(all_true_s, all_pred_s, target_names=SEVERITY_LABELS, zero_division=0))
    print("── Category Classification Report ──")
    present_cats = sorted(set(all_true_c) | set(all_pred_c))
    cat_names = [FAULT_CATEGORIES[i] for i in present_cats]
    print(classification_report(all_true_c, all_pred_c,
                                labels=present_cats,
                                target_names=cat_names,
                                zero_division=0))

    # ── Save artefacts ────────────────────────────────────────────────────────
    model_cpu = model.cpu()

    # PyTorch checkpoint
    ckpt_path = save_dir / "dtc_classifier.pt"
    torch.save({
        "model_state": best_state,
        "input_dim":   featuriser.input_dim(),
        "hidden":      hidden,
        "best_val_acc": best_val_acc,
    }, ckpt_path)

    # Featuriser (TF-IDF + column order)
    feat_path = save_dir / "dtc_featuriser.pkl"
    featuriser.save(feat_path)

    # Export to ONNX
    onnx_path = save_dir / "dtc_classifier.onnx"
    dummy = torch.zeros(1, featuriser.input_dim())
    torch.onnx.export(
        model_cpu, dummy, str(onnx_path),
        input_names=["features"],
        output_names=["severity_logits", "category_logits"],
        dynamic_axes={"features": {0: "batch"}},
        opset_version=17,
    )

    print(f"\n  Saved: {ckpt_path.name}, {feat_path.name}, {onnx_path.name}")
    print(f"  Best validation severity accuracy: {best_val_acc:.4f}")

    return {
        "model": model_cpu,
        "featuriser": featuriser,
        "best_val_acc": best_val_acc,
        "onnx_path": str(onnx_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Inference helper
# ══════════════════════════════════════════════════════════════════════════════

def predict_dtc(records: list[dict], save_dir: Path = MODELS_DIR) -> list[dict]:
    """
    records : list of dicts with keys: dtc_code, description, possible_causes,
              code_prefix, code_number
    Returns enriched dicts with severity_label, severity_int, category_label,
            severity_probs, category_probs
    """
    feat_path = save_dir / "dtc_featuriser.pkl"
    ckpt_path = save_dir / "dtc_classifier.pt"
    featuriser = DTCFeaturiser.load(feat_path)
    ckpt = torch.load(ckpt_path, weights_only=False)

    model = DTCClassifier(input_dim=ckpt["input_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.DataFrame(records)
    X = featuriser.transform(df)
    with torch.no_grad():
        ls, lc = model(torch.from_numpy(X))
        s_probs = torch.softmax(ls, dim=1).numpy()
        c_probs = torch.softmax(lc, dim=1).numpy()

    out = []
    for i, rec in enumerate(records):
        s_idx = int(np.argmax(s_probs[i]))
        c_idx = int(np.argmax(c_probs[i]))
        out.append({
            **rec,
            "severity_int":   s_idx,
            "severity_label": SEVERITY_LABELS[s_idx],
            "category_label": FAULT_CATEGORIES[c_idx],
            "severity_probs": s_probs[i].tolist(),
            "category_probs": c_probs[i].tolist(),
        })
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all
    from labeling import label_dataframe

    df = load_all()
    df = label_dataframe(df)
    train_dtc_classifier(df)
