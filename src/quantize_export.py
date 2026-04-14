"""
quantize_export.py — Post-training dynamic INT8 quantization of all ONNX models
                     + model_meta.json generation.

ONNX Runtime's quantize_dynamic() applies INT8 weight quantization without
requiring a calibration dataset, making it suitable for the sub-5 MB on-device
bundle target.

Produces:
  models/dtc_classifier_int8.onnx
  models/sensor_anomaly_int8.onnx
  models/health_scorer_int8.onnx
  models/model_meta.json
"""

import json
import os
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS_TO_QUANTIZE = [
    ("dtc_classifier.onnx",  "dtc_classifier_int8.onnx"),
    ("sensor_anomaly.onnx",  "sensor_anomaly_int8.onnx"),
    ("health_scorer.onnx",   "health_scorer_int8.onnx"),
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_048_576, 3)


def quantize_all(models_dir: Path = MODELS_DIR) -> dict:
    print("\n=== INT8 Dynamic Quantization ===")
    meta_models = {}

    for src_name, dst_name in MODELS_TO_QUANTIZE:
        src = models_dir / src_name
        dst = models_dir / dst_name

        if not src.exists():
            print(f"  [SKIP] {src_name} not found — run training first")
            continue

        src_mb = _size_mb(src)
        print(f"  Quantizing {src_name} ({src_mb} MB) …", end=" ", flush=True)

        try:
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QUInt8,
            )
            quantized = True
        except Exception as e:
            # Newer torch dynamo ONNX graphs can fail ORT shape-inference;
            # fall back to float32 ONNX (still deployable on ONNX Runtime Mobile).
            print(f"\n    [WARN] INT8 quantization failed ({type(e).__name__}); "
                  f"using float32 ONNX model instead.")
            import shutil
            shutil.copy2(src, dst)
            quantized = False

        dst_mb = _size_mb(dst)
        reduction = round((1 - dst_mb / src_mb) * 100, 1) if quantized else 0.0
        sha = _file_sha256(dst)
        quant_label = "INT8_dynamic" if quantized else "float32_fallback"

        if quantized:
            print(f"{src_mb} MB → {dst_mb} MB  ({reduction}% smaller)  sha256={sha[:12]}…")
        else:
            print(f"  Copied float32: {dst_mb} MB  sha256={sha[:12]}…")

        # Validate the model runs under ONNX Runtime
        try:
            sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
            inputs  = sess.get_inputs()
            outputs = sess.get_outputs()
            ort_ok  = True
        except Exception as e:
            print(f"    [WARN] ORT validation failed: {e}")
            inputs, outputs, ort_ok = [], [], False

        meta_models[dst_name.replace("_int8.onnx", "")] = {
            "filename":      dst_name,
            "original_file": src_name,
            "size_mb":       dst_mb,
            "sha256":        sha,
            "input_names":   [i.name for i in inputs]  if ort_ok else [],
            "output_names":  [o.name for o in outputs] if ort_ok else [],
            "quantization":  quant_label,
            "reduction_pct": reduction,
            "ort_validated": ort_ok,
        }

    return meta_models


def write_model_meta(
    meta_models: dict,
    trigger_session_id: str | None = None,
    models_dir: Path = MODELS_DIR,
    training_results: dict | None = None,
) -> Path:
    """Write models/model_meta.json consumed by the mobile app."""
    meta = {
        "schema_version": "1.0",
        "bundle_id":      str(uuid.uuid4()),
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "trigger_session": trigger_session_id or "initial_training",
        "models":          meta_models,
        "training_metrics": training_results or {},
        "validation_requirements": {
            "max_accuracy_drop_pct": 2.0,
            "description": "New model bundle must not drop >2% accuracy on held-out validation set before replacing live model."
        },
        "deployment_notes": {
            "platform":     ["Android API 21+", "iOS 12+"],
            "runtime":      "ONNX Runtime Mobile",
            "max_bundle_mb": 5.0,
            "inference_ms":  "<100ms on mid-range device",
            "offline":       True,
        },
    }

    out_path = models_dir / "model_meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  model_meta.json written → {out_path}")
    return out_path


if __name__ == "__main__":
    meta_models = quantize_all()
    write_model_meta(meta_models)
