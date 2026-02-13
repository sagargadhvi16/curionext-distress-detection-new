"""Train transformer-based fusion using XGBoost audio features + synthetic biometric/context data.

Notes:
- Uses audio_experiments/xgb_baseline artifacts (no changes there).
- Keeps existing audio/ and data/ folders unchanged.
- Generates biometric/context data in-memory for fusion training.
"""

from __future__ import annotations

import os
import sys
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Configuration
# -------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_experiments.xgb_baseline.features import extract_feature_chunk
from src.fusion.model import DistressDetectionModel
SYNTH_AUDIO_DIR = PROJECT_ROOT / "data" / "synthetic" / "audio" / "distress"
METADATA_PATH = SYNTH_AUDIO_DIR / "metadata.csv"

XGB_MODEL_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_xgb_3class.joblib"
XGB_SCALER_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_scaler_3class.joblib"
XGB_ENCODER_PATH = PROJECT_ROOT / "audio_experiments" / "xgb_baseline" / "artifacts" / "your_distress_label_encoder_3class.joblib"

SR = 16000
WINDOW_SEC = 10

SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

NUM_DISTRESS_TYPES = 5  # anxiety, panic, pain, fatigue, other


# -------------------------------
# Helpers
# -------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_audio_chunk(path: Path, sr: int = SR, window_sec: int = WINDOW_SEC) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sr)
    window_size = sr * window_sec
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    else:
        audio = audio[:window_size]
    return audio


def generate_normal_audio_chunk(sr: int = SR, window_sec: int = WINDOW_SEC) -> np.ndarray:
    window_size = sr * window_sec
    noise = np.random.normal(0.0, 0.01, size=window_size).astype(np.float32)
    return noise


def map_source_to_type(source: str) -> int:
    s = (source or "").lower()
    if "scream" in s:
        return 1  # panic
    if "cry" in s:
        return 0  # anxiety
    if "ravdess" in s:
        return 2  # pain (approx)
    return 4  # other


def severity_from_label(label: str) -> float:
    l = label.lower()
    if l == "high_arousal":
        return float(np.random.uniform(7.0, 10.0))
    if l == "distress":
        return float(np.random.uniform(4.0, 7.0))
    return float(np.random.uniform(0.0, 2.5))


def synth_biometric(distress: int, distress_type: int, size: int = 200) -> np.ndarray:
    base = np.random.normal(0.0, 1.0, size=size)
    if distress == 1:
        base += 0.8
        if distress_type == 1:  # panic
            base[:40] += 1.0
        elif distress_type == 0:  # anxiety
            base[40:80] += 0.7
        elif distress_type == 2:  # pain
            base[80:120] += 0.6
        elif distress_type == 3:  # fatigue
            base[120:160] -= 0.4
    return base.astype(np.float32)


def synth_context(distress: int, size: int = 12) -> np.ndarray:
    # Example context: [time_of_day, noise_level, motion_level, device_type_onehot(4), location_onehot(4)]
    time_of_day = np.random.uniform(0.0, 1.0)
    noise_level = np.random.uniform(0.0, 1.0) + (0.2 if distress == 1 else 0.0)
    motion_level = np.random.uniform(0.0, 1.0) + (0.1 if distress == 1 else 0.0)

    device_onehot = np.zeros(4, dtype=np.float32)
    device_onehot[np.random.randint(0, 4)] = 1.0

    location_onehot = np.zeros(4, dtype=np.float32)
    location_onehot[np.random.randint(0, 4)] = 1.0

    vec = np.hstack([time_of_day, noise_level, motion_level, device_onehot, location_onehot])
    if vec.shape[0] < size:
        vec = np.pad(vec, (0, size - vec.shape[0]))
    return vec[:size].astype(np.float32)


class XGBAudioFusionDataset(Dataset):
    def __init__(self, audio_paths: List[Path], metadata: Dict[str, Dict], xgb_model, scaler, label_encoder, include_normals: bool = True):
        self.audio_paths = audio_paths
        self.metadata = metadata
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.include_normals = include_normals

        # Precompute feature dimension
        test_audio = generate_normal_audio_chunk()
        feat = extract_feature_chunk(test_audio, SR)
        self.feature_dim = feat.shape[0] + 3  # +3 for XGB probas

    def __len__(self) -> int:
        base_len = len(self.audio_paths)
        return base_len * 2 if self.include_normals else base_len

    def _audio_to_features(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        feat = extract_feature_chunk(audio, SR)
        feat_scaled = self.scaler.transform([feat])[0]
        proba = self.xgb_model.predict_proba([feat_scaled])[0]
        pred_enc = self.xgb_model.predict([feat_scaled])[0]
        pred_label = self.label_encoder.inverse_transform([pred_enc])[0]
        audio_vec = np.hstack([feat_scaled, proba]).astype(np.float32)
        return audio_vec, proba.astype(np.float32), pred_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        is_normal = self.include_normals and idx >= len(self.audio_paths)

        if is_normal:
            audio = generate_normal_audio_chunk()
            audio_vec, _, pred_label = self._audio_to_features(audio)
            distress = 0
            distress_type = 4  # other
            severity = severity_from_label("neutral")
        else:
            audio_path = self.audio_paths[idx]
            audio = load_audio_chunk(audio_path)
            audio_vec, _, pred_label = self._audio_to_features(audio)

            meta = self.metadata.get(audio_path.name, {})
            source = meta.get("source", "")

            distress = 0 if pred_label == "neutral" else 1
            if distress == 0:
                distress_type = 4  # other
                severity = severity_from_label("neutral")
            else:
                # Prefer metadata mapping, fall back to prediction
                distress_type = map_source_to_type(source)
                if distress_type == 4 and pred_label == "high_arousal":
                    distress_type = 1  # panic
                elif distress_type == 4 and pred_label == "distress":
                    distress_type = 0  # anxiety
                severity = severity_from_label(pred_label)

        biometric = synth_biometric(distress, distress_type, size=200)
        context = synth_context(distress, size=12)

        targets = {
            "distress": torch.tensor(distress, dtype=torch.long),
            "severity": torch.tensor(severity, dtype=torch.float32),
            "type": torch.tensor(distress_type, dtype=torch.long),
        }

        return (
            torch.from_numpy(audio_vec),
            torch.from_numpy(biometric),
            torch.from_numpy(context),
            targets,
        )


def load_metadata(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    import csv
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row.get("file", "")] = row
    return out


def create_model(audio_input_dim: int) -> nn.Module:
    audio_encoder = nn.Sequential(
        nn.Linear(audio_input_dim, 256),
        nn.ReLU(),
    )
    biometric_encoder = nn.Sequential(
        nn.Linear(200, 256),
        nn.ReLU(),
    )
    context_encoder = nn.Sequential(
        nn.Linear(12, 64),
        nn.ReLU(),
    )

    return DistressDetectionModel(
        audio_encoder=audio_encoder,
        biometric_encoder=biometric_encoder,
        context_encoder=context_encoder,
        fusion_type="transformer",
        use_tcn_refiner=False,
        audio_dim=256,
        bio_dim=256,
        context_dim=64,
        num_distress_types=NUM_DISTRESS_TYPES,
    )


def compute_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {}
    if "distress_logits" in outputs:
        preds = torch.argmax(outputs["distress_logits"], dim=1)
        metrics["distress_acc"] = (preds == targets["distress"]).float().mean().item()
    if "type_logits" in outputs:
        preds = torch.argmax(outputs["type_logits"], dim=1)
        metrics["type_acc"] = (preds == targets["type"]).float().mean().item()
    if "severity" in outputs:
        metrics["severity_mae"] = torch.mean(torch.abs(outputs["severity"].squeeze() - targets["severity"])).item()
    return metrics


def train() -> None:
    set_seed(SEED)

    # Load XGB artifacts
    xgb_model = joblib.load(XGB_MODEL_PATH)
    scaler = joblib.load(XGB_SCALER_PATH)
    label_encoder = joblib.load(XGB_ENCODER_PATH)

    metadata = load_metadata(METADATA_PATH)
    audio_paths = sorted([p for p in SYNTH_AUDIO_DIR.glob("*.wav") if p.is_file()])

    if len(audio_paths) == 0:
        raise RuntimeError(f"No audio files found in {SYNTH_AUDIO_DIR}")

    dataset = XGBAudioFusionDataset(audio_paths, metadata, xgb_model, scaler, label_encoder, include_normals=True)

    # Split train/val
    num_total = len(dataset)
    num_train = int(0.8 * num_total)
    num_val = num_total - num_train
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])

    def collate_fn(batch):
        audio, biometric, context, targets = zip(*batch)
        audio = torch.stack(audio, dim=0).float()
        biometric = torch.stack(biometric, dim=0).float()
        context = torch.stack(context, dim=0).float()
        targets_batch = {
            "distress": torch.stack([t["distress"] for t in targets]),
            "severity": torch.stack([t["severity"] for t in targets]),
            "type": torch.stack([t["type"] for t in targets]),
        }
        return audio, biometric, context, targets_batch

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = create_model(dataset.feature_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_distress = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_severity = nn.MSELoss()
    loss_type = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for audio, biometric, context, targets in train_loader:
            audio = audio.to(device)
            biometric = biometric.to(device)
            context = context.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(audio, biometric, context)
            loss = 0.0
            loss += loss_distress(outputs["distress_logits"], targets["distress"]) * 1.0
            loss += loss_severity(outputs["severity"].squeeze(), targets["severity"]) * 0.5
            loss += loss_type(outputs["type_logits"], targets["type"]) * 0.8

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics_accum = {"distress_acc": 0.0, "type_acc": 0.0, "severity_mae": 0.0}
        with torch.no_grad():
            for audio, biometric, context, targets in val_loader:
                audio = audio.to(device)
                biometric = biometric.to(device)
                context = context.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                outputs = model(audio, biometric, context)
                loss = 0.0
                loss += loss_distress(outputs["distress_logits"], targets["distress"]) * 1.0
                loss += loss_severity(outputs["severity"].squeeze(), targets["severity"]) * 0.5
                loss += loss_type(outputs["type_logits"], targets["type"]) * 0.8
                val_loss += loss.item()

                metrics = compute_metrics(outputs, targets)
                for k in val_metrics_accum:
                    val_metrics_accum[k] += metrics.get(k, 0.0)

        val_loss /= max(len(val_loader), 1)
        for k in val_metrics_accum:
            val_metrics_accum[k] /= max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Distress Acc: {val_metrics_accum['distress_acc']:.3f} | "
            f"Type Acc: {val_metrics_accum['type_acc']:.3f} | "
            f"Severity MAE: {val_metrics_accum['severity_mae']:.3f}"
        )

    # Save checkpoint
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "transformer_fusion_xgb.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Training complete. Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    train()