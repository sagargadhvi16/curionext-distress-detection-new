import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

from src.audio.preprocessing import load_audio, normalize_audio
from src.audio.features import extract_mfcc


class AudioDistressDataset(Dataset):
    """
    Audio-only dataset for distress verification.

    Labeling logic:
    - Distress (1):
        * train_cry
        * test_cry
        * screaming/Screaming
        * RAVDESS: angry, fearful, sad
        * BESD: ANGER, FEAR, SAD
        * ADEMA / Prima (all)
    - Non-distress (0):
        * ESC-50
        * BESD: HAPPY, NEUTRAL
        * screaming/NotScreaming
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.samples = []

        self._collect_all()
        self._print_stats()

    # -------------------------------------------------
    # Dataset collection
    # -------------------------------------------------
    def _collect_all(self):
        # -------- Distress --------
        self._add_folder("train_cry", label=1)
        self._add_folder("test_cry", label=1)

        self._add_folder("screaming/Screaming", label=1)
        self._add_folder("screaming/NotScreaming", label=0)

        self._collect_ravdess()
        self._collect_besd()
        self._collect_adema()

        # -------- Non-distress --------
        self._add_folder("esc50", label=0)

    def _add_folder(self, rel_path: str, label: int):
        folder = self.root / rel_path
        if not folder.exists():
            return

        for wav in folder.rglob("*.wav"):
            self.samples.append((wav, label))

    # -------------------------------------------------
    # RAVDESS (Actor_xx structure)
    # -------------------------------------------------
    def _collect_ravdess(self):
        ravdess = self.root / "ravdess"
        if not ravdess.exists():
            return

        # RAVDESS emotion codes
        distress_emotions = {"04", "05", "06"}  # sad, angry, fear

        for wav in ravdess.rglob("*.wav"):
            try:
                parts = wav.stem.split("-")
                emotion = parts[2]
                label = 1 if emotion in distress_emotions else 0
                self.samples.append((wav, label))
            except Exception:
                continue

    # -------------------------------------------------
    # BESD (language/emotion folders)
    # -------------------------------------------------
    def _collect_besd(self):
        besd = self.root / "besd"
        if not besd.exists():
            return

        distress = {"ANGER", "FEAR", "SAD"}
        non_distress = {"HAPPY", "NEUTRAL"}

        for lang in besd.iterdir():
            if not lang.is_dir():
                continue

            for emo in lang.iterdir():
                if not emo.is_dir():
                    continue

                if emo.name.upper() in distress:
                    label = 1
                elif emo.name.upper() in non_distress:
                    label = 0
                else:
                    continue

                for wav in emo.rglob("*.wav"):
                    self.samples.append((wav, label))

    # -------------------------------------------------
    # ADEMA / Prima (ALL distress)
    # -------------------------------------------------
    def _collect_adema(self):
        prima = self.root / "Prima"
        if not prima.exists():
            return

        for sub in prima.rglob("*.wav"):
            self.samples.append((sub, 1))

    # -------------------------------------------------
    # Stats
    # -------------------------------------------------
    def _print_stats(self):
        labels = [l for _, l in self.samples]
        print(
            f"[AudioDataset] Total samples: {len(self.samples)} | "
            f"Distress: {sum(labels)} | "
            f"Non-distress: {len(labels) - sum(labels)}"
        )

    # -------------------------------------------------
    # PyTorch API
    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        audio, sr = load_audio(path)
        audio = normalize_audio(audio)

        mfcc = extract_mfcc(audio, sr)  # (39, T)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        return mfcc, torch.tensor(label, dtype=torch.float32)
