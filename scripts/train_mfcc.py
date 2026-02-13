"""
Train audio-only distress classifier using precomputed MFCC features.
This is the FINAL MFCC-based training script.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from collections import Counter

from src.audio.dataset import AudioDistressDataset
from src.audio.encoder import AudioEncoder
from src.audio.classifier import AudioDistressClassifier


# -------------------------------------------------
# Padding collate fn (for variable-length MFCCs)
# -------------------------------------------------
def pad_collate_fn(batch):
    features, labels = zip(*batch)

    max_len = max(x.shape[-1] for x in features)

    padded = []
    for x in features:
        pad_amt = max_len - x.shape[-1]
        padded.append(F.pad(x, (0, pad_amt)))

    features = torch.stack(padded)   # (B, 1, 39, T)
    labels = torch.stack(labels)     # (B,)

    return features, labels


# -------------------------------------------------
# Args
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("MFCC Audio Distress Training")
    parser.add_argument("--data-root", type=str, default="data/raw/audio")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


# -------------------------------------------------
# Train epoch
# -------------------------------------------------
def train_epoch(encoder, classifier, loader, criterion, optimizer, device):
    encoder.train()
    classifier.train()

    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        emb = encoder(x)
        logits = classifier(emb)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Validation epoch
# -------------------------------------------------
@torch.no_grad()
def validate_epoch(encoder, classifier, loader, criterion, device):
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        emb = encoder(x)
        logits = classifier(emb)

        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds.append((probs > 0.5).float().cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    acc = (preds == targets).float().mean().item()
    recall = (preds * targets).sum() / (targets.sum() + 1e-6)

    return total_loss / len(loader), acc, recall.item()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("CurioNext | MFCC Audio Distress Training")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print("=" * 60)

    # Dataset
    dataset = AudioDistressDataset(args.data_root)

    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    print("Label distribution:", Counter(labels))

    # Stratified 80/20 split
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    np.random.shuffle(idx_pos)
    np.random.shuffle(idx_neg)

    split_p = int(0.8 * len(idx_pos))
    split_n = int(0.8 * len(idx_neg))

    train_idx = np.concatenate([idx_pos[:split_p], idx_neg[:split_n]])
    val_idx   = np.concatenate([idx_pos[split_p:], idx_neg[split_n:]])

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    # Models
    encoder = AudioEncoder().to(device)
    classifier = AudioDistressClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            encoder, classifier, train_loader,
            criterion, optimizer, device
        )

        val_loss, val_acc, val_recall = validate_epoch(
            encoder, classifier, val_loader,
            criterion, device
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Recall: {val_recall:.4f}"
        )

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
        },
        "checkpoints/audio_mfcc_model.pt",
    )

    print("\n💾 Model saved to checkpoints/audio_mfcc_model.pt")
    print("✅ MFCC-based audio training complete")


if __name__ == "__main__":
    main()
