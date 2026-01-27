from collections import Counter
from src.audio.dataset import AudioDistressDataset
import random

def main():
    ds = AudioDistressDataset("data/raw/audio")

    print("\n================ DATASET SANITY CHECK ================\n")
    print(f"Total samples: {len(ds)}")

    # -------------------------------------------------
    # 1. Sample labels (not full scan)
    # -------------------------------------------------
    indices = random.sample(range(len(ds)), 200)

    labels = []
    for i in indices:
        _, y = ds[i]
        labels.append(int(y.item()))

    counter = Counter(labels)
    print("Sampled label distribution (200 samples):", counter)

    assert 1 in counter, "❌ No DISTRESS samples found!"
    assert 0 in counter, "❌ No NON-DISTRESS samples found!"

    # -------------------------------------------------
    # 2. Shape check
    # -------------------------------------------------
    print("\nSample preview:")
    for i in indices[:5]:
        x, y = ds[i]
        print(
            f"Shape: {tuple(x.shape)} | Label: {y.item()}"
        )
        assert x.shape[0] == 1
        assert x.shape[1] == 39

    print("\n✅ DATASET SANITY CHECK PASSED (FAST MODE)")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
