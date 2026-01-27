import numpy as np
from src.audio.encoder import YAMNetExtractor

def main():
    print("\n🔍 YAMNet Sanity Test\n")

    yamnet = YAMNetExtractor()

    # 3 seconds of fake audio @ 16kHz
    audio = np.random.randn(3 * 16000)

    # -----------------------------
    # 1. Default behavior (pool=False)
    # -----------------------------
    emb = yamnet.extract(audio)

    print("Default output shape:", emb.shape)

    assert emb.ndim == 2, "Expected temporal embeddings (T, 1024)"
    assert emb.shape[1] == 1024, "Embedding dim must be 1024"

    print("✅ Temporal YAMNet embeddings preserved")

    # -----------------------------
    # 2. Explicit pooled embeddings
    # -----------------------------
    emb_pooled = yamnet.extract(audio, pool=True)

    print("Pooled output shape:", emb_pooled.shape)

    assert emb_pooled.ndim == 1, "Expected pooled embedding (1024,)"
    assert emb_pooled.shape[0] == 1024, "Embedding dim must be 1024"

    print("✅ Pooled YAMNet embedding works")

    print("\n🎉 ALL YAMNET CHECKS PASSED\n")

if __name__ == "__main__":
    main()
