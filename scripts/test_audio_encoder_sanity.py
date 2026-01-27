import torch
from src.audio.encoder import AudioCNNEncoder, AudioEncoder

def main():
    print("\n🔍 Audio Encoder Sanity Test\n")

    # Dummy input
    B, F, T = 2, 64, 200
    x = torch.randn(B, 1, F, T, requires_grad=True)

    # -----------------------------
    # 1. Test CNN temporal output
    # -----------------------------
    cnn = AudioCNNEncoder(out_channels=128)
    cnn_out = cnn(x)

    print("CNN output shape:", cnn_out.shape)

    assert cnn_out.ndim == 3, "CNN output should be 3D (B, T, C)"
    assert cnn_out.shape[0] == B, "Batch dimension mismatch"
    assert cnn_out.shape[2] == 128, "Channel dimension mismatch"
    assert cnn_out.shape[1] > 10, "Time dimension collapsed — check pooling"

    print("✅ CNN preserves temporal dimension")

    # -----------------------------
    # 2. Test full audio encoder
    # -----------------------------
    encoder = AudioEncoder(cnn_channels=128, output_dim=256)
    out = encoder(x)

    print("Final encoder output shape:", out.shape)

    assert out.shape == (B, 256), "Final embedding shape incorrect"

    print("✅ Final audio embedding shape correct")

    # -----------------------------
    # 3. Gradient flow check
    # -----------------------------
    loss = out.mean()
    loss.backward()

    grads_ok = all(
        p.grad is not None
        for p in encoder.parameters()
        if p.requires_grad
    )

    assert grads_ok, "Some parameters did not receive gradients"

    print("✅ Gradients flow correctly")

    print("\n🎉 ALL CHECKS PASSED — encoder is correct and safe to use\n")

if __name__ == "__main__":
    main()
