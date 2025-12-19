"""Tests for audio processing module."""
import pytest
import numpy as np
from src.audio import preprocessing, features, encoder
import soundfile as sf

class TestAudioPreprocessing:
    """Test audio preprocessing functions."""

    def test_load_audio(self,tmp_path):
        """Test audio loading."""
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))

        file_path = tmp_path / "test.wav"
        sf.write(file_path, audio, sr)

        loaded_audio, loaded_sr = preprocessing.load_audio(str(file_path))

        assert loaded_sr == sr
        assert loaded_audio.ndim == 1
        assert loaded_audio.size > 0

    def test_load_audio_missing_file(self):
        with pytest.raises(FileNotFoundError):
            preprocessing.load_audio("missing.wav")

    def test_load_audio_corrupt_file(self, tmp_path):
        file_path = tmp_path / "corrupt.wav"
        file_path.write_text("not audio")

        with pytest.raises(ValueError):
            preprocessing.load_audio(str(file_path))

    def test_normalize_audio_peak(self):
        """Test peak normalization."""
        audio = np.array([0.2, -0.5, 0.5])
        normalized = preprocessing.normalize_audio(audio, method="peak")
        assert np.max(np.abs(normalized)) == pytest.approx(1.0)

    def test_normalize_audio_rms(self):
        """Test RMS normalization."""
        audio = np.array([0.2, -0.5, 0.5])
        normalized = preprocessing.normalize_audio(audio, method="rms")
        rms = np.sqrt(np.mean(normalized ** 2))
        assert rms == pytest.approx(1.0, rel=1e-2)

    def test_trim_silence(self):
        """Test silence trimming."""
        audio = np.concatenate([np.zeros(3000), np.ones(5000)])
        trimmed = preprocessing.trim_silence(audio, sr=16000)
        assert trimmed.size < audio.size

class TestAudioFeatures:
    """Test audio feature extraction."""

    def test_extract_mfcc(self):
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = 0.5 * np.sin(2 * np.pi * 220 * t)

        mfcc = features.extract_mfcc(audio, sr)

        assert isinstance(mfcc, np.ndarray)
        assert mfcc.shape == (39,)

    def test_extract_spectral_features(self):
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 220 * t)

        spectral = features.extract_spectral_features(audio, sr)

        assert isinstance(spectral, dict)
        assert "mel_spectrogram" in spectral
        assert "chroma" in spectral
        assert "spectral_centroid" in spectral
        assert "spectral_rolloff" in spectral
        assert "spectral_contrast" in spectral

        assert spectral["mel_spectrogram"].shape == (64,)
        assert spectral["chroma"].shape == (12,)


@pytest.mark.skip(reason="YAMNet encoder not implemented yet")
class TestYAMNetEncoder:
    """Test YAMNet encoder."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        # TODO: Implement test
        pass

    def test_encoder_forward_pass(self):
        """Test forward pass."""
        # TODO: Implement test
        pass

    def test_embedding_dimension(self):
        """Test output embedding dimension is 1024."""
        # TODO: Implement test
        pass
