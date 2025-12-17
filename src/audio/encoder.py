"""Audio encoder using YAMNet."""
import tensorflow_hub as hub
import numpy as np
from typing import Optional
import tensorflow as tf

class YAMNetExtractor:
    """
    YAMNet-based audio encoder for extracting embeddings.

    YAMNet outputs 1024-dimensional embeddings from audio.
    """

    def __init__(self, freeze_layers: int = 5):
        """
        Load pretrained YAMNet model from TensorFlow Hub.
        """
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract audio embeddings using YAMNet.

        Args:
            audio: Mono audio waveform (1D numpy array)
            sr: Sample rate (must be 16000 Hz)

        Returns:
            1024-dimensional embedding vector
        """
        if audio.size == 0:
            raise ValueError("Cannot extract YAMNet embeddings from empty audio")

        if audio.ndim != 1:
            raise ValueError("YAMNet expects mono audio input")

        if sr != 16000:
            raise ValueError("YAMNet requires 16 kHz audio")

         # Convert audio to TensorFlow tensor
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

        # Run YAMNet
        scores, embeddings, spectrogram = self.model(audio_tensor)

        # embeddings shape: (num_frames, 1024)
        # Mean pooling across time to get fixed-size vector
        embedding = tf.reduce_mean(embeddings, axis=0)

        return embedding.numpy()