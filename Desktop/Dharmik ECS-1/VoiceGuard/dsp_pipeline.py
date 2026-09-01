import numpy as np
from typing import Dict, Any
try:
    from .config import config
except ImportError:
    from config import config

class DSPPipeline:
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self.window_samples = config.audio.window_samples
        self.hop_samples = config.audio.hop_samples
        self.vad_threshold = config.audio.vad_threshold_rms
        self.vad_zcr_min = config.audio.vad_zcr_min
        # O(1) circular buffer — pre-allocated, write head advances modulo capacity
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.write_idx = 0          # next write position (mod window_samples)
        self.filled_samples = 0     # how many samples have been written so far
    def pcm16_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_data.astype(np.float32) / 32768.0

    def append_chunk(self, float_samples: np.ndarray) -> bool:
        """
        O(1) amortised circular-buffer append.
        Writes samples into the pre-allocated ring without shifting existing data.
        """
        chunk_size = len(float_samples)
        capacity = self.window_samples

        if chunk_size >= capacity:
            # Chunk fills or exceeds the whole window — keep only the last `capacity` samples
            self.buffer[:] = float_samples[-capacity:]
            self.write_idx = 0
            self.filled_samples = capacity
        else:
            space_at_end = capacity - self.write_idx
            if chunk_size <= space_at_end:
                # Fits contiguously from write_idx
                self.buffer[self.write_idx: self.write_idx + chunk_size] = float_samples
            else:
                # Wraps around: fill end then beginning
                self.buffer[self.write_idx:] = float_samples[:space_at_end]
                self.buffer[:chunk_size - space_at_end] = float_samples[space_at_end:]
            self.write_idx = (self.write_idx + chunk_size) % capacity
            self.filled_samples = min(capacity, self.filled_samples + chunk_size)

        return self.filled_samples >= capacity

    def compute_vad(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Strict dual-gate VAD:
        - RMS must exceed vad_threshold_rms (0.018) to pass energy gate.
        - ZCR must exceed vad_zcr_min (0.012) to confirm voice-like signal.
        - Ambient hiss/hum (RMS ~0.001-0.010) is rejected as SILENCE.
        """
        if len(audio_chunk) == 0:
            return {'rms': 0.0, 'zcr': 0.0, 'is_speech': False, 'status': 'SILENCE'}

        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2.0 * max(1, len(audio_chunk))))
        
        # Dual gate: both RMS and ZCR must pass to be classified as speech
        is_speech = (rms >= self.vad_threshold) and (zcr >= self.vad_zcr_min)
        return {
            'rms': round(rms, 6),
            'zcr': round(zcr, 6),
            'is_speech': is_speech,
            'status': 'SPEECH_DETECTED' if is_speech else 'SILENCE'
        }

    def get_latest_window(self) -> np.ndarray:
        """
        Reconstruct chronological order from the circular layout.
        Equivalent to the old self.buffer.copy() but accounts for write_idx wrap.
        Returns samples in oldest-to-newest order.
        """
        if self.filled_samples < self.window_samples:
            # Buffer not yet full — return filled portion (no wrap)
            return self.buffer[:self.filled_samples].copy()
        # Full buffer: oldest sample is at write_idx, wrap around
        return np.roll(self.buffer, -self.write_idx).copy()

