import numpy as np
from scipy.ndimage import median_filter
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
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.write_idx = 0
        self.filled_samples = 0

    def pcm16_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_data.astype(np.float32) / 32768.0

    def sanitize_input(self, audio: np.ndarray, bits: int = 8, kernel_size: int = 3) -> np.ndarray:
        """
        Input Sanitization / Feature Squeezing:
        Applies dynamic 8-bit quantization and 1D median filtering to strip
        high-frequency adversarial noise perturbations before scoring.
        """
        if len(audio) == 0:
            return audio
        levels = 2 ** bits
        max_val = float(np.max(np.abs(audio))) + 1e-9
        quantized = np.round((audio / max_val) * (levels / 2.0)) / (levels / 2.0) * max_val
        squeezed = median_filter(quantized, size=kernel_size)
        return squeezed.astype(np.float32)

    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        O(1) Signal-to-Noise Ratio (SNR) estimator on incoming audio frames in dB.
        Calculates SNR based on 90th percentile speech energy vs 10th percentile noise floor.
        """
        if len(audio) < 256:
            return 20.0
        frame_size = 256
        hop = 128
        n_frames = (len(audio) - frame_size) // hop + 1
        if n_frames < 1:
            return 20.0
        frames = np.lib.stride_tricks.sliding_window_view(audio, frame_size)[::hop][:n_frames]
        energies = np.mean(frames ** 2, axis=1) + 1e-9
        signal_pwr = float(np.percentile(energies, 90))
        noise_pwr = float(np.percentile(energies, 10))
        snr_db = 10.0 * np.log10((signal_pwr + 1e-9) / (noise_pwr + 1e-9))
        return float(np.clip(snr_db, 0.0, 60.0))

    def extract_lfcc(self, audio: np.ndarray, num_cep: int = 13, n_filters: int = 20) -> np.ndarray:
        """
        Linear Frequency Cepstral Coefficients (LFCC):
        Uses linearly spaced filterbank across spectrum (superior for vocoder/telephony detection).
        """
        if len(audio) < 256:
            return np.zeros(num_cep, dtype=np.float32)
        n_fft = 512
        stft = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=n_fft))
        freqs = np.linspace(0, self.sample_rate / 2.0, len(stft))
        
        bank_freqs = np.linspace(0, self.sample_rate / 2.0, n_filters + 2)
        fbanks = np.zeros((n_filters, len(stft)), dtype=np.float32)
        for m in range(1, n_filters + 1):
            f_m_minus = bank_freqs[m - 1]
            f_m = bank_freqs[m]
            f_m_plus = bank_freqs[m + 1]
            for k, f in enumerate(freqs):
                if f_m_minus <= f <= f_m:
                    fbanks[m - 1, k] = (f - f_m_minus) / (f_m - f_m_minus + 1e-6)
                elif f_m <= f <= f_m_plus:
                    fbanks[m - 1, k] = (f_m_plus - f) / (f_m_plus - f_m + 1e-6)
                    
        feat = np.dot(fbanks, stft) + 1e-6
        log_feat = np.log(feat)
        n = np.arange(num_cep)
        k = np.arange(n_filters)
        dct_matrix = np.cos(np.pi / n_filters * np.outer(n, k + 0.5))
        lfcc = np.dot(dct_matrix, log_feat)
        return lfcc.astype(np.float32)

    def extract_cqcc(self, audio: np.ndarray, num_cep: int = 13) -> np.ndarray:
        """
        Constant-Q Cepstral Coefficients (CQCC):
        Geometrically spaced Constant-Q log spectrum for detecting pitch/phase cloning anomalies.
        """
        if len(audio) < 256:
            return np.zeros(num_cep, dtype=np.float32)
        n_bins = 24
        fmin = 50.0
        bins_per_oct = 6
        freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_oct))
        stft = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=1024))
        stft_freqs = np.linspace(0, self.sample_rate / 2.0, len(stft))
        
        cq_energies = []
        for f_c in freqs:
            bandwidth = f_c / 12.0
            mask = np.abs(stft_freqs - f_c) <= (bandwidth / 2.0)
            energy = np.sum(stft[mask]) if np.any(mask) else 1e-6
            cq_energies.append(energy)
            
        log_cq = np.log(np.array(cq_energies) + 1e-6)
        n = np.arange(num_cep)
        k = np.arange(n_bins)
        dct_mat = np.cos(np.pi / n_bins * np.outer(n, k + 0.5))
        cqcc = np.dot(dct_mat, log_cq)
        return cqcc.astype(np.float32)

    def append_chunk(self, float_samples: np.ndarray) -> bool:
        chunk_size = len(float_samples)
        capacity = self.window_samples
        if chunk_size >= capacity:
            self.buffer[:] = float_samples[-capacity:]
            self.write_idx = 0
            self.filled_samples = capacity
        else:
            space_at_end = capacity - self.write_idx
            if chunk_size <= space_at_end:
                self.buffer[self.write_idx: self.write_idx + chunk_size] = float_samples
            else:
                self.buffer[self.write_idx:] = float_samples[:space_at_end]
                self.buffer[:chunk_size - space_at_end] = float_samples[space_at_end:]
            self.write_idx = (self.write_idx + chunk_size) % capacity
            self.filled_samples = min(capacity, self.filled_samples + chunk_size)
        return self.filled_samples >= capacity

    def compute_vad(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        if len(audio_chunk) == 0:
            return {'rms': 0.0, 'zcr': 0.0, 'is_speech': False, 'status': 'SILENCE'}
        rms = float(np.sqrt(np.mean(np.square(audio_chunk))))
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2.0 * max(1, len(audio_chunk))))
        is_speech = (rms >= self.vad_threshold) and (zcr >= self.vad_zcr_min)
        return {
            'rms': round(rms, 6),
            'zcr': round(zcr, 6),
            'is_speech': is_speech,
            'status': 'SPEECH_DETECTED' if is_speech else 'SILENCE'
        }

    def get_latest_window(self) -> np.ndarray:
        if self.filled_samples < self.window_samples:
            return self.buffer[:self.filled_samples].copy()
        return np.roll(self.buffer, -self.write_idx).copy()
