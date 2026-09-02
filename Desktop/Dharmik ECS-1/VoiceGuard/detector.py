import numpy as np
from typing import Dict, Any, Optional

try:
    from .config import config
    from .ml_classifier import MLVoiceClassifier
    from .dsp_pipeline import DSPPipeline
except ImportError:
    from config import config
    from ml_classifier import MLVoiceClassifier
    from dsp_pipeline import DSPPipeline

class DualLayerDetector:
    def __init__(self, sample_rate: int = 16000, ml_classifier: Optional[MLVoiceClassifier] = None):
        self.sample_rate = sample_rate
        self.ml_classifier = ml_classifier if ml_classifier is not None else MLVoiceClassifier()
        self.dsp = DSPPipeline()

    def analyze_acoustic_branch(self, audio: np.ndarray) -> Dict[str, Any]:
        _zero = {
            'acoustic_score': 0.0, 
            'spectral_flatness': 0.0, 
            'spectral_rolloff_hz': 0.0, 
            'phase_discontinuity_var': 0.0,
            't60_decay_sec': 0.0,
            'replay_decay_score': 0.0,
            '_sub_scores': {'phase_discontinuity': 0.0, 'spectral_flatness': 0.0, 'rolloff': 0.0, 'replay_decay': 0.0}
        }
        vad_thresh = config.audio.vad_threshold_rms
        rms = float(np.sqrt(np.mean(audio**2)))
        if len(audio) < 256 or rms < vad_thresh:
            return _zero

        n_fft = 512
        hop = 128
        n_frames = (len(audio) - n_fft) // hop + 1
        if n_frames < 1:
            return _zero

        window = np.hanning(n_fft)
        frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop][:n_frames]
        stft_complex = np.fft.rfft(frames * window, axis=-1).T  # (freq_bins, n_frames)
        stft_mag = np.abs(stft_complex)

        min_bin = max(1, int(n_fft * 100 / self.sample_rate))
        max_bin = min(stft_mag.shape[0], int(n_fft * 4000 / self.sample_rate))
        speech_mag = stft_mag[min_bin:max_bin, :] + 1e-6

        log_mag = np.log(speech_mag)
        geom_mean = np.exp(np.mean(log_mag, axis=0))
        arith_mean = np.mean(speech_mag, axis=0)
        spectral_flatness = float(np.mean(geom_mean / arith_mean))

        power = stft_mag ** 2
        total_power = np.sum(power, axis=0, keepdims=True)
        cum_power = np.cumsum(power, axis=0)
        freqs = np.linspace(0, self.sample_rate / 2.0, stft_mag.shape[0])
        idx = np.argmax(cum_power >= 0.85 * (total_power + 1e-9), axis=0)
        rolloff = float(np.mean(freqs[idx]))

        mask = stft_mag > (0.01 * np.max(stft_mag))
        angles = np.angle(stft_complex)
        phase_diff = np.diff(angles, axis=1)
        valid_diffs = phase_diff[mask[:, :-1]] if mask.shape[1] > 1 else np.array([])
        phase_discontinuity = float(np.var(np.angle(np.exp(1j * valid_diffs)))) if valid_diffs.size > 0 else 0.0

        # ── T60 High-Frequency Energy Decay (Replay Attack Detector) ───────────────────
        # Approximates room impulse response decay time in 4kHz-8kHz band
        hf_mask = freqs >= 4000.0
        if np.any(hf_mask) and stft_mag.shape[1] > 2:
            hf_energy = np.mean(stft_mag[hf_mask, :], axis=0)
            peak_idx = np.argmax(hf_energy)
            decay_energy = hf_energy[peak_idx:]
            if len(decay_energy) > 2 and decay_energy[0] > 1e-5:
                # Time to drop by 60dB approximated from initial decay slope
                energy_ratio = decay_energy / (decay_energy[0] + 1e-9)
                decay_rate = float(-np.mean(np.diff(np.log(energy_ratio + 1e-6))))
                t60_estimate = float(np.clip(0.05 / (decay_rate + 1e-6), 0.05, 1.5))
            else:
                t60_estimate = 0.15
        else:
            t60_estimate = 0.15

        # Replay attack score: excessive reverberation decay (T60 > 0.6s) or sharp unnatural decay (<0.08s)
        if t60_estimate > 0.60:
            replay_score = min(1.0, (t60_estimate - 0.60) / 0.60)
        elif t60_estimate < 0.08:
            replay_score = min(1.0, (0.08 - t60_estimate) / 0.08)
        else:
            replay_score = 0.0

        flatness_score = 0.0 if spectral_flatness < 0.025 else min(1.0, (spectral_flatness - 0.025) / 0.06)
        phase_score = 0.0 if phase_discontinuity < 2.0 else min(1.0, (phase_discontinuity - 2.0) / 3.0)

        if rolloff < 800.0:
            rolloff_score = min(1.0, (800.0 - rolloff) / 600.0)
        elif rolloff > 5500.0:
            rolloff_score = min(1.0, (rolloff - 5500.0) / 2000.0)
        else:
            rolloff_score = 0.0

        acoustic_score = float(np.clip(0.40 * phase_score + 0.30 * flatness_score + 0.15 * rolloff_score + 0.15 * replay_score, 0.0, 1.0))
        return {
            'acoustic_score': round(acoustic_score * 100.0, 2),
            'spectral_flatness': round(spectral_flatness, 6),
            'spectral_rolloff_hz': round(rolloff, 2),
            'phase_discontinuity_var': round(phase_discontinuity, 6),
            't60_decay_sec': round(t60_estimate, 4),
            'replay_decay_score': round(float(replay_score * 100.0), 2),
            '_sub_scores': {
                'phase_discontinuity': round(float(phase_score), 4),
                'spectral_flatness': round(float(flatness_score), 4),
                'rolloff': round(float(rolloff_score), 4),
                'replay_decay': round(float(replay_score), 4)
            }
        }

    def analyze_prosodic_branch(self, audio: np.ndarray) -> Dict[str, Any]:
        _zero = {
            'prosodic_score': 0.0, 
            'f0_mean_hz': 0.0, 
            'f0_std_hz': 0.0, 
            'jitter': 0.0, 
            'shimmer': 0.0,
            '_sub_scores': {'flat_pitch': 0.0, 'jitter': 0.0, 'shimmer': 0.0}
        }
        vad_thresh = config.audio.vad_threshold_rms
        rms_full = float(np.sqrt(np.mean(audio**2)))

        if len(audio) < 256 or rms_full < vad_thresh:
            return _zero

        frame_length = 1024
        hop_length = 512
        n_frames = (len(audio) - frame_length) // hop_length + 1
        if n_frames < 1:
            return _zero

        f0_list = []
        rms_list = []
        min_lag = int(self.sample_rate / 400)
        max_lag = int(self.sample_rate / 65)

        for i in range(n_frames):
            sub = audio[i * hop_length: i * hop_length + frame_length]
            frame_rms = float(np.sqrt(np.mean(sub**2)))
            rms_list.append(frame_rms)
            if frame_rms < vad_thresh:
                continue
            n = len(sub)
            f = np.fft.rfft(sub, n=2 * n)
            corr = np.fft.irfft(f * np.conj(f))[:n]
            if corr[0] <= 1e-9:
                continue
            norm_corr = corr / corr[0]
            if max_lag < len(norm_corr):
                peak_idx = np.argmax(norm_corr[min_lag:max_lag]) + min_lag
                if norm_corr[peak_idx] > 0.25:
                    f0 = self.sample_rate / peak_idx
                    f0_list.append(f0)

        f0_arr = np.array(f0_list, dtype=np.float32)
        rms_arr = np.array(rms_list, dtype=np.float32)

        if len(f0_arr) <= 3:
            if rms_full >= vad_thresh * 2.0:
                return {
                    'prosodic_score': round(20.0, 2),
                    'f0_mean_hz': 0.0,
                    'f0_std_hz': 0.0,
                    'jitter': 0.0,
                    'shimmer': 0.0,
                    '_sub_scores': {'flat_pitch': 0.20, 'jitter': 0.0, 'shimmer': 0.0}
                }
            return _zero

        median_f0 = float(np.median(f0_arr))
        if median_f0 > 0:
            doubled = f0_arr > (median_f0 * 1.7)
            halved  = f0_arr < (median_f0 * 0.55)
            f0_arr[doubled] /= 2.0
            f0_arr[halved]  *= 2.0

        if len(f0_arr) >= 3:
            f0_arr = np.array(
                [float(np.median(f0_arr[max(0, i - 1): i + 2])) for i in range(len(f0_arr))],
                dtype=np.float32
            )

        f0_mean = float(np.mean(f0_arr))
        f0_std = float(np.std(f0_arr))
        jitter = float(np.mean(np.abs(np.diff(f0_arr))) / (f0_mean + 1e-6))
        shimmer = float(np.std(rms_arr) / (np.mean(rms_arr) + 1e-6))

        flat_pitch_score = 0.0 if f0_std >= 10.0 else min(1.0, (10.0 - f0_std) / 10.0)
        jitter_score = 0.0 if jitter >= 0.003 else min(1.0, (0.003 - jitter) / 0.003)
        shimmer_score = 0.0 if shimmer >= 0.008 else min(1.0, (0.008 - shimmer) / 0.008)
        prosodic_score = float(np.clip(0.60 * flat_pitch_score + 0.25 * jitter_score + 0.15 * shimmer_score, 0.0, 1.0))
        return {
            'prosodic_score': round(prosodic_score * 100.0, 2),
            'f0_mean_hz': round(f0_mean, 2),
            'f0_std_hz': round(f0_std, 2),
            'jitter': round(jitter, 6),
            'shimmer': round(shimmer, 6),
            '_sub_scores': {
                'flat_pitch': round(float(flat_pitch_score), 4),
                'jitter': round(float(jitter_score), 4),
                'shimmer': round(float(shimmer_score), 4)
            }
        }

    def process_window(self, audio_window: np.ndarray) -> Dict[str, Any]:
        # 1. Feature Squeezing / Input Sanitization to strip adversarial perturbations
        sanitized_window = self.dsp.sanitize_input(audio_window)
        
        # 2. SNR & Channel Estimation for Adaptive Score Fusion
        snr_db = self.dsp.estimate_snr(audio_window)
        
        # Adaptive Weights Assignment
        if snr_db < 15.0 or self.sample_rate < 16000:
            # Degradation / Lossy Telephony mode: rely heavily on robust LFCC/DSP features
            w_ml, w_ac, w_pr = 0.40, 0.30, 0.30
            fusion_mode = "ADAPTIVE_DSP_HEAVY"
        else:
            # Clean High-Fidelity Channel mode
            w_ml, w_ac, w_pr = 0.60, 0.20, 0.20
            fusion_mode = "STANDARD_ML_HEAVY"

        acoustic_res = self.analyze_acoustic_branch(sanitized_window)
        prosodic_res = self.analyze_prosodic_branch(sanitized_window)
        ml_score = self.ml_classifier.predict_frame(sanitized_window, sample_rate=self.sample_rate)

        ac_score = acoustic_res['acoustic_score']
        pr_score = prosodic_res['prosodic_score']

        # Dynamic Weighted Score Fusion
        raw_frame_score = round(w_ml * ml_score + w_ac * ac_score + w_pr * pr_score, 2)

        ac_subs = acoustic_res.get('_sub_scores', {'phase_discontinuity': 0.0, 'spectral_flatness': 0.0, 'rolloff': 0.0, 'replay_decay': 0.0})
        pr_subs = prosodic_res.get('_sub_scores', {'flat_pitch': 0.0, 'jitter': 0.0, 'shimmer': 0.0})

        weighted_factors = {
            'ml.neural_vocoder_prob': round(ml_score * w_ml / 100.0, 4),
            'acoustic.phase_discontinuity': round(ac_subs['phase_discontinuity'] * w_ac, 4),
            'acoustic.spectral_flatness': round(ac_subs['spectral_flatness'] * w_ac * 0.35, 4),
            'prosodic.flat_pitch': round(pr_subs['flat_pitch'] * w_pr * 0.60, 4)
        }
        max_val = max(weighted_factors.values()) if weighted_factors else 0.0
        dominant_signal = max(weighted_factors, key=weighted_factors.get) if max_val > 0.01 else "none"

        contributing_factors = {
            'ml_score': ml_score,
            'acoustic': ac_subs,
            'prosodic': pr_subs,
            'snr_db': round(snr_db, 2),
            'fusion_mode': fusion_mode,
            'weights': {'ml': w_ml, 'acoustic': w_ac, 'prosodic': w_pr},
            'dominant_signal': dominant_signal
        }

        return {
            'raw_frame_score': raw_frame_score,
            'ml_score': ml_score,
            'snr_db': round(snr_db, 2),
            'fusion_mode': fusion_mode,
            'acoustic': {k: v for k, v in acoustic_res.items() if k != '_sub_scores'},
            'prosodic': {k: v for k, v in prosodic_res.items() if k != '_sub_scores'},
            'contributing_factors': contributing_factors
        }
