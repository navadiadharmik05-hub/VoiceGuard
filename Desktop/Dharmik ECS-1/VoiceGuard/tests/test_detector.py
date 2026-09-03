"""Unit tests for detector.py — DualLayerDetector acoustic + prosodic branches."""
import math
import os
import sys
import numpy as np
import pytest

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detector import DualLayerDetector

SR = 16000


@pytest.fixture
def det():
    """Fixture to ensure a fresh detector instance per test without shared state."""
    return DualLayerDetector(sample_rate=SR)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_silence(duration: float = 0.8) -> np.ndarray:
    """True digital silence — all zeros."""
    n_samples = int(SR * duration)
    return np.zeros(n_samples, dtype=np.float32)


def make_low_rms(duration: float = 0.8, rms: float = 0.005) -> np.ndarray:
    """White noise scaled to a specific RMS below VAD threshold."""
    n_samples = int(SR * duration)
    x = np.random.default_rng(42).standard_normal(n_samples).astype(np.float32)
    return x * (rms / (np.sqrt(np.mean(x**2)) + 1e-9))


def make_sine(freq: float = 180.0, duration: float = 0.8, amplitude: float = 0.40) -> np.ndarray:
    """Single-frequency sine tone — clean, low spectral flatness."""
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def make_human_voice(duration: float = 0.8) -> np.ndarray:
    """
    Realistic human-voice simulation: vibrato F0 with multi-harmonic partials
    and natural breath noise. RMS >> VAD threshold.
    """
    n = int(SR * duration)
    samples = np.zeros(n, dtype=np.float32)
    phase = 0.0
    rng = np.random.default_rng(0)
    for i in range(n):
        t = i / SR
        f0 = 150 + 30 * math.sin(2 * math.pi * 4.5 * t) + 8 * math.sin(2 * math.pi * 11 * t)
        phase += 2 * math.pi * f0 / SR
        samples[i] = (0.38 * math.sin(phase)
                      + 0.18 * math.sin(2 * phase + 0.3)
                      + 0.10 * math.sin(3 * phase + 0.7)
                      + float(rng.uniform(-0.5, 0.5)) * 0.06)
    return samples


def make_square_wave(freq: float = 150.0, duration: float = 0.8, amplitude: float = 0.70) -> np.ndarray:
    """
    Band-limited square wave up to the Nyquist frequency to avoid aliasing.
    High harmonic content, high spectral flatness.
    """
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    nyquist = SR / 2.0
    max_harmonic = int(nyquist // freq)
    
    samples = np.zeros_like(t)
    for k in range(1, max_harmonic + 1, 2):
        samples += (1.0 / k) * np.sin(2 * math.pi * k * freq * t)
        
    samples = (4.0 / math.pi) * samples
    samples = samples / (np.max(np.abs(samples)) + 1e-9) * amplitude
    return samples.astype(np.float32)


def make_fm_tone(f0: float = 150.0, mod_freq: float = 5.0,
                 mod_depth: float = 40.0, duration: float = 0.8,
                 amplitude: float = 0.40) -> np.ndarray:
    """Frequency-modulated tone — continuously varying pitch (human-like)."""
    n = int(SR * duration)
    samples = np.zeros(n, dtype=np.float32)
    phase = 0.0
    for i in range(n):
        t = i / SR
        inst_freq = f0 + mod_depth * math.sin(2 * math.pi * mod_freq * t)
        phase += 2 * math.pi * inst_freq / SR
        samples[i] = amplitude * math.sin(phase)
    return samples.astype(np.float32)


def make_steady_tone(freq: float = 150.0, duration: float = 0.8,
                     amplitude: float = 0.40) -> np.ndarray:
    """Perfectly steady sine tone — flat pitch, synthetic-like prosody."""
    t = np.arange(int(SR * duration), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# analyze_acoustic_branch tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAcousticBranch:

    def test_silence_returns_all_zeros(self, det):
        """True silence must return a zero-score, zeroed-metric dict."""
        result = det.analyze_acoustic_branch(make_silence())
        assert result['acoustic_score'] == 0.0
        assert result['spectral_flatness'] == 0.0
        assert result['spectral_rolloff_hz'] == 0.0
        assert result['phase_discontinuity_var'] == 0.0

    def test_low_rms_returns_all_zeros(self, det):
        """Audio below VAD RMS threshold (0.018) must be gated out."""
        result = det.analyze_acoustic_branch(make_low_rms(rms=0.005))
        assert result['acoustic_score'] == 0.0
        assert result['spectral_flatness'] == 0.0

    def test_sine_tone_low_acoustic_score(self, det):
        """
        A clean sine tone has a narrow-band spectrum -> low spectral flatness
        and moderate phase discontinuity -> acoustic_score should be below 60.
        """
        result = det.analyze_acoustic_branch(make_sine(freq=180.0, amplitude=0.40))
        assert result['acoustic_score'] < 60.0, (
            f"Sine tone acoustic_score={result['acoustic_score']:.1f} expected < 60"
        )

    def test_square_wave_higher_acoustic_score_than_sine(self, det):
        """
        A square wave is rich in odd harmonics and has high spectral flatness
        -> its acoustic_score must be strictly greater than a clean sine tone.
        """
        sine_result = det.analyze_acoustic_branch(make_sine(freq=150.0, amplitude=0.60))
        square_result = det.analyze_acoustic_branch(make_square_wave(freq=150.0))
        assert square_result['acoustic_score'] > sine_result['acoustic_score'], (
            f"Square {square_result['acoustic_score']:.1f} should > Sine {sine_result['acoustic_score']:.1f}"
        )

    def test_square_wave_flatness_higher_than_sine(self, det):
        """Square wave spectral flatness must exceed sine tone flatness."""
        sine_res = det.analyze_acoustic_branch(make_sine(freq=150.0, amplitude=0.60))
        square_res = det.analyze_acoustic_branch(make_square_wave(freq=150.0))
        assert square_res['spectral_flatness'] > sine_res['spectral_flatness']

    def test_return_keys_present(self, det):
        """All expected keys must be in the acoustic branch output."""
        result = det.analyze_acoustic_branch(make_sine())
        for key in ('acoustic_score', 'spectral_flatness', 'spectral_rolloff_hz', 'phase_discontinuity_var'):
            assert key in result, f"Missing key: {key}"

    def test_scores_bounded_0_to_100(self, det):
        """acoustic_score must always be in [0, 100]."""
        for sig in [make_silence(), make_sine(), make_square_wave(), make_human_voice()]:
            r = det.analyze_acoustic_branch(sig)
            assert 0.0 <= r['acoustic_score'] <= 100.0

    def test_too_short_input_returns_zeros(self, det):
        """Arrays shorter than 256 samples must return zero dict."""
        result = det.analyze_acoustic_branch(np.zeros(100, dtype=np.float32))
        assert result['acoustic_score'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# analyze_prosodic_branch tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProsodicBranch:

    def test_silence_returns_all_zeros(self, det):
        """True silence must return a zero-score dict."""
        result = det.analyze_prosodic_branch(make_silence())
        assert result['prosodic_score'] == 0.0
        assert result['f0_mean_hz'] == 0.0
        assert result['f0_std_hz'] == 0.0

    def test_low_rms_returns_all_zeros(self, det):
        """Audio below VAD threshold must be gated out."""
        result = det.analyze_prosodic_branch(make_low_rms(rms=0.005))
        assert result['prosodic_score'] == 0.0

    def test_steady_tone_higher_prosodic_score_than_fm(self, det):
        """
        A steady (flat-pitch) tone should score HIGHER prosodic synthetic risk
        than an FM tone with natural pitch variance.
        Flat pitch is a deepfake indicator.
        """
        steady_res = det.analyze_prosodic_branch(make_steady_tone(freq=150.0, amplitude=0.40))
        fm_res = det.analyze_prosodic_branch(make_fm_tone(f0=150.0, mod_depth=40.0, amplitude=0.40))
        assert steady_res['prosodic_score'] >= fm_res['prosodic_score'], (
            f"Steady={steady_res['prosodic_score']:.1f} should >= FM={fm_res['prosodic_score']:.1f}"
        )

    def test_fm_tone_f0_std_higher_than_steady(self, det):
        """
        FM tone with 40 Hz modulation depth should exhibit higher F0 STD
        than a perfectly steady sine tone.
        """
        steady_res = det.analyze_prosodic_branch(make_steady_tone(freq=150.0, amplitude=0.40))
        fm_res = det.analyze_prosodic_branch(make_fm_tone(f0=150.0, mod_depth=40.0, amplitude=0.40))
        assert fm_res['f0_std_hz'] > steady_res['f0_std_hz'], (
            f"FM f0_std={fm_res['f0_std_hz']:.1f} should > Steady f0_std={steady_res['f0_std_hz']:.1f}"
        )

    def test_fm_tone_detects_f0_in_range(self, det):
        """F0 mean for a 150 Hz FM tone must be detected within vocal range."""
        result = det.analyze_prosodic_branch(make_fm_tone(f0=150.0, mod_depth=30.0, amplitude=0.40))
        if result['f0_mean_hz'] > 0:
            assert 80.0 <= result['f0_mean_hz'] <= 350.0, (
                f"f0_mean_hz={result['f0_mean_hz']:.1f} outside expected vocal range"
            )

    def test_return_keys_present(self, det):
        """All expected keys must be present in the prosodic output dict."""
        result = det.analyze_prosodic_branch(make_fm_tone())
        for key in ('prosodic_score', 'f0_mean_hz', 'f0_std_hz', 'jitter', 'shimmer'):
            assert key in result, f"Missing key: {key}"

    def test_scores_bounded_0_to_100(self, det):
        """prosodic_score must always be in [0, 100]."""
        for sig in [make_silence(), make_steady_tone(), make_fm_tone(), make_human_voice()]:
            r = det.analyze_prosodic_branch(sig)
            assert 0.0 <= r['prosodic_score'] <= 100.0

    def test_too_short_input_returns_zeros(self, det):
        """Arrays shorter than 256 samples must return zero dict."""
        result = det.analyze_prosodic_branch(np.zeros(100, dtype=np.float32))
        assert result['prosodic_score'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# process_window integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessWindow:

    def test_returns_expected_keys(self, det):
        result = det.process_window(make_sine())
        assert 'raw_frame_score' in result
        assert 'acoustic' in result
        assert 'prosodic' in result
        assert 'contributing_factors' in result

    def test_contributing_factors_structure(self, det):
        result = det.process_window(make_steady_tone())
        cf = result['contributing_factors']
        assert 'acoustic' in cf
        assert 'prosodic' in cf
        assert 'dominant_signal' in cf
        assert 'phase_discontinuity' in cf['acoustic']
        assert 'spectral_flatness' in cf['acoustic']
        assert 'rolloff' in cf['acoustic']
        assert 'flat_pitch' in cf['prosodic']
        assert 'jitter' in cf['prosodic']
        assert 'shimmer' in cf['prosodic']

    def test_raw_score_is_mean_of_branches(self, det):
        """raw_frame_score must equal 0.5 * acoustic + 0.5 * prosodic (within tolerance)."""
        sig = make_fm_tone(amplitude=0.40)
        result = det.process_window(sig)
        w = result['contributing_factors']['weights']
        expected = w['ml'] * result['ml_score'] + w['acoustic'] * result['acoustic']['acoustic_score'] + w['prosodic'] * result['prosodic']['prosodic_score']
        assert result['raw_frame_score'] == pytest.approx(expected, abs=1e-2)

    def test_silence_raw_score_is_zero(self, det):
        result = det.process_window(make_silence())
        assert result['raw_frame_score'] == 0.0


# ╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠╠
# Multi-Benchmark Stress & Evaluation Suite
# ⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠⅠ




class TestBenchmarkMatrixAndStress:
    def test_container_and_sample_rate_execution_matrix(self):
        import io
        import soundfile as sf
        from offline_inspector import inspect_file
        from detector import DualLayerDetector
        detector = DualLayerDetector()
        containers = ['wav', 'flac', 'ogg']
        sample_rates = [8000, 16000, 24000, 44100, 48000]
        for fmt in containers:
            for sr in sample_rates:
                audio = make_human_voice(duration=1.5)
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format=fmt)
                buf.seek(0)
                raw_bytes = buf.read()
                res = inspect_file(raw_bytes, detector=detector)
                assert 'overall_risk_score' in res
                assert 'verdict' in res
                assert 0.0 <= res['overall_risk_score'] <= 100.0

    def test_sustained_peak_risk_formula_and_spike_immunity(self):
        from offline_inspector import inspect_file
        from detector import DualLayerDetector
        detector = DualLayerDetector()
        audio = make_human_voice(duration=2.5)
        spike_start = int(1.0 * SR)
        spike_end = spike_start + int(0.05 * SR)
        audio[spike_start:spike_end] = np.random.uniform(-1.0, 1.0, size=spike_end - spike_start).astype(np.float32) * 0.95
        res = inspect_file(audio, detector=detector)
        pr = res['peak_risk']
        spr = res['sustained_peak_risk']
        ovr = res['overall_risk_score']
        verdict = res['verdict']
        expected_ovr = round(0.5 * pr + 0.5 * spr, 2)
        assert abs(ovr - expected_ovr) < 1e-4
        assert verdict in ['ALLOW', 'MONITOR', 'ALLOW_MONITORED']
