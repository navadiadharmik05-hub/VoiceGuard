"""Unit tests for risk_engine.py — DynamicRiskEngine EWMA, decay, thresholds, reset."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk_engine import DynamicRiskEngine, _WARMUP_FRAMES
from config import config


def fresh() -> DynamicRiskEngine:
    """Return a freshly initialised engine for each test."""
    return DynamicRiskEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA update — consecutive speech frames
# ═══════════════════════════════════════════════════════════════════════════════

class TestEWMAUpdate:

    def test_cold_start_first_frame_is_low(self):
        """
        On frame 1 (warmup alpha ≈ 0.08), smoothed_score must be much lower
        than raw_frame_score (no instant spike to raw value).
        """
        eng = fresh()
        raw = 50.0
        result = eng.update_risk(raw, is_speech=True)
        # ramp_alpha on frame 1 = 0.08 + (0.30-0.08)*(1/5) = 0.124
        # smoothed = 0.124 * 50 = 6.2  (was 50.0 before fix)
        assert result['dynamic_risk_score'] < 20.0, (
            f"Cold-start spike: got {result['dynamic_risk_score']:.1f}, expected < 20"
        )

    def test_score_increases_over_sustained_speech(self):
        """EWMA score must monotonically increase when fed a constant high raw score."""
        eng = fresh()
        scores = [eng.update_risk(80.0, True)['dynamic_risk_score'] for _ in range(15)]
        assert scores == sorted(scores), f"Scores not monotonically increasing: {scores}"

    def test_ewma_converges_toward_raw_score(self):
        """
        After many frames of the same raw_score, EWMA must converge
        within 5% of that raw score.
        """
        eng = fresh()
        target = 60.0
        for _ in range(60):
            eng.update_risk(target, True)
        final = eng.update_risk(target, True)['dynamic_risk_score']
        assert abs(final - target) < 5.0, (
            f"EWMA did not converge: final={final:.1f}, target={target}"
        )

    def test_warmup_alpha_ramps_correctly(self):
        """
        Verify alpha ramp: frame k gets alpha = 0.08 + (full_alpha - 0.08) * (k / WARMUP_FRAMES).
        After WARMUP_FRAMES, the engine must use full alpha.
        """
        eng = fresh()
        full_alpha = config.risk.ewma_alpha
        assert eng._ramp_alpha() == pytest.approx(0.08, abs=1e-6)  # before first speech frame
        # Feed WARMUP_FRAMES speech frames
        for _ in range(_WARMUP_FRAMES):
            eng._speech_frame_count += 1  # simulate frame count without side-effects
        assert eng._ramp_alpha() == pytest.approx(full_alpha, abs=1e-6)

    def test_peak_score_tracks_maximum(self):
        """peak_risk_score must reflect the highest score seen across all frames."""
        eng = fresh()
        eng._speech_frame_count = _WARMUP_FRAMES  # skip warmup for direct EWMA testing
        for _ in range(10):
            eng.update_risk(80.0, True)
        peak_high = eng.peak_score
        # Now feed lower score
        r = eng.update_risk(10.0, True)
        assert r['peak_risk_score'] == pytest.approx(round(peak_high, 2), abs=1e-2)

    def test_history_trend_length_capped(self):
        """history_trend must not grow beyond buffer_history_size."""
        eng = fresh()
        limit = config.risk.buffer_history_size
        for i in range(limit + 20):
            r = eng.update_risk(50.0, True)
        assert len(r['history_trend']) <= limit


# ═══════════════════════════════════════════════════════════════════════════════
# Silence decay behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestSilenceDecay:

    def test_score_decays_toward_zero_on_silence(self):
        """
        After seeding a high score, consecutive silence frames must drive
        the EWMA score toward 0.
        """
        eng = fresh()
        # Seed a high score over many speech frames
        for _ in range(30):
            eng.update_risk(80.0, True)
        seeded = eng.smoothed_score
        assert seeded > 30.0  # confirm seed worked

        # Now run 10 silence frames
        for _ in range(10):
            eng.update_risk(0.0, False)
        assert eng.smoothed_score < seeded, "Score did not decay on silence"

    def test_score_reaches_zero_after_enough_silence(self):
        """
        After enough silence frames, the score must snap to exactly 0.0
        (hard floor at 0.5 triggers snap).
        """
        eng = fresh()
        for _ in range(20):
            eng.update_risk(70.0, True)
        # Run many silence frames
        for _ in range(50):
            eng.update_risk(0.0, False)
        assert eng.smoothed_score == 0.0

    def test_silence_decay_factor_applied_per_frame(self):
        """
        Each silence frame multiplies smoothed_score by silence_decay_factor.
        Verify the first decay step precisely.
        """
        eng = fresh()
        # Bypass warmup: set smoothed_score directly
        eng.smoothed_score = 50.0
        eng._silence_frames = 0
        decay = config.audio.silence_decay_factor

        eng.update_risk(0.0, False)
        expected = 50.0 * decay
        if expected < 0.5:
            expected = 0.0
        assert eng.smoothed_score == pytest.approx(expected, abs=0.01)

    def test_silence_frame_counter_increments(self):
        """_silence_frames must increment on each non-speech frame."""
        eng = fresh()
        for i in range(1, 6):
            eng.update_risk(0.0, False)
            assert eng._silence_frames == i

    def test_silence_counter_resets_on_speech(self):
        """_silence_frames must reset to 0 when speech resumes."""
        eng = fresh()
        for _ in range(5):
            eng.update_risk(0.0, False)
        assert eng._silence_frames == 5
        eng.update_risk(40.0, True)
        assert eng._silence_frames == 0


# ═══════════════════════════════════════════════════════════════════════════════
# risk_level / action_trigger threshold boundary mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdMapping:
    """
    Thresholds from config:
        LOW      < 30.0
        ELEVATED < 65.0
        SUSPICIOUS < 85.0
        CRITICAL >= 85.0
    """

    def _set_score(self, eng: DynamicRiskEngine, score: float) -> dict:
        """Directly set smoothed_score and call update to get the mapped result."""
        eng.smoothed_score = score
        eng._silence_frames = 0
        # Use a tiny alpha so the score barely moves
        eng.alpha = 0.01
        eng._speech_frame_count = _WARMUP_FRAMES  # skip warmup ramp
        return eng.update_risk(score, True)

    def test_low_risk_below_threshold(self):
        r = self._set_score(fresh(), 15.0)
        assert r['risk_level'] == 'LOW'
        assert r['action_trigger'] == 'ALLOW'
        assert r['status_color'] == '#10B981'

    def test_elevated_at_low_boundary(self):
        r = self._set_score(fresh(), config.risk.threshold_low_risk)
        assert r['risk_level'] == 'ELEVATED'
        assert r['action_trigger'] == 'ALLOW_MONITORED'

    def test_elevated_mid_range(self):
        r = self._set_score(fresh(), 50.0)
        assert r['risk_level'] == 'ELEVATED'
        assert r['action_trigger'] == 'ALLOW_MONITORED'
        assert r['status_color'] == '#F59E0B'

    def test_suspicious_at_mid_boundary(self):
        r = self._set_score(fresh(), config.risk.threshold_mid_risk)
        assert r['risk_level'] == 'SUSPICIOUS'
        assert r['action_trigger'] == 'TRIGGER_MFA'
        assert r['status_color'] == '#F97316'

    def test_suspicious_mid_range(self):
        r = self._set_score(fresh(), 75.0)
        assert r['risk_level'] == 'SUSPICIOUS'
        assert r['action_trigger'] == 'TRIGGER_MFA'

    def test_critical_at_high_boundary(self):
        r = self._set_score(fresh(), config.risk.threshold_high_risk)
        assert r['risk_level'] == 'CRITICAL'
        assert r['action_trigger'] == 'INTERCEPT_BLOCK'
        assert r['status_color'] == '#EF4444'

    def test_critical_at_100(self):
        r = self._set_score(fresh(), 100.0)
        assert r['risk_level'] == 'CRITICAL'
        assert r['action_trigger'] == 'INTERCEPT_BLOCK'

    def test_score_clamped_to_100(self):
        """smoothed_score must never exceed 100.0."""
        eng = fresh()
        eng.smoothed_score = 99.0
        eng.alpha = 1.0
        eng._speech_frame_count = _WARMUP_FRAMES
        r = eng.update_risk(200.0, True)  # raw way above 100
        assert r['dynamic_risk_score'] <= 100.0

    def test_score_clamped_to_zero(self):
        """smoothed_score must never go below 0.0."""
        eng = fresh()
        eng.smoothed_score = 0.1
        r = eng.update_risk(0.0, False)
        assert r['dynamic_risk_score'] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# reset() state clearing
# ═══════════════════════════════════════════════════════════════════════════════

class TestReset:

    def test_reset_clears_smoothed_score(self):
        eng = fresh()
        for _ in range(20):
            eng.update_risk(80.0, True)
        assert eng.smoothed_score > 0
        eng.reset()
        assert eng.smoothed_score == 0.0

    def test_reset_clears_peak_score(self):
        eng = fresh()
        for _ in range(10):
            eng.update_risk(90.0, True)
        eng.reset()
        assert eng.peak_score == 0.0

    def test_reset_clears_speech_scores(self):
        eng = fresh()
        for _ in range(5):
            eng.update_risk(50.0, True)
        eng.reset()
        assert len(eng.speech_scores) == 0

    def test_reset_clears_history(self):
        eng = fresh()
        for _ in range(10):
            eng.update_risk(60.0, True)
        eng.reset()
        assert len(eng.history) == 0

    def test_reset_clears_silence_counter(self):
        eng = fresh()
        for _ in range(7):
            eng.update_risk(0.0, False)
        assert eng._silence_frames == 7
        eng.reset()
        assert eng._silence_frames == 0

    def test_reset_clears_speech_frame_count(self):
        eng = fresh()
        for _ in range(10):
            eng.update_risk(50.0, True)
        eng.reset()
        assert eng._speech_frame_count == 0

    def test_post_reset_cold_start_works(self):
        """After reset, the engine must behave as a fresh instance on next frame."""
        eng = fresh()
        for _ in range(20):
            eng.update_risk(80.0, True)
        eng.reset()
        r = eng.update_risk(50.0, True)
        # Cold-start: first frame alpha ≈ 0.08+... so score << 50
        assert r['dynamic_risk_score'] < 20.0

    def test_get_summary_after_reset_returns_zeros(self):
        eng = fresh()
        for _ in range(10):
            eng.update_risk(70.0, True)
        eng.reset()
        s = eng.get_summary()
        assert s['peak_risk_score'] == 0.0
        assert s['avg_speech_risk'] == 0.0
        assert s['final_risk_score'] == 0.0


def test_single_speech_frame_microburst_registers_peak():
    from risk_engine import DynamicRiskEngine
    eng = DynamicRiskEngine()
    eng.update_risk(80.0, True)
    summary = eng.get_summary()
    assert summary['peak_risk_score'] > 0.0
    assert summary['peak_risk_score'] == 80.0
