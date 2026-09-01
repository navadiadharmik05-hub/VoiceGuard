from typing import Dict, Any, List
try:
    from .config import config
except ImportError:
    from config import config

_WARMUP_FRAMES = 5   # number of speech frames over which alpha ramps to full value

class DynamicRiskEngine:
    def __init__(self):
        self.alpha = config.risk.ewma_alpha
        self.silence_decay = config.audio.silence_decay_factor
        self.smoothed_score = 0.0
        self.peak_score = 0.0
        self.speech_scores: List[float] = []
        self.history: List[float] = []
        self.history_limit = config.risk.buffer_history_size
        self._silence_frames = 0       # consecutive silence frame counter
        self._speech_frame_count = 0   # total speech frames seen (for warmup ramp)

    def _ramp_alpha(self) -> float:
        """
        Returns a ramped alpha for the current speech frame.
        Starts at 0.08 on frame 1 and linearly reaches self.alpha by frame _WARMUP_FRAMES.
        Prevents the cold-start spike where smoothed_score instantly jumped to raw_frame_score.
        """
        if self._speech_frame_count >= _WARMUP_FRAMES:
            return self.alpha
        # Linear ramp from 0.08 up to self.alpha over _WARMUP_FRAMES frames
        ramp_start = 0.08
        t = self._speech_frame_count / _WARMUP_FRAMES   # 0.0 → 1.0
        return ramp_start + (self.alpha - ramp_start) * t

    def update_risk(self, raw_frame_score: float, is_speech: bool) -> Dict[str, Any]:
        if not is_speech:
            self._silence_frames += 1
            # Aggressive exponential decay: multiply by decay_factor per silence frame.
            # After 5 silence frames (~250ms at 50ms hop), a 70% score drops to ~10%.
            self.smoothed_score *= self.silence_decay
            # Hard floor: snap to 0 once below 0.5 to avoid ghost drift
            if self.smoothed_score < 0.5:
                self.smoothed_score = 0.0
        else:
            self._silence_frames = 0
            self._speech_frame_count += 1
            # Gradual cold-start: blend in at low alpha initially, ramp up over warmup frames
            alpha = self._ramp_alpha()
            self.smoothed_score = alpha * raw_frame_score + (1.0 - alpha) * self.smoothed_score
            
            self.speech_scores.append(self.smoothed_score)
            if self.smoothed_score > self.peak_score:
                self.peak_score = self.smoothed_score

        self.smoothed_score = float(max(0.0, min(100.0, self.smoothed_score)))
        self.history.append(round(self.smoothed_score, 2))
        if len(self.history) > self.history_limit:
            self.history.pop(0)

        if self.smoothed_score < config.risk.threshold_low_risk:
            risk_level = 'LOW'
            action_trigger = 'ALLOW'
            status_color = '#10B981'
        elif self.smoothed_score < config.risk.threshold_mid_risk:
            risk_level = 'ELEVATED'
            action_trigger = 'ALLOW_MONITORED'
            status_color = '#F59E0B'
        elif self.smoothed_score < config.risk.threshold_high_risk:
            risk_level = 'SUSPICIOUS'
            action_trigger = 'TRIGGER_MFA'
            status_color = '#F97316'
        else:
            risk_level = 'CRITICAL'
            action_trigger = 'INTERCEPT_BLOCK'
            status_color = '#EF4444'

        return {
            'dynamic_risk_score': round(self.smoothed_score, 2),
            'peak_risk_score': round(self.peak_score, 2),
            'risk_level': risk_level,
            'action_trigger': action_trigger,
            'status_color': status_color,
            'history_trend': list(self.history),
            'silence_frames': self._silence_frames
        }

    def get_summary(self) -> Dict[str, Any]:
        peak = float(self.peak_score)
        avg = float(sum(self.speech_scores) / len(self.speech_scores)) if self.speech_scores else 0.0
        final = float(self.smoothed_score)
        return {
            'peak_risk_score': round(peak, 2),
            'avg_speech_risk': round(avg, 2),
            'final_risk_score': round(final, 2)
        }

    def reset(self):
        self.smoothed_score = 0.0
        self.peak_score = 0.0
        self._silence_frames = 0
        self._speech_frame_count = 0
        self.speech_scores.clear()
        self.history.clear()

