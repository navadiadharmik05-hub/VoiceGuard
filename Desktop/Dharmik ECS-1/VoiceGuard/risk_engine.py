from typing import Dict, Any, List
import numpy as np
try:
    from .config import config
except ImportError:
    from config import config

_WARMUP_FRAMES = 5

class DynamicRiskEngine:
    def __init__(self):
        self.alpha = config.risk.ewma_alpha
        self.silence_decay = config.audio.silence_decay_factor
        self.smoothed_score = 0.0
        self.peak_score = 0.0
        self.speech_scores: List[float] = []
        self.history: List[float] = []
        self.history_limit = config.risk.buffer_history_size
        self._silence_frames = 0
        self._speech_frame_count = 0
        
        self.buf_100ms: List[float] = []
        self.buf_500ms: List[float] = []
        self.buf_2000ms: List[float] = []

    def _ramp_alpha(self) -> float:
        if self._speech_frame_count >= _WARMUP_FRAMES:
            return self.alpha
        ramp_start = 0.08
        t = self._speech_frame_count / _WARMUP_FRAMES
        return ramp_start + (self.alpha - ramp_start) * t

    def update_risk(self, raw_frame_score: float, is_speech: bool) -> Dict[str, Any]:
        if not is_speech:
            self._silence_frames += 1
            if self.peak_score >= 65.0 and self._silence_frames < 15:
                self.smoothed_score *= 0.98
            else:
                self.smoothed_score *= self.silence_decay
                
            if self.smoothed_score < 0.5:
                self.smoothed_score = 0.0
                self.buf_100ms.clear()
                self.buf_500ms.clear()
                self.buf_2000ms.clear()
        else:
            self._silence_frames = 0
            self._speech_frame_count += 1
            alpha = self._ramp_alpha()
            self.smoothed_score = alpha * raw_frame_score + (1.0 - alpha) * self.smoothed_score
            
            self.speech_scores.append(self.smoothed_score)
            if self.smoothed_score > self.peak_score:
                self.peak_score = self.smoothed_score

        score_val = float(self.smoothed_score if is_speech or self._silence_frames < 15 else 0.0)
        self.buf_100ms.append(score_val)
        self.buf_500ms.append(score_val)
        self.buf_2000ms.append(score_val)
        
        if len(self.buf_100ms) > 2: self.buf_100ms.pop(0)
        if len(self.buf_500ms) > 10: self.buf_500ms.pop(0)
        if len(self.buf_2000ms) > 40: self.buf_2000ms.pop(0)
        
        risk_100ms = float(np.mean(self.buf_100ms)) if self.buf_100ms else 0.0
        risk_500ms = float(np.mean(self.buf_500ms)) if self.buf_500ms else 0.0
        risk_2000ms = float(np.mean(self.buf_2000ms)) if self.buf_2000ms else 0.0
        
        if not is_speech and self._silence_frames >= 15:
            final_eval_score = float(max(0.0, min(100.0, self.smoothed_score)))
        else:
            multi_scale_fused = max(self.smoothed_score, 0.50 * risk_100ms + 0.30 * risk_500ms + 0.20 * risk_2000ms)
            final_eval_score = float(max(0.0, min(100.0, multi_scale_fused)))
        
        self.smoothed_score = final_eval_score
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
            'multi_scale': {
                'risk_100ms': round(risk_100ms, 2),
                'risk_500ms': round(risk_500ms, 2),
                'risk_2000ms': round(risk_2000ms, 2)
            },
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
        self.buf_100ms.clear()
        self.buf_500ms.clear()
        self.buf_2000ms.clear()
