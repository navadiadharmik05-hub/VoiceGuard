import os
from pydantic import BaseModel
from typing import List

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2
    window_duration_sec: float = 0.8
    hop_duration_sec: float = 0.05
    vad_threshold_rms: float = 0.018        # Raised: filters ambient hiss/hum (< 0.018 RMS = SILENCE)
    vad_zcr_min: float = 0.012              # Zero-crossing rate minimum for speech classification
    silence_decay_factor: float = 0.55      # Aggressive EWMA decay on silence frames

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_duration_sec)

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_duration_sec)

class RiskConfig(BaseModel):
    ewma_alpha: float = 0.30                # Slightly smoothed: reduces spike sensitivity
    buffer_history_size: int = 15
    threshold_low_risk: float = 30.0
    threshold_mid_risk: float = 65.0
    threshold_high_risk: float = 85.0

class SecurityConfig(BaseModel):
    # Comma-separated allowed origins read from env var at import time.
    # e.g. VOICEGUARD_ALLOWED_ORIGINS=https://app.example.com,http://localhost:8000
    allowed_origins: List[str] = [
        o.strip()
        for o in os.environ.get(
            "VOICEGUARD_ALLOWED_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if o.strip()
    ]
    # API key for protected admin endpoints (POST /api/config, POST /api/logs/clear)
    # Set via env var VOICEGUARD_API_KEY. Empty string = auth disabled (dev mode only).
    api_key: str = os.environ.get("VOICEGUARD_API_KEY", "")

class EngineConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    risk: RiskConfig = RiskConfig()
    security: SecurityConfig = SecurityConfig()
    debug_mode: bool = True

config = EngineConfig()