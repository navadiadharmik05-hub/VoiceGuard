import os
import io
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Optional

try:
    from .config import config
    from .detector import DualLayerDetector
    from .risk_engine import DynamicRiskEngine
except ImportError:
    from config import config
    from detector import DualLayerDetector
    from risk_engine import DynamicRiskEngine

def inspect_file(file_input: Any, detector: Optional[DualLayerDetector] = None) -> Dict[str, Any]:
    """
    Complete offline audio file inspector for deepfake detection.
    - Accepts file path (str), bytes, or BytesIO.
    - Automatically resamples to 16 kHz mono float32.
    - Peak-normalizes low gain recordings.
    - Runs a 2.0s sliding window with a 0.5s hop size across full audio duration.
    - Aggregates frame scores into overall_deepfake_risk, peak_risk, and verdict.
    """
    target_sr = config.audio.sample_rate  # 16000
    
    # 1. Load Audio Data
    if isinstance(file_input, (str, os.PathLike)):
        data, sr = sf.read(file_input, dtype='float32')
    elif isinstance(file_input, bytes):
        data, sr = sf.read(io.BytesIO(file_input), dtype='float32')
    elif hasattr(file_input, 'read'):
        data, sr = sf.read(file_input, dtype='float32')
    else:
        raise ValueError(f"Unsupported file input type: {type(file_input)}")
        
    # Downmix multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample to 16kHz if needed
    if sr != target_sr:
        import scipy.signal as signal
        num_samples = int(len(data) * target_sr / sr)
        data = signal.resample(data, num_samples).astype(np.float32)
        
    # Peak normalize audio gain
    max_peak = float(np.max(np.abs(data)))
    if max_peak > 1e-5:
        data = data / max_peak * 0.90
        
    if detector is None:
        detector = DualLayerDetector(sample_rate=target_sr)
        
    risk_engine = DynamicRiskEngine()
    window_samples = config.audio.window_samples  # 2.0s (32000 samples @ 16kHz)
    hop_samples = int(target_sr * 0.5)            # 0.5s hop (8000 samples)
    
    timeline: List[Dict[str, Any]] = []
    speech_scores: List[float] = []
    
    # 2. Sliding Window Analysis across full file duration
    for start_idx in range(0, max(1, len(data) - window_samples + 1), hop_samples):
        window = data[start_idx: start_idx + window_samples]
        if len(window) < 256:
            continue
            
        rms = float(np.sqrt(np.mean(np.square(window))))
        is_speech = rms >= config.audio.vad_threshold_rms
        
        det_res = detector.process_window(window) if is_speech else {
            "raw_frame_score": 0.0,
            "ml_score": 0.0,
            "acoustic": {"acoustic_score": 0.0, "spectral_flatness": 0.0, "spectral_rolloff_hz": 0.0, "phase_discontinuity_var": 0.0},
            "prosodic": {"prosodic_score": 0.0, "f0_mean_hz": 0.0, "f0_std_hz": 0.0, "jitter": 0.0, "shimmer": 0.0},
            "contributing_factors": {
                "acoustic": {"phase_discontinuity": 0.0, "spectral_flatness": 0.0, "rolloff": 0.0},
                "prosodic": {"flat_pitch": 0.0, "jitter": 0.0, "shimmer": 0.0},
                "dominant_signal": "none"
            }
        }
        
        risk_res = risk_engine.update_risk(
            raw_frame_score=det_res["raw_frame_score"],
            is_speech=is_speech
        )
        
        if is_speech:
            speech_scores.append(det_res["raw_frame_score"])
            
        timeline.append({
            "time_sec": round(start_idx / target_sr, 2),
            "is_speech": is_speech,
            "detection": det_res,
            "risk": risk_res
        })
        
    summary = risk_engine.get_summary()
    peak_risk = summary["peak_risk_score"]
    avg_speech_risk = round(float(np.mean(speech_scores)), 2) if speech_scores else 0.0
    
    # Overall deepfake risk evaluation combines peak risk and speech average
    overall_deepfake_risk = round(0.60 * peak_risk + 0.40 * avg_speech_risk, 2) if speech_scores else round(peak_risk, 2)
    
    if overall_deepfake_risk < config.risk.threshold_low_risk:
        verdict, color = "ALLOW", "#10B981"
    elif overall_deepfake_risk < config.risk.threshold_mid_risk:
        verdict, color = "ALLOW_MONITORED", "#F59E0B"
    elif overall_deepfake_risk < config.risk.threshold_high_risk:
        verdict, color = "TRIGGER_MFA", "#F97316"
    else:
        verdict, color = "INTERCEPT_BLOCK", "#EF4444"
        
    return {
        "overall_deepfake_risk": overall_deepfake_risk,
        "overall_risk_score": overall_deepfake_risk,
        "peak_risk": peak_risk,
        "peak_risk_score": peak_risk,
        "avg_speech_risk": avg_speech_risk,
        "final_risk_score": summary["final_risk_score"],
        "verdict": verdict,
        "overall_action": verdict,
        "status_color": color,
        "duration_sec": round(len(data) / target_sr, 2),
        "frames_analyzed": len(timeline),
        "frame_scores": timeline,
        "timeline": timeline
    }
