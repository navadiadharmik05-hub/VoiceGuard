import numpy as np
import pytest
from dsp_pipeline import DSPPipeline
from detector import DualLayerDetector
from risk_engine import DynamicRiskEngine

def test_feature_squeezing():
    dsp = DSPPipeline()
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    clean = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noisy = clean.copy()
    noisy[100:102] += 5.0  # isolated narrow noise spike
    
    sanitized = dsp.sanitize_input(noisy, bits=8, kernel_size=5)
    assert len(sanitized) == len(noisy)
    # Single spike should be completely removed by median filter & 8-bit quantization
    assert np.max(np.abs(sanitized[100:102])) < 2.0

def test_snr_and_cepstral_extraction():
    dsp = DSPPipeline()
    sig = np.random.randn(16000).astype(np.float32)
    snr = dsp.estimate_snr(sig)
    assert 0.0 <= snr <= 60.0
    
    lfcc = dsp.extract_lfcc(sig, num_cep=13)
    assert len(lfcc) == 13
    
    cqcc = dsp.extract_cqcc(sig, num_cep=13)
    assert len(cqcc) == 13

def test_adaptive_weight_fusion_snr():
    det_clean = DualLayerDetector(sample_rate=16000)
    sig_clean = np.sin(2 * np.pi * 300 * np.linspace(0, 1.0, 32000, endpoint=False)).astype(np.float32)
    res_clean = det_clean.process_window(sig_clean)
    assert 'fusion_mode' in res_clean
    assert res_clean['contributing_factors']['weights']['ml'] in [0.60, 0.40]
    
    det_low_sr = DualLayerDetector(sample_rate=8000)
    res_low_sr = det_low_sr.process_window(sig_clean[:16000])
    assert res_low_sr['fusion_mode'] == 'ADAPTIVE_DSP_HEAVY'
    assert res_low_sr['contributing_factors']['weights']['ml'] == 0.40

def test_multi_scale_temporal_risk_and_anti_splicing():
    engine = DynamicRiskEngine()
    # Warmup and simulate high risk initial trigger
    for _ in range(10):
        res = engine.update_risk(raw_frame_score=90.0, is_speech=True)
        
    assert 'multi_scale' in res
    assert 'risk_100ms' in res['multi_scale']
    assert 'risk_500ms' in res['multi_scale']
    assert 'risk_2000ms' in res['multi_scale']
    assert res['dynamic_risk_score'] > 60.0
    
    # Test Anti-Splicing Protection: sudden single frame silence should NOT drop score to zero
    res_silence = engine.update_risk(raw_frame_score=0.0, is_speech=False)
    assert res_silence['dynamic_risk_score'] > 40.0


from ml_classifier import MLVoiceClassifier

def test_ml_classifier_singleton_and_predict():
    clf1 = MLVoiceClassifier()
    clf2 = MLVoiceClassifier()
    assert clf1 is clf2  # Singleton check
    
    # Test buffer prediction
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    score = clf1.predict_frame(audio, sample_rate=16000)
    assert 0.0 <= score <= 100.0
    
    # Test error handling fallback
    invalid_audio = "non_existent_file_path_xyz.wav"
    fallback_score = clf1.predict_frame(invalid_audio)
    assert fallback_score == 50.0
