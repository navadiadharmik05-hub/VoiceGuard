import io
import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'active'
    assert 'version' in data

def test_get_dashboard():
    response = client.get('/')
    assert response.status_code == 200
    assert 'VoiceGuard' in response.text

def test_get_config():
    response = client.get('/api/config')
    assert response.status_code == 200
    data = response.json()
    assert 'audio' in data
    assert 'risk' in data

def test_update_config():
    response = client.post('/api/config', json={'ewma_alpha': 0.30})
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'updated'

def test_audit_logs_and_clear():
    response = client.get('/api/logs')
    assert response.status_code == 200
    assert 'logs' in response.json()
    
    clear_resp = client.post('/api/logs/clear')
    assert clear_resp.status_code == 200
    assert clear_resp.json()['status'] == 'cleared'

def test_analyze_file():
    # Generate a small 1-second 16kHz synthetic PCM WAV file
    import wave
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio_pcm = (np.sin(2 * np.pi * 150 * t) * 16384).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_pcm.tobytes())
    buf.seek(0)
    
    response = client.post('/api/analyze-file', files={'file': ('test.wav', buf, 'audio/wav')})
    assert response.status_code == 200
    data = response.json()
    assert data['filename'] == 'test.wav'
    assert 'overall_risk_score' in data
    assert 'timeline' in data

def test_websocket_stream():
    with client.websocket_connect('/ws/stream') as websocket:
        # Send PCM16 audio bytes (0.5 sec)
        samples = np.zeros(8000, dtype=np.int16)
        websocket.send_bytes(samples.tobytes())
        data = websocket.receive_json()
        assert 'risk' in data
        assert 'detection' in data
        assert 'contributing_factors' in data['detection']
