import time
import asyncio
import json
import numpy as np
import websockets

def generate_simulated_pcm_chunk(duration_sec=0.5, sample_rate=16000, mode='organic'):
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    if mode == 'organic':
        f0 = 130 + 35 * np.sin(2 * np.pi * 1.5 * t)
        audio = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * f0 * 2 * t) + 0.05 * np.random.normal(0, 1, num_samples)
    elif mode == 'synthetic':
        # Neural TTS proxy (Tacotron / HiFi-GAN / VITS artifact model):
        # 1. Near-flat F0 contour (very low pitch variance, std < 0.3 Hz)
        # 2. Smooth harmonic structure (fundamental + 4 formant-like partials)
        # 3. Subtle vocoder phase noise & high-frequency spectral flattening
        f0 = 145.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
        phase_noise = 0.35 * np.random.normal(0, 1, num_samples)
        audio = (
            0.40 * np.sin(2 * np.pi * f0 * t) +
            0.25 * np.sin(2 * np.pi * 2 * f0 * t + phase_noise) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t + 0.5 * phase_noise) +
            0.10 * np.sin(2 * np.pi * 4 * f0 * t + phase_noise) +
            0.05 * np.sin(2 * np.pi * 5 * f0 * t) +
            0.03 * np.random.normal(0, 1, num_samples)
        )
    else:
        audio = np.zeros(num_samples)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

async def run_benchmark(ws_url='ws://127.0.0.1:8000/ws/stream'):
    print('Connecting to VoiceGuard WebSocket server...')
    try:
        async with websockets.connect(ws_url) as ws:
            print('Connected! Simulating Audio Stream (6 Organic -> 6 Ai Synthetic)')
            modes = ['organic'] * 6 + ['synthetic'] * 6
            for i, mode in enumerate(modes):
                pcm_bytes = generate_simulated_pcm_chunk(duration_sec=0.5, mode=mode)
                await ws.send(pcm_bytes)
                telemetry = json.loads(await ws.recv())
                score = telemetry['risk']['dynamic_risk_score']
                level = telemetry['risk']['risk_level']
                action = telemetry['risk']['action_trigger']
                lat = telemetry['latency_ms']
                print(f'[Frame {i+1:02d} | Mode: {mode.upper():9s}] Risk: {score:5.1f}% | Level: {level:10s} | Action: {action:18s} | Latency: {lat:5.2f}ms')
                await asyncio.sleep(0.5)
    except Exception as e:
        print('Streaming Error:', e)

if __name__ == '__main__':
    asyncio.run(run_benchmark())

