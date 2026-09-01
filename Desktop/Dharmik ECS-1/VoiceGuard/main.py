import time, json, pathlib, io, asyncio, librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import Dict, Any, List

try:
    from .config import config
    from .dsp_pipeline import DSPPipeline
    from .detector import DualLayerDetector
    from .risk_engine import DynamicRiskEngine
except ImportError:
    from config import config
    from dsp_pipeline import DSPPipeline
    from detector import DualLayerDetector
    from risk_engine import DynamicRiskEngine
    from offline_inspector import inspect_file

app = FastAPI(title="VoiceGuard Enterprise API Engine", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins,  # set VOICEGUARD_ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

static_path = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

audit_logs_history: List[Dict[str, Any]] = []

def append_audit_log(entry: Dict[str, Any]):
    audit_logs_history.append(entry)
    if len(audit_logs_history) > 500:
        audit_logs_history.pop(0)

_SILENCE_DETECTION: Dict[str, Any] = {
    "raw_frame_score": 0.0,
    "ml_score": 0.0,
    "raw_frame_score": 0.0,
    "acoustic": {
        "acoustic_score": 0.0,
        "spectral_flatness": 0.0,
        "spectral_rolloff_hz": 0.0,
        "phase_discontinuity_var": 0.0
    },
    "prosodic": {
        "prosodic_score": 0.0,
        "f0_mean_hz": 0.0,
        "f0_std_hz": 0.0,
        "jitter": 0.0,
        "shimmer": 0.0
    },
    "contributing_factors": {
        "acoustic": {"phase_discontinuity": 0.0, "spectral_flatness": 0.0, "rolloff": 0.0},
        "prosodic": {"flat_pitch": 0.0, "jitter": 0.0, "shimmer": 0.0},
        "dominant_signal": "none"
    }
}

@app.get("/")
async def get_dashboard():
    return FileResponse(str(static_path / "index.html"))

@app.get("/api/health")
async def health_check():
    return {
        "status": "active",
        "system": "VoiceGuard Enterprise Platform",
        "target_latency": "<200ms",
        "version": "2.1.0",
        "timestamp": round(time.time(), 3)
    }

@app.get("/api/config")
async def get_system_config():
    return {
        "audio": config.audio.model_dump(),
        "risk": config.risk.model_dump(),
        "debug_mode": config.debug_mode
    }

@app.post("/api/config")
async def update_system_config(payload: Dict[str, Any] = Body(...)):
    if "ewma_alpha" in payload:
        config.risk.ewma_alpha = float(payload["ewma_alpha"])
    if "threshold_low_risk" in payload:
        config.risk.threshold_low_risk = float(payload["threshold_low_risk"])
    if "threshold_mid_risk" in payload:
        config.risk.threshold_mid_risk = float(payload["threshold_mid_risk"])
    if "threshold_high_risk" in payload:
        config.risk.threshold_high_risk = float(payload["threshold_high_risk"])
    return {"status": "updated", "config": await get_system_config()}

@app.get("/api/logs")
async def get_audit_logs():
    return {"total": len(audit_logs_history), "logs": audit_logs_history}

@app.post("/api/logs/clear")
async def clear_audit_logs():
    audit_logs_history.clear()
    return {"status": "cleared"}

@app.post("/api/analyze-file")
@app.post("/api/analyze-file")
async def analyze_audio_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = await asyncio.to_thread(inspect_file, file_bytes)
        result["filename"] = file.filename
        
        append_audit_log({
            "timestamp": round(time.time(), 3),
            "event": "FILE_ANALYSIS_COMPLETED",
            "filename": file.filename,
            "risk_score": result["overall_deepfake_risk"],
            "risk_level": result["verdict"],
            "action": result["overall_action"]
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File Analysis Failed: {e}")

async def analyze_audio_file_legacy(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        audio_data, _ = await asyncio.to_thread(
            lambda: librosa.load(io.BytesIO(file_bytes), sr=config.audio.sample_rate, mono=True)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    detector = DualLayerDetector(sample_rate=config.audio.sample_rate)
    risk_engine = DynamicRiskEngine()
    window_samples = config.audio.window_samples
    hop_samples = max(config.audio.hop_samples, 1)
    timeline: List[Dict[str, Any]] = []

    def _process_frames():
        for start_idx in range(0, len(audio_data), hop_samples):
            window = audio_data[start_idx: start_idx + window_samples]
            if len(window) < 256:
                continue
            rms = float(np.sqrt(np.mean(np.square(window))))
            is_speech = rms >= config.audio.vad_threshold_rms
            det_res = detector.process_window(window) if is_speech else dict(_SILENCE_DETECTION)
            risk_res = risk_engine.update_risk(
                raw_frame_score=det_res["raw_frame_score"],
                is_speech=is_speech
            )
            timeline.append({
                "time_sec": round(start_idx / config.audio.sample_rate, 2),
                "is_speech": is_speech,
                "detection": det_res,
                "risk": risk_res
            })

    await asyncio.to_thread(_process_frames)

    summary = risk_engine.get_summary()
    peak_risk = summary["peak_risk_score"]

    if peak_risk < config.risk.threshold_low_risk:
        overall_level, overall_action, overall_color = "LOW", "ALLOW", "#10B981"
    elif peak_risk < config.risk.threshold_mid_risk:
        overall_level, overall_action, overall_color = "ELEVATED", "ALLOW_MONITORED", "#F59E0B"
    elif peak_risk < config.risk.threshold_high_risk:
        overall_level, overall_action, overall_color = "SUSPICIOUS", "TRIGGER_MFA", "#F97316"
    else:
        overall_level, overall_action, overall_color = "CRITICAL", "INTERCEPT_BLOCK", "#EF4444"

    append_audit_log({
        "timestamp": round(time.time(), 3),
        "event": "FILE_ANALYSIS_COMPLETED",
        "filename": file.filename,
        "risk_score": round(peak_risk, 2),
        "risk_level": overall_level,
        "action": overall_action
    })

    return {
        "filename": file.filename,
        "duration_sec": round(len(audio_data) / config.audio.sample_rate, 2),
        "overall_risk_score": round(peak_risk, 2),
        "peak_risk_score": round(peak_risk, 2),
        "avg_speech_risk": round(summary["avg_speech_risk"], 2),
        "final_risk_score": round(summary["final_risk_score"], 2),
        "overall_risk_level": overall_level,
        "overall_action": overall_action,
        "status_color": overall_color,
        "frames_analyzed": len(timeline),
        "timeline": timeline
    }

@app.websocket("/ws/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    dsp = DSPPipeline()
    detector = DualLayerDetector(sample_rate=config.audio.sample_rate)
    risk_engine = DynamicRiskEngine()
    compute_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    dropped_frames = 0   # running count of frames silently dropped due to full queue

    async def ingestion_loop():
        nonlocal dropped_frames  # declared once at top of closure
        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    fc = dsp.pcm16_to_float32(message["bytes"])
                    if compute_queue.full():
                        dropped_frames += 1
                        try:
                            await websocket.send_json({"frame_dropped": True, "total_dropped": dropped_frames})
                        except Exception:
                            pass
                    else:
                        await compute_queue.put(("audio", fc))
                elif "text" in message and message["text"]:
                    try:
                        p = json.loads(message["text"])
                        if p.get("action") == "reset":
                            await compute_queue.put(("reset", None))
                        elif "audio_samples" in p:
                            fc = np.array(p["audio_samples"], dtype=np.float32)
                            if compute_queue.full():
                                dropped_frames += 1
                                try:
                                    await websocket.send_json({"frame_dropped": True, "total_dropped": dropped_frames})
                                except Exception:
                                    pass
                            else:
                                await compute_queue.put(("audio", fc))
                    except Exception:
                        pass
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await compute_queue.put(("stop", None))

    async def compute_and_send_loop():
        while True:
            try:
                item = await asyncio.wait_for(compute_queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                continue
            kind, data = item
            if kind == "stop":
                break
            if kind == "reset":
                risk_engine.reset()
                try:
                    await websocket.send_json({"status": "reset_complete"})
                except Exception:
                    break
                append_audit_log({"timestamp": round(time.time(), 3), "event": "ENGINE_RESET"})
                continue
            start_time = time.perf_counter()
            float_chunk: np.ndarray = data
            vad_res = dsp.compute_vad(float_chunk)
            dsp.append_chunk(float_chunk)
            audio_window = dsp.get_latest_window()
            if vad_res["is_speech"]:
                detection_res = await asyncio.to_thread(detector.process_window, audio_window)
            else:
                detection_res = dict(_SILENCE_DETECTION)
            risk_res = risk_engine.update_risk(
                raw_frame_score=detection_res["raw_frame_score"],
                is_speech=vad_res["is_speech"]
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            telemetry = {
                "timestamp": round(time.time(), 3),
                "latency_ms": latency_ms,
                "latency_pass": latency_ms < 200.0,
                "vad": vad_res,
                "detection": detection_res,
                "risk": risk_res,
                "dropped_frames": dropped_frames
            }
            if vad_res["is_speech"] and risk_res["dynamic_risk_score"] >= 65.0:
                append_audit_log({
                    "timestamp": telemetry["timestamp"],
                    "event": "THREAT_ALERT",
                    "risk_score": risk_res["dynamic_risk_score"],
                    "risk_level": risk_res["risk_level"],
                    "action": risk_res["action_trigger"],
                    "latency_ms": latency_ms
                })
            try:
                await websocket.send_json(telemetry)
            except Exception:
                break

    await asyncio.gather(ingestion_loop(), compute_and_send_loop())
