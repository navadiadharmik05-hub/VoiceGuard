#!/usr/bin/env python3
"""
VoiceGuard Pre-Demo Checklist
------------------------------
Run this 2-5 minutes before your showcase to confirm the server is fully
warm and healthy. Fix anything that shows [FAIL] before going on stage.

Usage:
    python3 predemo_check.py [base_url]

    base_url defaults to http://127.0.0.1:8000
"""

import sys
import time
import io
import struct

try:
    import requests
except ImportError:
    print("Missing dependency. Run: pip install requests")
    sys.exit(1)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check(name, fn):
    print(f"Checking: {name} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        ok, detail = fn()
        dt = time.time() - t0
        status = PASS if ok else FAIL
        print(f"{status} ({dt:.2f}s) {detail}")
        results.append((name, ok))
    except Exception as e:
        dt = time.time() - t0
        print(f"{FAIL} ({dt:.2f}s) crashed: {type(e).__name__}: {e}")
        results.append((name, False))


def make_silent_wav(duration_sec=0.5, sample_rate=16000):
    """Build a minimal silent WAV file in memory, no external deps needed."""
    n_samples = int(sample_rate * duration_sec)
    data = b"\x00\x00" * n_samples  # 16-bit silence
    byte_rate = sample_rate * 2
    block_align = 2
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    buf.seek(0)
    return buf.getvalue()


def check_health():
    r = requests.get(f"{BASE_URL}/api/health", timeout=5)
    ok = r.status_code == 200
    return ok, f"status={r.status_code}"


def check_dashboard():
    r = requests.get(f"{BASE_URL}/", timeout=5)
    ok = r.status_code == 200 and len(r.text) > 0
    if r.status_code != 200:
        return False, f"status={r.status_code} -- static/index.html may be missing!"
    return ok, f"status={r.status_code}, {len(r.text)} bytes"


def check_config():
    r = requests.get(f"{BASE_URL}/api/config", timeout=5)
    ok = r.status_code == 200 and "audio" in r.json()
    return ok, f"status={r.status_code}"


def check_warmup_upload():
    """First real upload -- this is where cold-start delay would show up."""
    wav_bytes = make_silent_wav(0.5)
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/api/analyze-file",
        files={"file": ("warmup.wav", wav_bytes, "audio/wav")},
        timeout=30,
    )
    dt = time.time() - t0
    ok = r.status_code == 200
    slow_warning = " -- SLOW, run this script again before demo!" if dt > 2.0 else ""
    return ok, f"status={r.status_code}, took {dt:.2f}s{slow_warning}"


def check_second_upload_speed():
    """Second upload should be fast if warmup worked."""
    wav_bytes = make_silent_wav(1.0)
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/api/analyze-file",
        files={"file": ("second.wav", wav_bytes, "audio/wav")},
        timeout=30,
    )
    dt = time.time() - t0
    ok = r.status_code == 200 and dt < 2.0
    detail = f"status={r.status_code}, took {dt:.2f}s"
    if dt >= 2.0:
        detail += " -- unexpectedly slow for a 1s clip"
    return ok, detail


def check_bad_file_handling():
    """Confirm garbage input doesn't crash the server."""
    r = requests.post(
        f"{BASE_URL}/api/analyze-file",
        files={"file": ("garbage.mp3", b"not real audio data", "audio/mpeg")},
        timeout=10,
    )
    ok = r.status_code == 400
    return ok, f"status={r.status_code} (expect 400)"


def check_logs_endpoint():
    r = requests.get(f"{BASE_URL}/api/logs", timeout=5)
    ok = r.status_code == 200
    return ok, f"status={r.status_code}, {r.json().get('total', '?')} log entries"


if __name__ == "__main__":
    print(f"\nVoiceGuard Pre-Demo Check -- target: {BASE_URL}\n" + "-" * 50)

    check("Server reachable (/api/health)", check_health)
    check("Dashboard loads (static/index.html present)", check_dashboard)
    check("Config endpoint", check_config)
    check("First file upload (cold-start timing)", check_warmup_upload)
    check("Second file upload (should be fast now)", check_second_upload_speed)
    check("Garbage file rejected cleanly (no crash)", check_bad_file_handling)
    check("Audit logs endpoint", check_logs_endpoint)

    print("-" * 50)
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed.\n")

    if n_pass == n_total:
        print("\033[92mAll clear -- you're good to demo.\033[0m")
    else:
        print("\033[91mFix the FAILed items above before going on stage.\033[0m")
        sys.exit(1)
