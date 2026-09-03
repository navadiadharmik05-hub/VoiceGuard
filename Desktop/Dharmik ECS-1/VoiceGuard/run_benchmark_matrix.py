import os
import time
import glob
import math
from typing import Dict, Any, List, Tuple
import numpy as np
from offline_inspector import inspect_file

# Directory schema: benchmarks/<category>/<bona_fide|spoof>/*.<ext>
BENCHMARK_ROOT = "benchmarks"
DEFAULT_THRESHOLDS = {"low": 35.0, "mid": 55.0, "high": 75.0}

def compute_roc_eer(scores: List[float], labels: List[int]) -> Tuple[float, float]:
    """
    Computes AUROC and Equal Error Rate (EER).
    labels: 0 for bona_fide, 1 for spoof.
    """
    if not scores or len(set(labels)) < 2:
        return 0.0, 0.0

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # Sort descending
    desc_order = np.argsort(-scores_arr)
    sorted_scores = scores_arr[desc_order]
    sorted_labels = labels_arr[desc_order]

    n_pos = int(np.sum(sorted_labels == 1))
    n_neg = int(np.sum(sorted_labels == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # Trapezoidal ROC AUC
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    auroc = float(np.trapz(tpr, fpr))

    # Equal Error Rate where FPR ~= FNR (1 - TPR)
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    min_idx = int(np.argmin(diffs))
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0) * 100.0

    return round(auroc, 4), round(eer, 2)

def evaluate_category(category_dir: str) -> Dict[str, Any]:
    bona_fide_files = []
    spoof_files = []

    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
        bona_fide_files.extend(glob.glob(os.path.join(category_dir, "bona_fide", ext)))
        spoof_files.extend(glob.glob(os.path.join(category_dir, "spoof", ext)))

    total_files = len(bona_fide_files) + len(spoof_files)
    if total_files == 0:
        return {}

    scores: List[float] = []
    labels: List[int] = []
    latencies: List[float] = []
    durations: List[float] = []
    results: List[Dict[str, Any]] = []

    # Process Bona Fide (Label 0)
    for path in bona_fide_files:
        t0 = time.perf_counter()
        try:
            res = inspect_file(path)
            lat = time.perf_counter() - t0
            latencies.append(lat)
            durations.append(res.get("duration_sec", 0.0))
            score = float(res.get("overall_deepfake_risk", 0.0))
            scores.append(score)
            labels.append(0)
            results.append({"path": path, "label": 0, "score": score, "lat": lat, "res": res})
        except Exception as e:
            print(f"[ERROR] Failed {path}: {e}")

    # Process Spoof (Label 1)
    for path in spoof_files:
        t0 = time.perf_counter()
        try:
            res = inspect_file(path)
            lat = time.perf_counter() - t0
            latencies.append(lat)
            durations.append(res.get("duration_sec", 0.0))
            score = float(res.get("overall_deepfake_risk", 0.0))
            scores.append(score)
            labels.append(1)
            results.append({"path": path, "label": 1, "score": score, "lat": lat, "res": res})
        except Exception as e:
            print(f"[ERROR] Failed {path}: {e}")

    if not scores:
        return {}

    auroc, eer = compute_roc_eer(scores, labels)

    # Standard metrics at mid threshold (55.0)
    eval_threshold = DEFAULT_THRESHOLDS["mid"]
    preds = [1 if s >= eval_threshold else 0 for s in scores]

    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    n_pos = tp + fn
    n_neg = tn + fp

    acc = ((tp + tn) / len(scores)) * 100.0 if scores else 0.0
    fpr = (fp / n_neg) * 100.0 if n_neg > 0 else 0.0
    fnr = (fn / n_pos) * 100.0 if n_pos > 0 else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    return {
        "total": len(scores),
        "auroc": auroc,
        "eer": eer,
        "accuracy": round(acc, 2),
        "fpr": round(fpr, 2),
        "fnr": round(fnr, 2),
        "avg_latency_s": round(avg_lat, 3),
    }

def run_all_benchmarks():
    if not os.path.exists(BENCHMARK_ROOT):
        print(f"Directory '{BENCHMARK_ROOT}' does not exist. Please create test benchmarks.")
        return

    categories = [
        d for d in os.listdir(BENCHMARK_ROOT)
        if os.path.isdir(os.path.join(BENCHMARK_ROOT, d))
    ]

    if not categories:
        print(f"No benchmark categories found under '{BENCHMARK_ROOT}'.")
        return

    table_rows = []
    print("Executing benchmark evaluation across categories...\n")

    for cat in sorted(categories):
        cat_path = os.path.join(BENCHMARK_ROOT, cat)
        stats = evaluate_category(cat_path)
        if not stats:
            continue
        table_rows.append((
            cat,
            stats["total"],
            stats["eer"],
            stats["accuracy"],
            stats["fpr"],
            stats["fnr"],
            stats["avg_latency_s"]
        ))

    # Output Markdown summary table
    print("| Benchmark / Stress Category | Total Files | EER (%) | Accuracy (%) | FPR (%) | FNR (%) | Avg Latency (s) |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for row in table_rows:
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |")

if __name__ == "__main__":
    run_all_benchmarks()
