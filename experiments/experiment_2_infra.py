"""Experiment 2: Infrastructure Healing Loop — anomaly detection & decision latency.

Simulates CloudWatch-style metric monitoring with:
  - Synthetic time series (mean-reverting + noise)
  - Injected anomalies at known timestamps
  - Rule-based detection with rolling baseline + sustained threshold breach
  - Composite alarm test (multi-metric correlation)
"""

import json
import time

import numpy as np

from config import INFRA_THRESHOLDS, NUM_METRIC_POINTS, NUM_INJECTED_ANOMALIES, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def generate_metric_series(
    name: str, baseline: float, noise_std: float,
    mean_reversion_rate: float = 0.05, n_points: int = NUM_METRIC_POINTS,
) -> np.ndarray:
    """Generate a mean-reverting time series with Gaussian noise."""
    series = np.zeros(n_points)
    series[0] = baseline
    for i in range(1, n_points):
        reversion = mean_reversion_rate * (baseline - series[i - 1])
        series[i] = series[i - 1] + reversion + np.random.normal(0, noise_std)
        if "pct" in name:
            series[i] = np.clip(series[i], 0, 100)
        else:
            series[i] = max(0, series[i])
    return series


def inject_anomalies(series: np.ndarray, n_anomalies: int, anomaly_magnitude: float,
                     min_gap: int = 30) -> tuple:
    """Inject sustained anomalies (3-5 consecutive elevated points)."""
    modified = series.copy()
    n = len(series)
    candidates = list(range(50, n - 20))
    anomaly_ranges = []

    for _ in range(n_anomalies):
        if not candidates:
            break
        idx = int(np.random.choice(candidates))
        duration = np.random.randint(3, 6)
        anomaly_ranges.append((idx, idx + duration))
        for j in range(duration):
            if idx + j < n:
                modified[idx + j] += anomaly_magnitude
        candidates = [c for c in candidates if abs(c - idx) > min_gap]

    return modified, anomaly_ranges


def detect_anomalies(series: np.ndarray, threshold: float,
                     window_size: int = 30, min_consecutive: int = 2) -> list:
    """Threshold + rolling z-score detection with sustained-breach requirement.

    A detection is raised when min_consecutive consecutive points exceed
    either the static threshold or 3-sigma from the rolling baseline.
    """
    n = len(series)
    point_flags = np.zeros(n, dtype=bool)

    for i in range(window_size, n):
        window = series[max(0, i - window_size):i]
        rolling_mean = np.mean(window)
        rolling_std = np.std(window)
        z = (series[i] - rolling_mean) / rolling_std if rolling_std > 1e-9 else 0.0
        if series[i] > threshold or z > 3.0:
            point_flags[i] = True

    # Require min_consecutive consecutive flagged points → reduce false positives
    detected = []
    i = window_size
    while i < n:
        if point_flags[i]:
            run_start = i
            while i < n and point_flags[i]:
                i += 1
            run_len = i - run_start
            if run_len >= min_consecutive:
                detected.append(run_start)  # report start of sustained breach
        else:
            i += 1

    return detected


def evaluate_detection(detected_indices: list, true_ranges: list,
                       tolerance: int = 5) -> dict:
    """Evaluate detection accuracy against known anomaly time ranges."""
    matched_ranges = set()
    tp = 0
    fp = 0

    for d in detected_indices:
        matched = False
        for ri, (start, end) in enumerate(true_ranges):
            if start - tolerance <= d <= end + tolerance:
                if ri not in matched_ranges:
                    tp += 1
                    matched_ranges.add(ri)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(true_ranges) - len(matched_ranges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


REMEDIATION_MAP = {
    "latency_p99_ms": "Scale Lambda memory / enable provisioned concurrency",
    "error_rate_pct": "Analyze CloudWatch Logs Insights / trigger code rollback",
    "memory_util_pct": "Increase Lambda memory allocation",
    "cold_start_pct": "Enable provisioned concurrency / warm-up schedule",
}


def run_experiment():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Infrastructure Healing Loop")
    print("=" * 60)

    metrics_config = {
        "latency_p99_ms": {"baseline": 200, "noise_std": 30, "anomaly_mag": 400},
        "error_rate_pct": {"baseline": 1.0, "noise_std": 0.5, "anomaly_mag": 8.0},
        "memory_util_pct": {"baseline": 55, "noise_std": 5, "anomaly_mag": 35},
        "cold_start_pct": {"baseline": 8, "noise_std": 2, "anomaly_mag": 18},
    }

    per_metric_results = {}
    all_decision_latencies = []

    for metric_name, cfg in metrics_config.items():
        print(f"\n  Metric: {metric_name}")
        threshold = INFRA_THRESHOLDS[metric_name]

        series = generate_metric_series(metric_name, cfg["baseline"], cfg["noise_std"])
        anomalous_series, true_ranges = inject_anomalies(
            series, NUM_INJECTED_ANOMALIES, cfg["anomaly_mag"]
        )
        print(f"    Injected {len(true_ranges)} anomalies")

        # Detect
        t_start = time.time()
        detected = detect_anomalies(anomalous_series, threshold)
        detection_time = time.time() - t_start

        # Evaluate
        eval_result = evaluate_detection(detected, true_ranges)
        print(f"    Detection: P={eval_result['precision']:.2f} R={eval_result['recall']:.2f} "
              f"F1={eval_result['f1']:.2f}")
        print(f"    Detection time: {detection_time * 1000:.2f}ms for {NUM_METRIC_POINTS} points")

        # Decision engine timing
        t_start = time.time()
        decisions = []
        for d_idx in detected:
            decisions.append({
                "timestamp_idx": int(d_idx),
                "metric": metric_name,
                "value": round(float(anomalous_series[d_idx]), 2),
                "threshold": threshold,
                "action": REMEDIATION_MAP[metric_name],
            })
        decision_time = time.time() - t_start
        all_decision_latencies.append(detection_time + decision_time)

        per_metric_results[metric_name] = {
            "threshold": threshold,
            "num_true_anomalies": len(true_ranges),
            "num_detected": len(detected),
            "evaluation": eval_result,
            "detection_time_ms": round(detection_time * 1000, 2),
            "decision_time_ms": round(decision_time * 1000, 2),
            "total_time_ms": round((detection_time + decision_time) * 1000, 2),
            "sample_decisions": decisions[:3],
        }

    # --- Composite alarm test ---
    print("\n  Composite Alarm Test (latency + error rate):")
    np.random.seed(RANDOM_SEED + 100)  # different seed for this test
    lat_series = generate_metric_series("latency_p99_ms", 200, 30)
    err_series = generate_metric_series("error_rate_pct", 1.0, 0.5)
    lat_series, lat_ranges = inject_anomalies(lat_series, NUM_INJECTED_ANOMALIES, 400)
    err_series, err_ranges = inject_anomalies(err_series, NUM_INJECTED_ANOMALIES, 8.0)

    lat_det = set(detect_anomalies(lat_series, INFRA_THRESHOLDS["latency_p99_ms"]))
    err_det = set(detect_anomalies(err_series, INFRA_THRESHOLDS["error_rate_pct"]))
    composite_det = lat_det & err_det
    single_det = lat_det | err_det
    fp_reduction = 1 - len(composite_det) / len(single_det) if single_det else 0
    print(f"    Single-metric alarms: {len(single_det)}")
    print(f"    Composite alarms: {len(composite_det)}")
    print(f"    False-positive reduction: {fp_reduction * 100:.1f}%")

    # Overall summary
    all_tp = sum(r["evaluation"]["tp"] for r in per_metric_results.values())
    all_fp = sum(r["evaluation"]["fp"] for r in per_metric_results.values())
    all_fn = sum(r["evaluation"]["fn"] for r in per_metric_results.values())
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 1.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0)
    avg_latency_ms = float(np.mean(all_decision_latencies)) * 1000

    overall = {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
        "avg_detection_decision_latency_ms": round(avg_latency_ms, 2),
    }
    print(f"\n  Overall: P={overall['precision']:.2f} R={overall['recall']:.2f} F1={overall['f1']:.2f}")
    print(f"  Avg detection+decision latency: {avg_latency_ms:.2f}ms")

    return {
        "experiment": "Infrastructure Healing Loop",
        "config": {
            "thresholds": INFRA_THRESHOLDS,
            "num_metric_points": NUM_METRIC_POINTS,
            "num_injected_anomalies": NUM_INJECTED_ANOMALIES,
        },
        "per_metric": per_metric_results,
        "composite_alarm": {
            "single_metric_alarms": len(single_det),
            "composite_alarms": len(composite_det),
            "false_positive_reduction_pct": round(fp_reduction * 100, 1),
        },
        "overall": overall,
    }
