#!/usr/bin/env python3
"""Master runner: executes all experiments, saves JSON + formatted tables."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR

import experiment_1_knowledge
import experiment_2_infra
import experiment_3_cost


def save_json(data: dict, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def format_summary(r1: dict, r2: dict, r3: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 70)

    # --- Experiment 1: Detection ---
    lines.append("\n1. KNOWLEDGE DRIFT DETECTION")
    lines.append("-" * 70)
    lines.append(f"{'Modification Level':<24} {'Hash':>6} {'Sem.D':>6} {'Sem.C':>6} {'Comp':>6} {'Cons':>6}  {'ChunkDist':>10}")
    lines.append("-" * 76)
    for level_name, data in r1["detection"].items():
        h = data["hash"]["f1"]
        sd = data["semantic_doc"]["f1"]
        sc = data["semantic_chunk"]["f1"]
        co = data["composite"]["f1"]
        cc = data["composite_cons"]["f1"]
        cd = data["chunk_cosine_distance"]["mean"]
        lines.append(f"{level_name:<24} {h:>6.2f} {sd:>6.2f} {sc:>6.2f} {co:>6.2f} {cc:>6.2f}  {cd:>10.4f}")

    lines.append("\n  Key: Hash=hash-only, Sem.D=semantic(doc), Sem.C=semantic(chunk),")
    lines.append("       Comp=composite(strict), Cons=composite(conservative/hash-primary)")

    # Overall across all levels
    lines.append(f"\n  OVERALL DETECTION ACCURACY (Levels 2-5 = drift, Levels 0-1 = no drift)")
    lines.append("  " + "-" * 56)
    det = r1["detection"]
    meaningful = ["Small numerical (~1%)", "Large numerical (3x)", "Partial rewrite", "Full rewrite"]
    for method in ["hash", "semantic_doc", "semantic_chunk", "composite", "composite_cons"]:
        total_tp = sum(det[l][method]["tp"] for l in det)
        total_fp = sum(det[l][method]["fp"] for l in det)
        total_fn = sum(det[l][method]["fn"] for l in det)
        total_tn = sum(det[l][method]["tn"] for l in det)
        total = total_tp + total_fp + total_tn + total_fn
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        a = (total_tp + total_tn) / total if total > 0 else 0.0
        lines.append(f"  {method:<16}  P={p:.2f}  R={r:.2f}  F1={f:.2f}  Acc={a:.2f}")

    # FNR for level 2
    l2 = det.get("Small numerical (~1%)", {}).get("composite", {})
    if l2:
        tp2 = l2.get("tp", 0)
        fn2 = l2.get("fn", 0)
        total2 = tp2 + fn2
        if total2 > 0:
            fnr2 = fn2 / total2
            lines.append(f"\n  Composite FNR at small numerical changes: {fnr2:.0%} ({fn2}/{total2})")

    # Cosine distance table
    lines.append(f"\n  COSINE DISTANCE DISTRIBUTIONS (Chunk-Level)")
    lines.append("  " + "-" * 56)
    lines.append(f"  {'Level':<24} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    lines.append("  " + "-" * 56)
    for level_name, data in r1["detection"].items():
        cd = data["chunk_cosine_distance"]
        lines.append(f"  {level_name:<24} {cd['mean']:>8.4f} {cd['std']:>8.4f} {cd['min']:>8.4f} {cd['max']:>8.4f}")

    # --- Experiment 1: Latency ---
    lines.append(f"\n2. KNOWLEDGE HEALING LATENCY")
    lines.append("-" * 70)
    avg = r1["latency"]["average"]
    lines.append(f"  Re-fetch:      {avg['refetch_s'] * 1000:>8.1f} ms")
    lines.append(f"  Re-chunk:      {avg['rechunk_s'] * 1000:>8.1f} ms")
    lines.append(f"  Re-embed:      {avg['reembed_s'] * 1000:>8.1f} ms")
    lines.append(f"  Vector upsert: {avg['upsert_s'] * 1000:>8.1f} ms")
    lines.append(f"  Validation:    {avg['validate_s'] * 1000:>8.1f} ms")
    lines.append(f"  {'─' * 35}")
    lines.append(f"  TOTAL:         {avg['total_s'] * 1000:>8.1f} ms  ({avg['total_s']:.2f}s)")
    lines.append(f"\n  Note: Local execution with {r1['config']['model']}.")
    lines.append(f"  AWS Bedrock Titan would add network latency (~100-300ms per batch).")

    # --- Experiment 2 ---
    lines.append(f"\n3. INFRASTRUCTURE HEALING")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Time(ms)':>10}")
    lines.append("-" * 70)
    for metric_name, data in r2["per_metric"].items():
        e = data["evaluation"]
        t = data["total_time_ms"]
        lines.append(f"{metric_name:<22} {e['precision']:>6.2f} {e['recall']:>6.2f} {e['f1']:>6.2f} {t:>10.2f}")

    ov = r2["overall"]
    lines.append(f"\n  Overall: P={ov['precision']:.2f} R={ov['recall']:.2f} F1={ov['f1']:.2f}")
    lines.append(f"  Avg detection+decision latency: {ov['avg_detection_decision_latency_ms']:.2f}ms")

    ca = r2["composite_alarm"]
    lines.append(f"\n  Composite alarm false-positive reduction: {ca['false_positive_reduction_pct']}%")

    # --- Experiment 3 ---
    lines.append(f"\n4. AWS COST PROJECTION (Monthly)")
    lines.append("-" * 70)
    lines.append(f"{'Component':<38} {'Agent':>12} {'Manual':>12}")
    lines.append("-" * 70)
    for component, costs in r3["breakdown"].items():
        a_str = f"${costs['agent']:.2f}"
        m_str = f"${costs['manual']:.2f}" if costs['manual'] > 0 else "\u2014"
        lines.append(f"{component:<38} {a_str:>12} {m_str:>12}")
    lines.append("-" * 70)
    t = r3["totals"]
    lines.append(f"{'TOTAL':<38} {'$' + str(t['agent_monthly']):>12} {'$' + str(t['manual_monthly']):>12}")
    lines.append(f"\n  Savings: {t['savings_pct']}% \u2014 agent at {t['agent_as_pct_of_manual']}% of manual cost")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "#" * 60)
    print("  SELF-HEALING RAG \u2014 PROOF OF CONCEPT EXPERIMENTS")
    print("#" * 60)

    t_total = time.time()

    r1 = experiment_1_knowledge.run_experiment()
    r2 = experiment_2_infra.run_experiment()
    r3 = experiment_3_cost.run_experiment()

    total_time = time.time() - t_total

    print("\n\nSaving results...")
    save_json(r1, "experiment_1_knowledge.json")
    save_json(r2, "experiment_2_infra.json")
    save_json(r3, "experiment_3_cost.json")
    save_json({
        "knowledge_healing": r1,
        "infra_healing": r2,
        "cost_projection": r3,
        "total_runtime_s": round(total_time, 2),
    }, "all_results.json")

    summary = format_summary(r1, r2, r3)
    summary_path = os.path.join(RESULTS_DIR, "summary_tables.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    print("\n" + summary)
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
