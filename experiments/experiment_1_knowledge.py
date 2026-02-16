"""Experiment 1: Knowledge Healing Loop — drift detection accuracy & healing latency.

Detection is evaluated at chunk level (realistic for RAG systems) with three methods:
  1. Hash-only: SHA-256 content hash comparison
  2. Semantic-only: max chunk-level cosine distance > threshold
  3. Composite: hash gate + semantic severity classification
"""

import hashlib
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from config import (
    DRIFT_THRESHOLD,
    MODEL_NAME,
    NUM_DOCUMENTS,
    NUM_TRIALS,
    RANDOM_SEED,
)
from test_data import apply_modification, get_corpus

np.random.seed(RANDOM_SEED)

# Chunk-level semantic threshold (lower than document-level since changes
# are less diluted at chunk granularity)
CHUNK_DRIFT_THRESHOLD = 0.08


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(text: str) -> list:
    """Split document into sentence-level chunks (simulates RAG chunking)."""
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    # Combine into overlapping chunks of ~2 sentences
    chunks = []
    for i in range(0, len(sentences), 1):
        chunk = '. '.join(sentences[i:i + 2])
        if not chunk.endswith('.'):
            chunk += '.'
        chunks.append(chunk)
    return chunks


def max_chunk_drift(model, old_text: str, new_text: str) -> float:
    """Compute the maximum cosine distance between aligned chunk embeddings.

    For each chunk in the new document, finds the closest chunk in the old
    document and reports the maximum drift across all new chunks.
    """
    old_chunks = chunk_document(old_text)
    new_chunks = chunk_document(new_text)

    if not old_chunks or not new_chunks:
        return 0.0

    old_embs = model.encode(old_chunks, show_progress_bar=False)
    new_embs = model.encode(new_chunks, show_progress_bar=False)

    # For each new chunk, find minimum distance to any old chunk
    sim_matrix = cos_sim(new_embs, old_embs)  # shape: (new, old)
    # Max similarity for each new chunk → convert to distance
    best_sim = sim_matrix.max(axis=1)  # best match per new chunk
    distances = 1.0 - best_sim
    return float(np.max(distances))


def classify(detected: bool, is_meaningful: bool) -> dict:
    """Return TP/FP/TN/FN counts for a single prediction."""
    if detected and is_meaningful:
        return {"tp": 1, "fp": 0, "tn": 0, "fn": 0}
    elif detected and not is_meaningful:
        return {"tp": 0, "fp": 1, "tn": 0, "fn": 0}
    elif not detected and not is_meaningful:
        return {"tp": 0, "fp": 0, "tn": 1, "fn": 0}
    else:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 1}


def aggregate_counts(counts_list: list) -> dict:
    tp = sum(c["tp"] for c in counts_list)
    fp = sum(c["fp"] for c in counts_list)
    tn = sum(c["tn"] for c in counts_list)
    fn = sum(c["fn"] for c in counts_list)
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def run_experiment():
    print("=" * 60)
    print("EXPERIMENT 1: Knowledge Healing Loop")
    print("=" * 60)

    print(f"\nLoading embedding model: {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model_load_time = time.time() - t0
    print(f"  Model loaded in {model_load_time:.2f}s")

    corpus = get_corpus()
    levels = [0, 1, 2, 3, 4, 5]
    level_names = {
        0: "No change",
        1: "Formatting only",
        2: "Small numerical (~1%)",
        3: "Large numerical (3x)",
        4: "Partial rewrite",
        5: "Full rewrite",
    }

    # ---- Part A: Drift Detection Accuracy ----
    print("\n--- Part A: Drift Detection Accuracy ---")

    # Store per-level, per-method counts
    all_counts = {
        lvl: {"hash": [], "semantic_doc": [], "semantic_chunk": [],
              "composite": [], "composite_cons": []}
        for lvl in levels
    }
    doc_cosine_dists = {lvl: [] for lvl in levels}
    chunk_cosine_dists = {lvl: [] for lvl in levels}

    for trial in range(NUM_TRIALS):
        print(f"\n  Trial {trial + 1}/{NUM_TRIALS}")

        # Compute baselines
        baseline_texts = [d.content for d in corpus]
        baseline_embeddings = model.encode(baseline_texts, show_progress_bar=False)
        baseline_hashes = [compute_hash(t) for t in baseline_texts]

        for level in levels:
            for i, doc in enumerate(corpus):
                modified_doc, mod_info = apply_modification(doc, level)
                is_meaningful = mod_info.expected_semantic_change

                # --- Hash detection ---
                new_hash = compute_hash(modified_doc.content)
                hash_changed = new_hash != baseline_hashes[i]

                # --- Document-level semantic ---
                new_emb = model.encode([modified_doc.content], show_progress_bar=False)
                doc_cos_dist = float(1.0 - cos_sim(
                    baseline_embeddings[i].reshape(1, -1), new_emb
                )[0][0])
                doc_cosine_dists[level].append(doc_cos_dist)
                doc_semantic_drift = doc_cos_dist > DRIFT_THRESHOLD

                # --- Chunk-level semantic ---
                chunk_dist = max_chunk_drift(model, doc.content, modified_doc.content)
                chunk_cosine_dists[level].append(chunk_dist)
                chunk_semantic_drift = chunk_dist > CHUNK_DRIFT_THRESHOLD

                # --- Composite strict: hash gate + chunk semantic threshold ---
                composite_drift = hash_changed and chunk_semantic_drift

                # --- Composite conservative: hash-primary detection ---
                # Any hash change triggers healing; semantic classifies severity only.
                # This is the paper's recommended approach: never miss a real change.
                composite_cons_drift = hash_changed

                # Classify
                all_counts[level]["hash"].append(classify(hash_changed, is_meaningful))
                all_counts[level]["semantic_doc"].append(classify(doc_semantic_drift, is_meaningful))
                all_counts[level]["semantic_chunk"].append(classify(chunk_semantic_drift, is_meaningful))
                all_counts[level]["composite"].append(classify(composite_drift, is_meaningful))
                all_counts[level]["composite_cons"].append(classify(composite_cons_drift, is_meaningful))

    # Aggregate per level
    detection_summary = {}
    for level in levels:
        name = level_names[level]
        level_data = {}
        for method in ["hash", "semantic_doc", "semantic_chunk", "composite", "composite_cons"]:
            level_data[method] = aggregate_counts(all_counts[level][method])

        ddists = doc_cosine_dists[level]
        cdists = chunk_cosine_dists[level]
        level_data["doc_cosine_distance"] = {
            "mean": round(float(np.mean(ddists)), 6),
            "std": round(float(np.std(ddists)), 6),
            "min": round(float(np.min(ddists)), 6),
            "max": round(float(np.max(ddists)), 6),
        }
        level_data["chunk_cosine_distance"] = {
            "mean": round(float(np.mean(cdists)), 6),
            "std": round(float(np.std(cdists)), 6),
            "min": round(float(np.min(cdists)), 6),
            "max": round(float(np.max(cdists)), 6),
        }
        detection_summary[name] = level_data

    # Print results
    print("\n\n  DRIFT DETECTION RESULTS (F1 Score)")
    print("  " + "-" * 80)
    print(f"  {'Level':<24} {'Hash':>6} {'Sem.D':>6} {'Sem.C':>6} {'Comp':>6} {'Cons':>6}")
    print("  " + "-" * 80)
    for name in [level_names[l] for l in levels]:
        d = detection_summary[name]
        print(f"  {name:<24} {d['hash']['f1']:>6.2f} {d['semantic_doc']['f1']:>6.2f} "
              f"{d['semantic_chunk']['f1']:>6.2f} {d['composite']['f1']:>6.2f} "
              f"{d['composite_cons']['f1']:>6.2f}")

    print("\n  COSINE DISTANCE DISTRIBUTIONS")
    print("  " + "-" * 72)
    print(f"  {'Level':<24} {'Doc Mean':>9} {'Doc Max':>9} {'Chunk Mean':>11} {'Chunk Max':>10}")
    print("  " + "-" * 72)
    for name in [level_names[l] for l in levels]:
        d = detection_summary[name]
        dd = d["doc_cosine_distance"]
        cd = d["chunk_cosine_distance"]
        print(f"  {name:<24} {dd['mean']:>9.4f} {dd['max']:>9.4f} {cd['mean']:>11.4f} {cd['max']:>10.4f}")

    # Overall metrics for meaningful changes (levels 2-5) vs benign (0-1)
    meaningful_names = [level_names[l] for l in [2, 3, 4, 5]]
    benign_names = [level_names[l] for l in [0, 1]]

    print("\n  OVERALL ACCURACY (Levels 2-5 = positive, Levels 0-1 = negative)")
    print("  " + "-" * 68)
    for method in ["hash", "semantic_doc", "semantic_chunk", "composite", "composite_cons"]:
        all_c = []
        for name in [level_names[l] for l in levels]:
            all_c.extend(all_counts[levels[[level_names[l] for l in levels].index(name)]][method])
        agg = aggregate_counts(all_c)
        print(f"  {method:<16}  P={agg['precision']:.2f}  R={agg['recall']:.2f}  "
              f"F1={agg['f1']:.2f}  Acc={agg['accuracy']:.2f}")

    # Semantic FNR at level 2 (small numerical)
    sem_chunk_l2 = aggregate_counts(all_counts[2]["semantic_chunk"])
    if (sem_chunk_l2["tp"] + sem_chunk_l2["fn"]) > 0:
        fnr_l2 = sem_chunk_l2["fn"] / (sem_chunk_l2["tp"] + sem_chunk_l2["fn"])
        print(f"\n  Semantic (chunk) FNR at Level 2 (small numerical): {fnr_l2:.0%}")
        print(f"    ({sem_chunk_l2['fn']} missed / {sem_chunk_l2['tp'] + sem_chunk_l2['fn']} total)")

    # ---- Part B: Healing Latency ----
    print("\n\n--- Part B: Healing Latency ---")

    latency_results = []
    for trial in range(NUM_TRIALS):
        texts = [d.content for d in corpus]

        # Step 1: Re-fetch
        t0 = time.time()
        fetched = list(texts)
        t_refetch = time.time() - t0

        # Step 2: Re-chunk
        t0 = time.time()
        chunks = []
        for text in fetched:
            chunks.extend(chunk_document(text))
        t_rechunk = time.time() - t0

        # Step 3: Re-embed (real computation)
        t0 = time.time()
        _ = model.encode(chunks, show_progress_bar=False, batch_size=32)
        t_reembed = time.time() - t0

        # Step 4: Vector upsert (simulated)
        t0 = time.time()
        store = {f"vec_{j}": c for j, c in enumerate(chunks)}
        t_upsert = time.time() - t0

        # Step 5: Validation (re-embed sample, compare)
        t0 = time.time()
        _ = model.encode(chunks[:10], show_progress_bar=False)
        t_validate = time.time() - t0

        total = t_refetch + t_rechunk + t_reembed + t_upsert + t_validate
        trial_result = {
            "trial": trial + 1,
            "refetch_s": round(t_refetch, 4),
            "rechunk_s": round(t_rechunk, 4),
            "reembed_s": round(t_reembed, 4),
            "upsert_s": round(t_upsert, 4),
            "validate_s": round(t_validate, 4),
            "total_s": round(total, 4),
        }
        latency_results.append(trial_result)
        print(f"  Trial {trial + 1}: {total:.3f}s "
              f"(fetch={t_refetch:.3f}s, chunk={t_rechunk:.3f}s, "
              f"embed={t_reembed:.3f}s, upsert={t_upsert:.3f}s, validate={t_validate:.3f}s)")

    avg_latency = {
        key: round(float(np.mean([r[key] for r in latency_results])), 4)
        for key in ["refetch_s", "rechunk_s", "reembed_s", "upsert_s", "validate_s", "total_s"]
    }
    print(f"\n  Average total healing latency: {avg_latency['total_s']:.3f}s")

    return {
        "experiment": "Knowledge Healing Loop",
        "config": {
            "model": MODEL_NAME,
            "doc_drift_threshold": DRIFT_THRESHOLD,
            "chunk_drift_threshold": CHUNK_DRIFT_THRESHOLD,
            "num_documents": NUM_DOCUMENTS,
            "num_trials": NUM_TRIALS,
        },
        "detection": detection_summary,
        "latency": {
            "trials": latency_results,
            "average": avg_latency,
        },
        "model_load_time_s": round(model_load_time, 2),
    }
