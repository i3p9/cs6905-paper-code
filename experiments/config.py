"""Shared configuration for all experiments."""

import os

# Reproducibility
RANDOM_SEED = 42

# Embedding model (local, via sentence-transformers)
MODEL_NAME = "all-MiniLM-L6-v2"

# Knowledge healing thresholds
DRIFT_THRESHOLD = 0.15  # cosine distance above which drift is "significant"

# Infrastructure healing thresholds
INFRA_THRESHOLDS = {
    "latency_p99_ms": 500,
    "error_rate_pct": 5.0,
    "memory_util_pct": 85.0,
    "cold_start_pct": 20.0,
}

# Experiment parameters
NUM_DOCUMENTS = 20
NUM_TRIALS = 5
NUM_QUERIES = 10
NUM_METRIC_POINTS = 1000
NUM_INJECTED_ANOMALIES = 15

# AWS pricing (USD, as of Feb 2026 — from public pricing pages)
AWS_PRICING = {
    "lambda_per_request": 0.0000002,         # $0.20 per 1M requests
    "lambda_per_gb_sec": 0.0000166667,        # per GB-second
    "lambda_memory_mb": 256,                  # allocated memory
    "lambda_avg_duration_ms": 200,            # avg execution time
    "eventbridge_per_event": 0.000001,        # $1.00 per 1M events
    "cloudwatch_metrics_per_metric": 0.30,    # per metric per month (first 10k)
    "cloudwatch_alarms_standard": 0.10,       # per alarm per month
    "cloudwatch_alarms_anomaly": 3.00,        # per anomaly detection alarm per month
    "sns_per_notification": 0.00,             # first 1M free
    "dynamodb_wcu_per_month": 0.00065,        # per WCU-hour
    "dynamodb_rcu_per_month": 0.00013,        # per RCU-hour
    "bedrock_titan_embed_per_1k_tokens": 0.0001,   # Titan Embeddings v2
    "bedrock_claude_input_per_1k_tokens": 0.003,    # Claude 3 Sonnet input
    "bedrock_claude_output_per_1k_tokens": 0.015,   # Claude 3 Sonnet output
}

# Default scenario parameters (for cost calculation)
SCENARIO = {
    "monitoring_interval_hours": 6,
    "num_sources": 20,
    "avg_healing_events_per_month": 8,
    "avg_tokens_per_doc": 500,
    "avg_chunks_per_doc": 5,
    "validation_queries": 10,
    "avg_query_input_tokens": 200,
    "avg_query_output_tokens": 150,
    "num_cloudwatch_metrics": 8,
    "num_standard_alarms": 4,
    "num_anomaly_alarms": 2,
    "dynamodb_wcu": 5,
    "dynamodb_rcu": 10,
    "developer_hourly_rate": 50,
    "manual_hours_per_week": 6,  # midpoint of 4-8
}

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
