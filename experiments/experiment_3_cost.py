"""Experiment 3: AWS Cost Calculator — real pricing, itemized breakdown."""

import json

from config import AWS_PRICING, SCENARIO, RESULTS_DIR


def run_experiment():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: AWS Cost Projection")
    print("=" * 60)

    p = AWS_PRICING
    s = SCENARIO

    # ---- Monitoring costs ----
    # CloudWatch metrics
    cw_metrics_cost = s["num_cloudwatch_metrics"] * p["cloudwatch_metrics_per_metric"]
    # CloudWatch alarms
    cw_alarms_standard = s["num_standard_alarms"] * p["cloudwatch_alarms_standard"]
    cw_alarms_anomaly = s["num_anomaly_alarms"] * p["cloudwatch_alarms_anomaly"]
    # EventBridge: monitoring runs every interval_hours, 24/7
    events_per_month = (30 * 24 / s["monitoring_interval_hours"]) * 2  # knowledge + infra
    eventbridge_cost = events_per_month * p["eventbridge_per_event"]

    monitoring_total = cw_metrics_cost + cw_alarms_standard + cw_alarms_anomaly + eventbridge_cost

    # ---- Healing compute (Lambda) ----
    # Monitor Lambda: runs on every EventBridge trigger
    monitor_invocations = events_per_month
    # Healing Lambda: runs on each healing event
    healing_invocations = s["avg_healing_events_per_month"]
    total_lambda_invocations = monitor_invocations + healing_invocations

    lambda_request_cost = total_lambda_invocations * p["lambda_per_request"]
    lambda_compute_gb_sec = (
        total_lambda_invocations
        * (p["lambda_memory_mb"] / 1024)
        * (p["lambda_avg_duration_ms"] / 1000)
    )
    lambda_compute_cost = lambda_compute_gb_sec * p["lambda_per_gb_sec"]
    lambda_total = lambda_request_cost + lambda_compute_cost

    # ---- Embedding API (Bedrock Titan) ----
    # Re-embedding per healing event: all docs × avg tokens
    tokens_per_healing = s["num_sources"] * s["avg_tokens_per_doc"]
    embed_cost_per_healing = (tokens_per_healing / 1000) * p["bedrock_titan_embed_per_1k_tokens"]
    embedding_total = embed_cost_per_healing * s["avg_healing_events_per_month"]

    # ---- Validation queries (Bedrock Claude) ----
    # After each healing, run validation queries
    val_input_tokens = s["validation_queries"] * s["avg_query_input_tokens"]
    val_output_tokens = s["validation_queries"] * s["avg_query_output_tokens"]
    val_cost_per_healing = (
        (val_input_tokens / 1000) * p["bedrock_claude_input_per_1k_tokens"]
        + (val_output_tokens / 1000) * p["bedrock_claude_output_per_1k_tokens"]
    )
    validation_total = val_cost_per_healing * s["avg_healing_events_per_month"]

    # ---- State store (DynamoDB) ----
    dynamodb_write = s["dynamodb_wcu"] * 730 * p["dynamodb_wcu_per_month"]  # 730 hrs/month
    dynamodb_read = s["dynamodb_rcu"] * 730 * p["dynamodb_rcu_per_month"]
    dynamodb_total = dynamodb_write + dynamodb_read

    # ---- SNS (free tier) ----
    sns_total = 0.00

    # ---- Agent total ----
    agent_total = monitoring_total + lambda_total + embedding_total + validation_total + dynamodb_total + sns_total

    # ---- Manual maintenance cost ----
    manual_weekly = s["manual_hours_per_week"] * s["developer_hourly_rate"]
    manual_monthly = manual_weekly * 4.33  # avg weeks per month
    # Manual still needs embedding costs for re-indexing
    manual_embed_cost = embedding_total  # same re-indexing work, just done manually

    # Build itemized breakdown
    breakdown = {
        "Monitoring (CloudWatch + EventBridge)": {
            "agent": round(monitoring_total, 2),
            "manual": 0.00,
            "detail": {
                "CloudWatch metrics": round(cw_metrics_cost, 2),
                "Standard alarms": round(cw_alarms_standard, 2),
                "Anomaly detection alarms": round(cw_alarms_anomaly, 2),
                "EventBridge events": round(eventbridge_cost, 4),
            },
        },
        "Healing compute (Lambda)": {
            "agent": round(lambda_total, 2),
            "manual": 0.00,
            "detail": {
                "Request cost": round(lambda_request_cost, 6),
                "Compute cost": round(lambda_compute_cost, 4),
                "Total invocations": int(total_lambda_invocations),
            },
        },
        "Embedding API (Bedrock Titan)": {
            "agent": round(embedding_total, 2),
            "manual": round(manual_embed_cost, 2),
            "detail": {
                "Tokens per healing": int(tokens_per_healing),
                "Cost per healing event": round(embed_cost_per_healing, 4),
                "Events per month": s["avg_healing_events_per_month"],
            },
        },
        "Validation queries (Bedrock Claude)": {
            "agent": round(validation_total, 2),
            "manual": 0.00,
            "detail": {
                "Queries per healing": s["validation_queries"],
                "Cost per healing event": round(val_cost_per_healing, 4),
            },
        },
        "State store (DynamoDB)": {
            "agent": round(dynamodb_total, 2),
            "manual": 0.00,
        },
        "Developer time": {
            "agent": 0.00,
            "manual": round(manual_monthly, 2),
            "detail": {
                "Hours per week": s["manual_hours_per_week"],
                "Hourly rate": s["developer_hourly_rate"],
                "Weeks per month": 4.33,
            },
        },
    }

    agent_grand_total = round(agent_total, 2)
    manual_grand_total = round(manual_monthly + manual_embed_cost, 2)
    savings_pct = round((1 - agent_grand_total / manual_grand_total) * 100, 1)

    # Print summary table
    print("\n  MONTHLY COST COMPARISON")
    print("  " + "-" * 62)
    print(f"  {'Component':<38} {'Agent':>10} {'Manual':>10}")
    print("  " + "-" * 62)
    for component, costs in breakdown.items():
        agent_str = f"${costs['agent']:.2f}"
        manual_str = f"${costs['manual']:.2f}" if costs['manual'] > 0 else "—"
        print(f"  {component:<38} {agent_str:>10} {manual_str:>10}")
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<38} {'$' + str(agent_grand_total):>10} {'$' + str(manual_grand_total):>10}")
    print(f"\n  Cost savings: {savings_pct}%")
    print(f"  Agent operates at {round(agent_grand_total / manual_grand_total * 100, 1)}% of manual cost")

    full_results = {
        "experiment": "AWS Cost Projection",
        "config": {
            "pricing": AWS_PRICING,
            "scenario": SCENARIO,
        },
        "breakdown": breakdown,
        "totals": {
            "agent_monthly": agent_grand_total,
            "manual_monthly": manual_grand_total,
            "savings_pct": savings_pct,
            "agent_as_pct_of_manual": round(agent_grand_total / manual_grand_total * 100, 1),
        },
    }

    return full_results
