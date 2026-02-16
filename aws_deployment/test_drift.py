"""Test drift detection by modifying a document in S3 and invoking the Lambda.

This is the DEMO script for the March presentation. It:
  1. Picks a document (default: prod-001, CloudRAG Pro Plan)
  2. Shows the original content
  3. Applies a modification (default: 3x price change, $49.99 → $149.97)
  4. Uploads the modified version to S3
  5. Invokes the Lambda function manually
  6. Shows the detection result from DynamoDB

Usage:
  python3 test_drift.py              # Default: prod-001, level 3 (3x price)
  python3 test_drift.py prod-002 4   # Custom: prod-002, level 4 (partial rewrite)
  python3 test_drift.py --reset      # Reset all documents to original content

Modification levels:
  0: No change (control test)
  1: Formatting only (should detect hash change but low drift score)
  2: Small numerical ~1% (hash catches it, embedding misses it)
  3: Large numerical 3x (hash catches it, embedding still misses it)
  4: Partial semantic rewrite (both hash and embedding detect it)
  5: Full content rewrite (maximum drift)
"""

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone

import boto3

# ─── Configuration ───
REGION = "us-east-1"
TABLE_NAME = "rag-healing-state"
FUNCTION_NAME = "self-healing-rag-monitor"

# ─── AWS clients ───
sts = boto3.client("sts", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)


def get_bucket_name():
    account_id = sts.get_caller_identity()["Account"]
    return f"self-healing-rag-{account_id}"


def apply_modification(content, level):
    """Apply a modification to document content. Same logic as experiments/test_data.py."""
    if level == 0:
        return content, "No change"

    if level == 1:
        modified = content.replace(". ", ".  ", 1)
        modified = modified.replace("the ", "The ", 1)
        return modified, "Formatting only (extra space, capitalization)"

    if level == 2:
        numbers = re.findall(r'\$[\d,]+\.?\d*|\d+\.?\d*%|\d{2,}', content)
        if numbers:
            target = numbers[0]
            if '$' in target:
                val = float(target.replace('$', '').replace(',', ''))
                new_val = val * 1.01
                new_str = f"${new_val:.2f}"
                if new_str == target:
                    new_val = val + 0.01
                    new_str = f"${new_val:.2f}"
                return content.replace(target, new_str, 1), f"Small numerical: {target} → {new_str}"
        return content, "No numbers found"

    if level == 3:
        numbers = re.findall(r'\$[\d,]+\.?\d*|\d+\.?\d*%|\d{2,}', content)
        if numbers:
            target = numbers[0]
            if '$' in target:
                val = float(target.replace('$', '').replace(',', ''))
                new_val = val * 3
                new_str = f"${new_val:.2f}"
                return content.replace(target, new_str, 1), f"Large numerical (3x): {target} → {new_str}"
        return content, "No numbers found"

    if level == 4:
        sentences = content.split('. ')
        if len(sentences) >= 2:
            sentences[-1] = "This feature was recently updated with significant performance improvements"
            return '. '.join(sentences), "Partial rewrite (replaced last sentence)"
        return content + " This feature was recently updated.", "Appended new sentence"

    if level == 5:
        return (
            "This product has been completely redesigned with a new architecture. "
            "All previous specifications are deprecated. The new system uses a "
            "microservices-based approach with Kubernetes orchestration, supporting "
            "up to 10,000 concurrent connections. Pricing has been restructured to "
            "a usage-based model starting at $0.001 per query."
        ), "Full content rewrite"

    raise ValueError(f"Invalid level: {level}")


def reset_all_documents(bucket_name):
    """Re-upload all original documents and re-seed DynamoDB (calls seed_data logic)."""
    print("Resetting all documents to original content...")
    print("Run: python3 seed_data.py")
    print("(This will re-upload all 20 documents and re-compute embeddings)")


def main():
    bucket_name = get_bucket_name()
    table = dynamodb.Table(TABLE_NAME)

    # Parse arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_all_documents(bucket_name)
        return

    doc_id = sys.argv[1] if len(sys.argv) > 1 else "prod-001"
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    s3_key = f"documents/{doc_id}.json"

    print(f"{'='*60}")
    print(f"Self-Healing RAG — Drift Detection Demo")
    print(f"{'='*60}")
    print(f"  Document: {doc_id}")
    print(f"  Modification level: {level}")
    print(f"  S3 bucket: {bucket_name}")
    print()

    # ── Step 1: Read current document from S3 ──
    print("Step 1: Reading current document from S3...")
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
        doc_data = json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        print(f"  ERROR: Document {s3_key} not found in S3.")
        print(f"  Have you run: python3 seed_data.py ?")
        return

    original_content = doc_data["content"]
    print(f"  Title: {doc_data['title']}")
    print(f"  Original: {original_content[:100]}...")
    print()

    # ── Step 2: Read current DynamoDB state ──
    print("Step 2: Current DynamoDB state...")
    db_item = table.get_item(Key={"doc_id": doc_id}).get("Item", {})
    print(f"  Hash: {db_item.get('content_hash', 'N/A')[:16]}...")
    print(f"  Drift score: {db_item.get('drift_score', 'N/A')}")
    print(f"  Heal count: {db_item.get('heal_count', 0)}")
    print(f"  Last checked: {db_item.get('last_checked', 'N/A')}")
    print()

    # ── Step 3: Apply modification ──
    print(f"Step 3: Applying level-{level} modification...")
    modified_content, description = apply_modification(original_content, level)
    print(f"  Change: {description}")
    print(f"  Modified: {modified_content[:100]}...")

    # Show the diff
    old_hash = hashlib.sha256(original_content.encode()).hexdigest()[:16]
    new_hash = hashlib.sha256(modified_content.encode()).hexdigest()[:16]
    print(f"  Hash before: {old_hash}...")
    print(f"  Hash after:  {new_hash}...")
    print(f"  Hash changed: {old_hash != new_hash}")
    print()

    # ── Step 4: Upload modified document to S3 ──
    print("Step 4: Uploading modified document to S3...")
    doc_data["content"] = modified_content
    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=json.dumps(doc_data),
        ContentType="application/json",
    )
    print(f"  ✓ Uploaded to s3://{bucket_name}/{s3_key}")
    print()

    # ── Step 5: Invoke Lambda ──
    print("Step 5: Invoking Lambda (self-healing-rag-monitor)...")
    print("  This triggers the full MAPE-K loop: Monitor → Analyze → Plan → Execute")
    t0 = time.time()
    response = lambda_client.invoke(
        FunctionName=FUNCTION_NAME,
        InvocationType="RequestResponse",  # Synchronous — wait for result
        Payload=json.dumps({"source": "test_drift.py", "doc_id": doc_id}),
    )
    elapsed = time.time() - t0
    payload = json.loads(response["Payload"].read())
    print(f"  ✓ Lambda completed in {elapsed:.1f}s")
    print(f"  Status: {response['StatusCode']}")

    if "errorMessage" in payload:
        print(f"  ERROR: {payload['errorMessage']}")
        return

    print(f"  Results: {payload.get('checked', '?')} checked, "
          f"{payload.get('healed', '?')} healed, "
          f"{payload.get('unchanged', '?')} unchanged")
    print()

    # ── Step 6: Show updated DynamoDB state ──
    print("Step 6: Updated DynamoDB state...")
    db_item = table.get_item(Key={"doc_id": doc_id}).get("Item", {})
    print(f"  Hash: {db_item.get('content_hash', 'N/A')[:16]}...")
    print(f"  Drift score: {db_item.get('drift_score', 'N/A')}")
    print(f"  Drift type: {db_item.get('drift_type', 'N/A')}")
    print(f"  Heal count: {db_item.get('heal_count', 0)}")
    print(f"  Last healed: {db_item.get('last_healed', 'N/A')}")
    print()

    # ── Summary ──
    drift_score = float(db_item.get("drift_score", 0))
    print(f"{'='*60}")
    print(f"RESULT: {'DRIFT DETECTED AND HEALED' if drift_score > 0 else 'NO DRIFT (already matched)'}")
    print(f"{'='*60}")
    print(f"  Drift score: {drift_score:.4f}")
    print(f"  Threshold: 0.08")
    if drift_score > 0.08:
        print(f"  Classification: SEMANTIC REWRITE (score > threshold)")
    elif drift_score > 0:
        print(f"  Classification: NUMERICAL CHANGE (hash detected, below semantic threshold)")
    else:
        print(f"  Classification: UNCHANGED")
    print()
    print(f"  View CloudWatch metrics:")
    print(f"    AWS Console → CloudWatch → Metrics → SelfHealingRAG → DriftScore")
    print()
    print(f"  To reset this document, run: python3 seed_data.py")


if __name__ == "__main__":
    main()
