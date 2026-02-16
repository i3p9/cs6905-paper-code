"""Self-Healing RAG Monitor — AWS Lambda Function

Implements Loop 1 (Knowledge Healing) of the MAPE-K autonomic architecture:
  Monitor:  Scans all documents in S3 on an EventBridge schedule
  Analyze:  SHA-256 hash comparison + Bedrock Titan chunk-level cosine distance
  Plan:     Classifies drift severity (formatting, numerical, semantic rewrite)
  Execute:  Updates DynamoDB state, publishes CloudWatch metrics, sends SNS alerts
  Knowledge: DynamoDB stores hashes, embeddings, and healing audit trail

Designed for Lambda: pure Python only (no numpy, no ML libraries).
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

import boto3

# ─── Configuration from environment variables ───
S3_BUCKET = os.environ.get("S3_BUCKET", "self-healing-rag")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "rag-healing-state")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.08"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ─── AWS clients ───
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
sns = boto3.client("sns", region_name=AWS_REGION)

# Bedrock Titan Embeddings v2 model ID
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


# ─── Pure Python math (no numpy needed) ───

def dot_product(a, b):
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def magnitude(v):
    """Compute vector magnitude."""
    return math.sqrt(sum(x * x for x in v))


def cosine_distance(a, b):
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    dot = dot_product(a, b)
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 1.0
    similarity = dot / (mag_a * mag_b)
    return 1.0 - similarity


# ─── Text processing ───

def compute_hash(text):
    """SHA-256 content hash for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(text):
    """Split document into overlapping sentence-pair chunks (simulates RAG chunking).

    Same logic as the local PoC experiment: split on '. ', then create
    overlapping windows of 2 sentences.
    """
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks = []
    for i in range(len(sentences)):
        chunk = ". ".join(sentences[i:i + 2])
        if not chunk.endswith("."):
            chunk += "."
        chunks.append(chunk)
    return chunks


# ─── Bedrock Titan Embeddings ───

def get_embedding(text):
    """Get embedding vector from Amazon Bedrock Titan Embeddings v2.

    Returns a list of floats (1024 dimensions for Titan v2).
    """
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    })
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def get_chunk_embeddings(text):
    """Chunk a document and embed each chunk. Returns list of (chunk_text, embedding)."""
    chunks = chunk_document(text)
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append({"text": chunk, "embedding": emb})
    return embeddings


# ─── Drift detection ───

def max_chunk_drift(old_chunk_embeddings, new_text):
    """Compute maximum chunk-level cosine distance between old and new versions.

    For each chunk in the new document, finds the closest chunk in the old
    document and reports the maximum drift across all new chunks.
    This is the same algorithm as the local PoC (experiment_1_knowledge.py).

    Args:
        old_chunk_embeddings: List of {"text": str, "embedding": list} from DynamoDB
        new_text: The current document text from S3

    Returns:
        float: Maximum cosine distance (0.0 = identical, 1.0 = completely different)
    """
    new_chunks = chunk_document(new_text)
    if not new_chunks or not old_chunk_embeddings:
        return 0.0

    old_embs = [ce["embedding"] for ce in old_chunk_embeddings]
    max_dist = 0.0

    for new_chunk in new_chunks:
        new_emb = get_embedding(new_chunk)
        # Find the minimum distance to any old chunk (best match)
        best_dist = min(cosine_distance(new_emb, old_emb) for old_emb in old_embs)
        max_dist = max(max_dist, best_dist)

    return max_dist


# ─── CloudWatch metrics ───

def publish_metrics(doc_id, drift_score, healed):
    """Publish custom CloudWatch metrics for the MAPE-K dashboard."""
    metrics = [
        {
            "MetricName": "DriftScore",
            "Dimensions": [{"Name": "DocumentId", "Value": doc_id}],
            "Value": drift_score,
            "Unit": "None",
        },
    ]
    if healed:
        metrics.append({
            "MetricName": "HealingEvents",
            "Dimensions": [{"Name": "DocumentId", "Value": doc_id}],
            "Value": 1,
            "Unit": "Count",
        })

    cloudwatch.put_metric_data(
        Namespace="SelfHealingRAG",
        MetricData=metrics,
    )


# ─── SNS notifications ───

def send_alert(doc_id, drift_score, drift_type):
    """Send SNS notification when drift is detected."""
    if not SNS_TOPIC_ARN:
        print(f"  [SNS] No topic ARN configured, skipping alert for {doc_id}")
        return

    message = (
        f"Knowledge Drift Detected!\n\n"
        f"Document: {doc_id}\n"
        f"Drift Score: {drift_score:.4f}\n"
        f"Drift Type: {drift_type}\n"
        f"Threshold: {DRIFT_THRESHOLD}\n"
        f"Action: Document re-indexed automatically\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
    )
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"RAG Drift Alert: {doc_id}",
        Message=message,
    )
    print(f"  [SNS] Alert sent for {doc_id}")


# ─── Main handler ───

def lambda_handler(event, context):
    """Main Lambda entry point — triggered by EventBridge schedule or manual invoke.

    Scans all documents tracked in DynamoDB, checks S3 for changes,
    and performs healing when drift is detected.
    """
    print(f"Self-Healing RAG Monitor started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Config: bucket={S3_BUCKET}, table={DYNAMODB_TABLE}, threshold={DRIFT_THRESHOLD}")

    # Scan DynamoDB for all tracked documents
    response = table.scan()
    items = response.get("Items", [])
    print(f"Monitoring {len(items)} documents")

    results = {
        "total": len(items),
        "checked": 0,
        "unchanged": 0,
        "healed": 0,
        "errors": 0,
        "details": [],
    }

    for item in items:
        doc_id = item["doc_id"]
        s3_key = item.get("s3_key", f"documents/{doc_id}.json")
        stored_hash = item.get("content_hash", "")
        old_chunk_embs = item.get("chunk_embeddings", [])

        try:
            # ── MONITOR: Fetch current document from S3 ──
            s3_obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            doc_data = json.loads(s3_obj["Body"].read().decode("utf-8"))
            current_text = doc_data["content"]

            # ── ANALYZE: Hash comparison (fast, cheap) ──
            current_hash = compute_hash(current_text)
            hash_changed = current_hash != stored_hash
            results["checked"] += 1

            if not hash_changed:
                # No change detected — skip expensive embedding comparison
                print(f"  [{doc_id}] No change (hash match)")
                results["unchanged"] += 1
                publish_metrics(doc_id, 0.0, False)

                # Update last_checked timestamp
                table.update_item(
                    Key={"doc_id": doc_id},
                    UpdateExpression="SET last_checked = :ts",
                    ExpressionAttributeValues={":ts": datetime.now(timezone.utc).isoformat()},
                )
                results["details"].append({
                    "doc_id": doc_id, "status": "unchanged", "drift_score": 0.0
                })
                continue

            # Hash changed — now do semantic analysis
            print(f"  [{doc_id}] Hash changed, computing semantic drift...")

            # ── ANALYZE: Chunk-level embedding comparison (expensive but informative) ──
            drift_score = 0.0
            drift_type = "formatting_only"

            if old_chunk_embs:
                drift_score = max_chunk_drift(old_chunk_embs, current_text)
                if drift_score > DRIFT_THRESHOLD:
                    drift_type = "semantic_rewrite"
                else:
                    drift_type = "numerical_change"
            else:
                # No stored embeddings — treat as new document
                drift_score = 1.0
                drift_type = "new_document"

            print(f"  [{doc_id}] Drift score: {drift_score:.4f}, type: {drift_type}")

            # ── PLAN + EXECUTE: Re-index (heal) the document ──
            new_chunk_embs = get_chunk_embeddings(current_text)

            # Convert embeddings to DynamoDB-compatible format
            # DynamoDB doesn't support float lists natively in all SDKs,
            # so we store as JSON string
            chunk_embs_for_db = json.dumps(new_chunk_embs)

            # ── KNOWLEDGE: Update DynamoDB state ──
            table.update_item(
                Key={"doc_id": doc_id},
                UpdateExpression=(
                    "SET content_hash = :hash, "
                    "chunk_embeddings = :embs, "
                    "drift_score = :score, "
                    "drift_type = :dtype, "
                    "last_healed = :ts, "
                    "last_checked = :ts, "
                    "heal_count = if_not_exists(heal_count, :zero) + :one"
                ),
                ExpressionAttributeValues={
                    ":hash": current_hash,
                    ":embs": chunk_embs_for_db,
                    ":score": str(drift_score),  # DynamoDB Number type via string
                    ":dtype": drift_type,
                    ":ts": datetime.now(timezone.utc).isoformat(),
                    ":zero": 0,
                    ":one": 1,
                },
            )

            # Publish CloudWatch metrics
            publish_metrics(doc_id, drift_score, True)

            # Send SNS alert
            send_alert(doc_id, drift_score, drift_type)

            results["healed"] += 1
            results["details"].append({
                "doc_id": doc_id,
                "status": "healed",
                "drift_score": drift_score,
                "drift_type": drift_type,
            })

        except Exception as e:
            print(f"  [{doc_id}] ERROR: {e}")
            results["errors"] += 1
            results["details"].append({
                "doc_id": doc_id, "status": "error", "error": str(e)
            })

    # Summary
    print(f"\nMonitoring complete:")
    print(f"  Total: {results['total']}, Checked: {results['checked']}, "
          f"Unchanged: {results['unchanged']}, Healed: {results['healed']}, "
          f"Errors: {results['errors']}")

    return results
