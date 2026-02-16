"""Seed the AWS deployment with test documents.

This script:
  1. Uploads 20 test documents to S3 as JSON files
  2. Computes SHA-256 hashes for each document
  3. Generates chunk-level embeddings via Bedrock Titan Embeddings v2
  4. Seeds DynamoDB with the baseline state (hash, embeddings, timestamps)

Run this ONCE after deploy.sh to initialize the system.
Requires: boto3, AWS CLI configured, Bedrock model access enabled.

Usage: python3 seed_data.py
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import boto3

# ─── Configuration ───
# Must match the values in deploy.sh / lambda_function.py
REGION = "us-east-1"
BUCKET_NAME = None  # Auto-detected from account ID
TABLE_NAME = "rag-healing-state"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# ─── AWS clients ───
sts = boto3.client("sts", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def get_bucket_name():
    """Get the S3 bucket name (matches deploy.sh naming convention)."""
    account_id = sts.get_caller_identity()["Account"]
    return f"self-healing-rag-{account_id}"


def get_corpus():
    """Return the same 20-document corpus used in local experiments.

    Duplicated here to avoid import path issues. These are the exact same
    documents from experiments/test_data.py.
    """
    docs = [
        {
            "doc_id": "prod-001",
            "category": "product_spec",
            "title": "CloudRAG Pro Plan",
            "content": "CloudRAG Pro supports up to 50 concurrent users with 10GB vector storage. "
            "The plan includes real-time indexing, automatic backup every 6 hours, and "
            "priority support with a 2-hour SLA. Pricing is $49.99 per month billed annually. "
            "Maximum document size is 25MB with support for PDF, DOCX, and HTML formats.",
        },
        {
            "doc_id": "prod-002",
            "category": "product_spec",
            "title": "CloudRAG Enterprise Plan",
            "content": "CloudRAG Enterprise offers unlimited users with 500GB vector storage and "
            "dedicated infrastructure. Features include custom embedding models, SSO integration, "
            "99.99% uptime SLA, and 24/7 dedicated support. Pricing starts at $999 per month. "
            "Supports all document formats including audio and video transcription.",
        },
        {
            "doc_id": "prod-003",
            "category": "product_spec",
            "title": "Vector Index Configuration",
            "content": "The HNSW index uses M=16 connections per layer with ef_construction=200 for "
            "build-time accuracy. Search uses ef_search=100 by default. The index supports "
            "cosine similarity, dot product, and L2 distance metrics. Maximum dimensions "
            "supported is 2048. Index rebuild takes approximately 45 minutes for 1M vectors.",
        },
        {
            "doc_id": "prod-004",
            "category": "product_spec",
            "title": "Embedding Model Specifications",
            "content": "The default embedding model produces 384-dimensional vectors with a maximum "
            "input length of 512 tokens. Throughput is 1200 documents per second on a single "
            "GPU instance. The model achieves 0.89 NDCG@10 on the MTEB benchmark. Fine-tuning "
            "is available for enterprise customers with a minimum of 10,000 training pairs.",
        },
        {
            "doc_id": "prod-005",
            "category": "product_spec",
            "title": "API Rate Limits",
            "content": "The API enforces rate limits of 100 requests per second for query endpoints and "
            "50 requests per second for indexing endpoints. Burst capacity allows up to 200 "
            "requests per second for 30 seconds. Rate limit headers include X-RateLimit-Remaining "
            "and X-RateLimit-Reset. Enterprise customers can request custom limits up to 1000 rps.",
        },
        {
            "doc_id": "faq-001",
            "category": "faq",
            "title": "How to connect data sources",
            "content": "To connect a data source, navigate to Settings > Data Sources > Add New. Supported "
            "sources include Amazon S3, Google Drive, Confluence, Notion, and SharePoint. "
            "Authentication uses OAuth 2.0 for cloud sources and IAM roles for AWS services. "
            "Initial sync takes 15-30 minutes depending on corpus size.",
        },
        {
            "doc_id": "faq-002",
            "category": "faq",
            "title": "Troubleshooting slow queries",
            "content": "Slow queries are typically caused by three factors: oversized chunks exceeding 1000 "
            "tokens, insufficient vector index warm-up after cold starts, or network latency to "
            "the vector database. Solutions include reducing chunk size to 300-500 tokens, enabling "
            "provisioned concurrency, and deploying in the same region as the vector store.",
        },
        {
            "doc_id": "faq-003",
            "category": "faq",
            "title": "Understanding retrieval scores",
            "content": "Retrieval scores range from 0.0 to 1.0, where 1.0 indicates perfect semantic match. "
            "Scores above 0.85 are considered high confidence, 0.70-0.85 medium confidence, and "
            "below 0.70 low confidence. The system returns the top 5 results by default. "
            "Minimum score threshold can be configured in the query parameters.",
        },
        {
            "doc_id": "faq-004",
            "category": "faq",
            "title": "Data retention and deletion",
            "content": "Documents are retained indefinitely unless explicitly deleted. Deletion removes both "
            "the source document and all associated vector embeddings within 24 hours. For GDPR "
            "compliance, use the bulk delete API with user_id parameter to remove all documents "
            "associated with a specific user. Deletion is irreversible after 30-day grace period.",
        },
        {
            "doc_id": "faq-005",
            "category": "faq",
            "title": "Multi-language support",
            "content": "The platform supports 50 languages for document ingestion and 30 languages for "
            "query processing. Cross-lingual retrieval is supported for the top 12 languages "
            "using multilingual embeddings. Translation quality varies: European languages achieve "
            "95% accuracy, Asian languages 88%, and right-to-left languages 82%.",
        },
        {
            "doc_id": "price-001",
            "category": "pricing",
            "title": "Compute pricing overview",
            "content": "Lambda functions are billed at $0.20 per 1 million requests plus $0.0000166667 per "
            "GB-second of compute time. Free tier includes 1 million requests and 400,000 GB-seconds "
            "per month. Average cost for a RAG query Lambda with 256MB memory and 200ms duration "
            "is approximately $0.0000035 per invocation.",
        },
        {
            "doc_id": "price-002",
            "category": "pricing",
            "title": "Storage pricing breakdown",
            "content": "OpenSearch Serverless charges $0.24 per OCU-hour for indexing and $0.24 per OCU-hour "
            "for search, with a minimum of 2 OCUs each. S3 storage costs $0.023 per GB per month "
            "for standard tier. DynamoDB on-demand pricing is $1.25 per million write requests and "
            "$0.25 per million read requests.",
        },
        {
            "doc_id": "price-003",
            "category": "pricing",
            "title": "Embedding API costs",
            "content": "Amazon Bedrock Titan Embeddings v2 costs $0.0001 per 1,000 input tokens. For a "
            "typical 500-token document, embedding cost is $0.00005 per document. Re-embedding "
            "a 20-document corpus costs approximately $0.001. Monthly cost for continuous "
            "monitoring with 8 re-indexing events averages $0.008.",
        },
        {
            "doc_id": "price-004",
            "category": "pricing",
            "title": "Monitoring costs",
            "content": "CloudWatch custom metrics cost $0.30 per metric per month for the first 10,000 "
            "metrics. Standard alarms cost $0.10 each per month. Anomaly detection alarms "
            "cost $3.00 each per month. EventBridge scheduled rules cost $1.00 per million "
            "invocations. Total monitoring cost for a typical setup: $3.50 per month.",
        },
        {
            "doc_id": "price-005",
            "category": "pricing",
            "title": "Cost comparison summary",
            "content": "A fully automated self-healing RAG system costs approximately $12-15 per month "
            "on AWS serverless. Manual maintenance requires 4-8 hours per week of developer "
            "time at $50/hour, totaling $800-1600 per month. The automated approach represents "
            "a 98.5% cost reduction while providing continuous rather than periodic monitoring.",
        },
        {
            "doc_id": "policy-001",
            "category": "policy",
            "title": "Data processing agreement",
            "content": "All customer data is processed in accordance with SOC 2 Type II and ISO 27001 "
            "standards. Data is encrypted at rest using AES-256 and in transit using TLS 1.3. "
            "Processing occurs exclusively in the customer-selected AWS region. Sub-processors "
            "include Amazon Web Services and Anthropic (for LLM inference only).",
        },
        {
            "doc_id": "policy-002",
            "category": "policy",
            "title": "Service level agreement",
            "content": "The platform guarantees 99.9% uptime for Pro plans and 99.99% for Enterprise plans, "
            "measured monthly. Planned maintenance windows are excluded and announced 72 hours in "
            "advance. Credit schedule: 10% for 99.0-99.9%, 25% for 95.0-99.0%, 50% below 95.0%. "
            "Maximum credit per month is capped at 50% of monthly fees.",
        },
        {
            "doc_id": "policy-003",
            "category": "policy",
            "title": "Acceptable use policy",
            "content": "The service may not be used for generating illegal content, impersonation, spam, or "
            "any activity that violates applicable law. Rate limiting applies to prevent abuse. "
            "Accounts exceeding 10TB of storage or 1 million queries per day require Enterprise "
            "plan approval. Violation results in 30-day notice before account suspension.",
        },
        {
            "doc_id": "policy-004",
            "category": "policy",
            "title": "Backup and disaster recovery",
            "content": "Automated backups run every 6 hours with 30-day retention. Cross-region replication "
            "is available for Enterprise plans. Recovery time objective (RTO) is 4 hours for Pro "
            "and 1 hour for Enterprise. Recovery point objective (RPO) is 6 hours for Pro and "
            "15 minutes for Enterprise. Annual disaster recovery testing is included.",
        },
        {
            "doc_id": "policy-005",
            "category": "policy",
            "title": "Change management procedure",
            "content": "All infrastructure changes follow a CI/CD pipeline with mandatory code review, "
            "automated testing, and staged rollout (canary 5%, then 25%, then 100%). Rollback "
            "is automatic if error rate exceeds 1% during canary phase. Change freeze periods "
            "apply during major holidays and customer-designated blackout windows.",
        },
    ]
    return docs


def compute_hash(text):
    """SHA-256 content hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(text):
    """Split into overlapping sentence-pair chunks (matches Lambda logic)."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    chunks = []
    for i in range(len(sentences)):
        chunk = ". ".join(sentences[i : i + 2])
        if not chunk.endswith("."):
            chunk += "."
        chunks.append(chunk)
    return chunks


def get_embedding(text):
    """Get embedding from Bedrock Titan Embeddings v2."""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def main():
    global BUCKET_NAME
    BUCKET_NAME = get_bucket_name()
    table = dynamodb.Table(TABLE_NAME)
    corpus = get_corpus()

    print(f"Seeding {len(corpus)} documents")
    print(f"  S3 bucket: {BUCKET_NAME}")
    print(f"  DynamoDB table: {TABLE_NAME}")
    print(f"  Embedding model: {EMBEDDING_MODEL_ID}")
    print()

    for i, doc in enumerate(corpus):
        doc_id = doc["doc_id"]
        content = doc["content"]
        s3_key = f"documents/{doc_id}.json"

        print(f"  [{i + 1}/{len(corpus)}] {doc_id}: {doc['title']}")

        # 1. Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(doc),
            ContentType="application/json",
        )
        print(f"    ✓ S3: s3://{BUCKET_NAME}/{s3_key}")

        # 2. Compute hash
        content_hash = compute_hash(content)

        # 3. Generate chunk embeddings via Bedrock
        chunks = chunk_document(content)
        chunk_embeddings = []
        for chunk in chunks:
            emb = get_embedding(chunk)
            chunk_embeddings.append({"text": chunk, "embedding": emb})
            time.sleep(0.1)  # Be gentle with Bedrock rate limits

        print(f"    ✓ Embeddings: {len(chunk_embeddings)} chunks × 1024 dims")

        # 4. Seed DynamoDB
        now = datetime.now(timezone.utc).isoformat()
        table.put_item(
            Item={
                "doc_id": doc_id,
                "s3_key": s3_key,
                "category": doc["category"],
                "title": doc["title"],
                "content_hash": content_hash,
                "chunk_embeddings": json.dumps(chunk_embeddings),
                "drift_score": "0.0",
                "drift_type": "none",
                "heal_count": 0,
                "last_checked": now,
                "last_healed": now,
                "created_at": now,
            }
        )
        print(
            f"    ✓ DynamoDB: seeded with hash + {len(chunk_embeddings)} chunk embeddings"
        )

    print(f"\nDone! {len(corpus)} documents seeded.")
    print(f"\nTo verify:")
    print(f"  aws s3 ls s3://{BUCKET_NAME}/documents/")
    print(
        f"  aws dynamodb scan --table-name {TABLE_NAME} --select COUNT --region {REGION}"
    )


if __name__ == "__main__":
    main()
