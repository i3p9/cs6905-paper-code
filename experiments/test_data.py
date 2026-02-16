"""20-document corpus with 6-level modification engine for drift experiments."""

import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from config import RANDOM_SEED

random.seed(RANDOM_SEED)


@dataclass
class Document:
    doc_id: str
    category: str
    title: str
    content: str


@dataclass
class Modification:
    level: int
    description: str
    expected_hash_change: bool
    expected_semantic_change: bool  # True = should be detected as meaningful drift


def get_corpus() -> List[Document]:
    """Return 20 documents across 4 categories."""
    docs = []

    # --- Product Specs (5) ---
    docs.append(Document(
        "prod-001", "product_spec", "CloudRAG Pro Plan",
        "CloudRAG Pro supports up to 50 concurrent users with 10GB vector storage. "
        "The plan includes real-time indexing, automatic backup every 6 hours, and "
        "priority support with a 2-hour SLA. Pricing is $49.99 per month billed annually. "
        "Maximum document size is 25MB with support for PDF, DOCX, and HTML formats."
    ))
    docs.append(Document(
        "prod-002", "product_spec", "CloudRAG Enterprise Plan",
        "CloudRAG Enterprise offers unlimited users with 500GB vector storage and "
        "dedicated infrastructure. Features include custom embedding models, SSO integration, "
        "99.99% uptime SLA, and 24/7 dedicated support. Pricing starts at $999 per month. "
        "Supports all document formats including audio and video transcription."
    ))
    docs.append(Document(
        "prod-003", "product_spec", "Vector Index Configuration",
        "The HNSW index uses M=16 connections per layer with ef_construction=200 for "
        "build-time accuracy. Search uses ef_search=100 by default. The index supports "
        "cosine similarity, dot product, and L2 distance metrics. Maximum dimensions "
        "supported is 2048. Index rebuild takes approximately 45 minutes for 1M vectors."
    ))
    docs.append(Document(
        "prod-004", "product_spec", "Embedding Model Specifications",
        "The default embedding model produces 384-dimensional vectors with a maximum "
        "input length of 512 tokens. Throughput is 1200 documents per second on a single "
        "GPU instance. The model achieves 0.89 NDCG@10 on the MTEB benchmark. Fine-tuning "
        "is available for enterprise customers with a minimum of 10,000 training pairs."
    ))
    docs.append(Document(
        "prod-005", "product_spec", "API Rate Limits",
        "The API enforces rate limits of 100 requests per second for query endpoints and "
        "50 requests per second for indexing endpoints. Burst capacity allows up to 200 "
        "requests per second for 30 seconds. Rate limit headers include X-RateLimit-Remaining "
        "and X-RateLimit-Reset. Enterprise customers can request custom limits up to 1000 rps."
    ))

    # --- FAQs (5) ---
    docs.append(Document(
        "faq-001", "faq", "How to connect data sources",
        "To connect a data source, navigate to Settings > Data Sources > Add New. Supported "
        "sources include Amazon S3, Google Drive, Confluence, Notion, and SharePoint. "
        "Authentication uses OAuth 2.0 for cloud sources and IAM roles for AWS services. "
        "Initial sync takes 15-30 minutes depending on corpus size."
    ))
    docs.append(Document(
        "faq-002", "faq", "Troubleshooting slow queries",
        "Slow queries are typically caused by three factors: oversized chunks exceeding 1000 "
        "tokens, insufficient vector index warm-up after cold starts, or network latency to "
        "the vector database. Solutions include reducing chunk size to 300-500 tokens, enabling "
        "provisioned concurrency, and deploying in the same region as the vector store."
    ))
    docs.append(Document(
        "faq-003", "faq", "Understanding retrieval scores",
        "Retrieval scores range from 0.0 to 1.0, where 1.0 indicates perfect semantic match. "
        "Scores above 0.85 are considered high confidence, 0.70-0.85 medium confidence, and "
        "below 0.70 low confidence. The system returns the top 5 results by default. "
        "Minimum score threshold can be configured in the query parameters."
    ))
    docs.append(Document(
        "faq-004", "faq", "Data retention and deletion",
        "Documents are retained indefinitely unless explicitly deleted. Deletion removes both "
        "the source document and all associated vector embeddings within 24 hours. For GDPR "
        "compliance, use the bulk delete API with user_id parameter to remove all documents "
        "associated with a specific user. Deletion is irreversible after 30-day grace period."
    ))
    docs.append(Document(
        "faq-005", "faq", "Multi-language support",
        "The platform supports 50 languages for document ingestion and 30 languages for "
        "query processing. Cross-lingual retrieval is supported for the top 12 languages "
        "using multilingual embeddings. Translation quality varies: European languages achieve "
        "95% accuracy, Asian languages 88%, and right-to-left languages 82%."
    ))

    # --- Pricing (5) ---
    docs.append(Document(
        "price-001", "pricing", "Compute pricing overview",
        "Lambda functions are billed at $0.20 per 1 million requests plus $0.0000166667 per "
        "GB-second of compute time. Free tier includes 1 million requests and 400,000 GB-seconds "
        "per month. Average cost for a RAG query Lambda with 256MB memory and 200ms duration "
        "is approximately $0.0000035 per invocation."
    ))
    docs.append(Document(
        "price-002", "pricing", "Storage pricing breakdown",
        "OpenSearch Serverless charges $0.24 per OCU-hour for indexing and $0.24 per OCU-hour "
        "for search, with a minimum of 2 OCUs each. S3 storage costs $0.023 per GB per month "
        "for standard tier. DynamoDB on-demand pricing is $1.25 per million write requests and "
        "$0.25 per million read requests."
    ))
    docs.append(Document(
        "price-003", "pricing", "Embedding API costs",
        "Amazon Bedrock Titan Embeddings v2 costs $0.0001 per 1,000 input tokens. For a "
        "typical 500-token document, embedding cost is $0.00005 per document. Re-embedding "
        "a 20-document corpus costs approximately $0.001. Monthly cost for continuous "
        "monitoring with 8 re-indexing events averages $0.008."
    ))
    docs.append(Document(
        "price-004", "pricing", "Monitoring costs",
        "CloudWatch custom metrics cost $0.30 per metric per month for the first 10,000 "
        "metrics. Standard alarms cost $0.10 each per month. Anomaly detection alarms "
        "cost $3.00 each per month. EventBridge scheduled rules cost $1.00 per million "
        "invocations. Total monitoring cost for a typical setup: $3.50 per month."
    ))
    docs.append(Document(
        "price-005", "pricing", "Cost comparison summary",
        "A fully automated self-healing RAG system costs approximately $12-15 per month "
        "on AWS serverless. Manual maintenance requires 4-8 hours per week of developer "
        "time at $50/hour, totaling $800-1600 per month. The automated approach represents "
        "a 98.5% cost reduction while providing continuous rather than periodic monitoring."
    ))

    # --- Policies (5) ---
    docs.append(Document(
        "policy-001", "policy", "Data processing agreement",
        "All customer data is processed in accordance with SOC 2 Type II and ISO 27001 "
        "standards. Data is encrypted at rest using AES-256 and in transit using TLS 1.3. "
        "Processing occurs exclusively in the customer-selected AWS region. Sub-processors "
        "include Amazon Web Services and Anthropic (for LLM inference only)."
    ))
    docs.append(Document(
        "policy-002", "policy", "Service level agreement",
        "The platform guarantees 99.9% uptime for Pro plans and 99.99% for Enterprise plans, "
        "measured monthly. Planned maintenance windows are excluded and announced 72 hours in "
        "advance. Credit schedule: 10% for 99.0-99.9%, 25% for 95.0-99.0%, 50% below 95.0%. "
        "Maximum credit per month is capped at 50% of monthly fees."
    ))
    docs.append(Document(
        "policy-003", "policy", "Acceptable use policy",
        "The service may not be used for generating illegal content, impersonation, spam, or "
        "any activity that violates applicable law. Rate limiting applies to prevent abuse. "
        "Accounts exceeding 10TB of storage or 1 million queries per day require Enterprise "
        "plan approval. Violation results in 30-day notice before account suspension."
    ))
    docs.append(Document(
        "policy-004", "policy", "Backup and disaster recovery",
        "Automated backups run every 6 hours with 30-day retention. Cross-region replication "
        "is available for Enterprise plans. Recovery time objective (RTO) is 4 hours for Pro "
        "and 1 hour for Enterprise. Recovery point objective (RPO) is 6 hours for Pro and "
        "15 minutes for Enterprise. Annual disaster recovery testing is included."
    ))
    docs.append(Document(
        "policy-005", "policy", "Change management procedure",
        "All infrastructure changes follow a CI/CD pipeline with mandatory code review, "
        "automated testing, and staged rollout (canary 5%, then 25%, then 100%). Rollback "
        "is automatic if error rate exceeds 1% during canary phase. Change freeze periods "
        "apply during major holidays and customer-designated blackout windows."
    ))

    return docs


def apply_modification(doc: Document, level: int) -> Tuple[Document, Modification]:
    """Apply a modification at the given level (0-5) to a document.

    Returns (modified_doc, modification_metadata).
    """
    content = doc.content

    if level == 0:
        return (
            Document(doc.doc_id, doc.category, doc.title, content),
            Modification(0, "No change", False, False),
        )

    if level == 1:
        # Formatting only: add extra spaces, change capitalization of one word
        modified = content.replace(". ", ".  ", 1)  # double space after first period
        modified = modified.replace("the ", "The ", 1)  # capitalize one 'the'
        return (
            Document(doc.doc_id, doc.category, doc.title, modified),
            Modification(1, "Formatting only", True, False),
        )

    if level == 2:
        # Small numerical change — the hard case
        # Find a number and nudge it slightly
        numbers = re.findall(r'\$[\d,]+\.?\d*|\d+\.?\d*%|\d{2,}', content)
        modified = content
        if numbers:
            target = numbers[0]
            if '$' in target:
                val = float(target.replace('$', '').replace(',', ''))
                new_val = val * 1.01  # 1% change
                new_str = f"${new_val:.2f}"
                if new_str == target:  # rounding produced same string
                    new_val = val + 0.01  # minimum 1 cent change
                    new_str = f"${new_val:.2f}"
                modified = content.replace(target, new_str, 1)
            elif '%' in target:
                val = float(target.replace('%', ''))
                new_val = val + 0.5
                modified = content.replace(target, f"{new_val}%", 1)
            else:
                val = int(target)
                new_val = val + 1
                modified = content.replace(target, str(new_val), 1)
        return (
            Document(doc.doc_id, doc.category, doc.title, modified),
            Modification(2, "Small numerical change (~1%)", True, True),
        )

    if level == 3:
        # Large numerical change
        numbers = re.findall(r'\$[\d,]+\.?\d*|\d+\.?\d*%|\d{2,}', content)
        modified = content
        if numbers:
            target = numbers[0]
            if '$' in target:
                val = float(target.replace('$', '').replace(',', ''))
                new_val = val * 3  # 3x change
                new_str = f"${new_val:.2f}"
                if new_str == target:
                    new_val = val + 1.00
                    new_str = f"${new_val:.2f}"
                modified = content.replace(target, new_str, 1)
            elif '%' in target:
                val = float(target.replace('%', ''))
                new_val = min(val * 2, 99.9)
                if new_val == val:  # at ceiling, halve instead
                    new_val = round(val / 2, 1)
                modified = content.replace(target, f"{new_val}%", 1)
            else:
                val = int(target)
                new_val = val * 3
                modified = content.replace(target, str(new_val), 1)
        return (
            Document(doc.doc_id, doc.category, doc.title, modified),
            Modification(3, "Large numerical change (3x)", True, True),
        )

    if level == 4:
        # Partial semantic rewrite: replace the last sentence with different info
        sentences = content.split('. ')
        if len(sentences) >= 2:
            sentences[-1] = "This feature was recently updated with significant performance improvements"
            modified = '. '.join(sentences)
        else:
            modified = content + " This feature was recently updated with significant performance improvements."
        return (
            Document(doc.doc_id, doc.category, doc.title, modified),
            Modification(4, "Partial semantic rewrite (last sentence)", True, True),
        )

    if level == 5:
        # Full content rewrite — completely different facts
        rewrites = {
            "product_spec": (
                "This product has been completely redesigned with a new architecture. "
                "All previous specifications are deprecated. The new system uses a "
                "microservices-based approach with Kubernetes orchestration, supporting "
                "up to 10,000 concurrent connections. Pricing has been restructured to "
                "a usage-based model starting at $0.001 per query."
            ),
            "faq": (
                "This FAQ has been replaced. The platform now uses a completely different "
                "workflow. Users should migrate to the new dashboard at dashboard.v2.example.com. "
                "All previous integrations are deprecated. New authentication uses passkeys "
                "instead of OAuth. Migration deadline is March 31, 2026."
            ),
            "pricing": (
                "All pricing has been restructured effective immediately. The platform "
                "now uses a flat-rate model at $299 per month for all tiers. Previous "
                "usage-based pricing is discontinued. Enterprise customers receive volume "
                "discounts starting at 20% for annual commitments."
            ),
            "policy": (
                "This policy has been completely revised to comply with the EU AI Act. "
                "All AI-generated content must now include disclosure labels. Data retention "
                "is limited to 90 days maximum. Third-party model providers must undergo "
                "annual security audits. Violation penalties have increased to $50,000 per incident."
            ),
        }
        modified = rewrites.get(doc.category, "Content has been completely replaced with new information.")
        return (
            Document(doc.doc_id, doc.category, doc.title, modified),
            Modification(5, "Full content rewrite", True, True),
        )

    raise ValueError(f"Invalid modification level: {level}")
