#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Self-Healing RAG — AWS Cleanup Guide
# ═══════════════════════════════════════════════════════════════════════
#
# This tears down ALL resources created by deploy.sh.
# Run this when you're done to avoid any AWS charges.
#
# Resources are deleted in reverse order of creation to respect
# dependencies (e.g., delete Lambda before its IAM role).
#
# Run each command one at a time so you can verify each step.
# ═══════════════════════════════════════════════════════════════════════

set -e

# ─── Variables (must match deploy.sh) ──────────────────────────────────

echo "═══ Setting up variables ═══"

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

BUCKET_NAME="self-healing-rag-${ACCOUNT_ID}"
TABLE_NAME="rag-healing-state"
SNS_TOPIC_NAME="rag-drift-alerts"
ROLE_NAME="self-healing-rag-lambda-role"
FUNCTION_NAME="self-healing-rag-monitor"
RULE_NAME="rag-monitor-schedule"
SNS_TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}"

echo "  Account: $ACCOUNT_ID"
echo "  Region: $REGION"
echo ""

# ─── STEP 1: Remove EventBridge Rule ──────────────────────────────────
# EventBridge rules must have their targets removed before deletion.

echo "═══ STEP 1: Removing EventBridge schedule ═══"
echo "  The schedule trigger must be removed first."

# Remove the target (Lambda) from the rule
aws events remove-targets \
    --rule "${RULE_NAME}" \
    --ids "self-healing-rag-target" \
    --region "${REGION}" \
    2>/dev/null || echo "  Target already removed (OK)"

# Delete the rule itself
aws events delete-rule \
    --name "${RULE_NAME}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Rule already deleted (OK)"

echo "  ✓ EventBridge rule deleted"
echo ""

# ─── STEP 2: Delete Lambda Function ───────────────────────────────────
# Delete the Lambda function. This also removes its CloudWatch log group
# permission, but the log group itself persists (see step below).

echo "═══ STEP 2: Deleting Lambda function ═══"

aws lambda delete-function \
    --function-name "${FUNCTION_NAME}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Function already deleted (OK)"

echo "  ✓ Lambda function deleted"
echo ""

# ─── STEP 3: Delete IAM Role ──────────────────────────────────────────
# IAM roles must have all policies removed before the role can be deleted.

echo "═══ STEP 3: Deleting IAM role ═══"
echo "  First remove the inline policy, then delete the role."

# Remove inline policy
aws iam delete-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name "self-healing-rag-permissions" \
    2>/dev/null || echo "  Policy already removed (OK)"

# Delete the role
aws iam delete-role \
    --role-name "${ROLE_NAME}" \
    2>/dev/null || echo "  Role already deleted (OK)"

echo "  ✓ IAM role deleted"
echo ""

# ─── STEP 4: Delete SNS Topic ─────────────────────────────────────────
# Deleting a topic also removes all subscriptions.

echo "═══ STEP 4: Deleting SNS topic ═══"

aws sns delete-topic \
    --topic-arn "${SNS_TOPIC_ARN}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Topic already deleted (OK)"

echo "  ✓ SNS topic deleted (subscriptions auto-removed)"
echo ""

# ─── STEP 5: Empty and Delete S3 Bucket ───────────────────────────────
# S3 buckets must be empty before deletion. aws s3 rb --force does both.

echo "═══ STEP 5: Deleting S3 bucket ═══"
echo "  Emptying bucket first (required before deletion)..."

aws s3 rb "s3://${BUCKET_NAME}" --force --region "${REGION}" \
    2>/dev/null || echo "  Bucket already deleted (OK)"

echo "  ✓ S3 bucket deleted"
echo ""

# ─── STEP 6: Delete DynamoDB Table ────────────────────────────────────
# Table deletion is immediate. All data is lost.

echo "═══ STEP 6: Deleting DynamoDB table ═══"

aws dynamodb delete-table \
    --table-name "${TABLE_NAME}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Table already deleted (OK)"

echo "  ✓ DynamoDB table deleted"
echo ""

# ─── STEP 7: Delete CloudWatch Log Group (optional) ───────────────────
# Lambda creates a log group automatically. It persists after Lambda deletion.
# Safe to leave (no charge for empty log groups) but let's be thorough.

echo "═══ STEP 7: Deleting CloudWatch log group ═══"

aws logs delete-log-group \
    --log-group-name "/aws/lambda/${FUNCTION_NAME}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Log group already deleted (OK)"

echo "  ✓ Log group deleted"
echo ""

# ─── Done ──────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════════════"
echo "  CLEANUP COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  All resources have been deleted:"
echo "    ✓ EventBridge rule: ${RULE_NAME}"
echo "    ✓ Lambda function: ${FUNCTION_NAME}"
echo "    ✓ IAM role: ${ROLE_NAME}"
echo "    ✓ SNS topic: ${SNS_TOPIC_NAME}"
echo "    ✓ S3 bucket: ${BUCKET_NAME}"
echo "    ✓ DynamoDB table: ${TABLE_NAME}"
echo "    ✓ CloudWatch log group"
echo ""
echo "  No recurring charges will be incurred."
echo ""
