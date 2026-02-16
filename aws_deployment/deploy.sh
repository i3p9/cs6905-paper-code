#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Self-Healing RAG — AWS Deployment Guide
# ═══════════════════════════════════════════════════════════════════════
#
# This is a STEP-BY-STEP guide. Run each command one at a time so you
# can see the output and understand what each piece does.
#
# Architecture:
#   EventBridge (every 6 hours)
#     → Lambda: self-healing-rag-monitor
#       ├── Reads docs from S3
#       ├── SHA-256 hash vs DynamoDB stored hash
#       ├── If changed: Bedrock Titan Embeddings → cosine distance
#       ├── Updates DynamoDB (new hash, embeddings, drift score)
#       ├── Publishes CloudWatch custom metrics
#       └── Sends SNS alert if drift detected
#
# Prerequisites:
#   1. AWS CLI configured (run: aws sts get-caller-identity)
#   2. Bedrock Titan Embeddings v2 — auto-enabled on first invoke in all
#      commercial regions (no manual model access activation needed).
#      Verify with: aws bedrock-runtime invoke-model --model-id amazon.titan-embed-text-v2:0 \
#        --content-type application/json --accept application/json \
#        --body '{"inputText":"test","dimensions":1024,"normalize":true}' \
#        --region us-east-1 /tmp/bedrock-test.json
#   3. Python 3 + boto3 installed locally
#
# Estimated cost: < $1/month (all free tier except ~$0.05 Bedrock)
# ═══════════════════════════════════════════════════════════════════════

set -e  # Stop on any error so you can see what went wrong

# ─── STEP 1: Set Up Variables ──────────────────────────────────────────
# First, let's grab your AWS account ID and set up names for everything.
# All resource names are prefixed with "self-healing-rag" for easy cleanup.

echo "═══ STEP 1: Setting up variables ═══"

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
echo "  Account ID: $ACCOUNT_ID"
echo "  Region: $REGION"

# Resource names — these will be used throughout
BUCKET_NAME="self-healing-rag-${ACCOUNT_ID}"
TABLE_NAME="rag-healing-state"
SNS_TOPIC_NAME="rag-drift-alerts"
ROLE_NAME="self-healing-rag-lambda-role"
FUNCTION_NAME="self-healing-rag-monitor"
RULE_NAME="rag-monitor-schedule"

echo "  S3 Bucket: $BUCKET_NAME"
echo "  DynamoDB Table: $TABLE_NAME"
echo "  SNS Topic: $SNS_TOPIC_NAME"
echo "  Lambda Function: $FUNCTION_NAME"
echo ""

# ─── STEP 2: Create S3 Bucket ─────────────────────────────────────────
# S3 stores our 20 source documents (the "knowledge base").
# The Lambda function reads documents from here to check for changes.
# Bucket names must be globally unique, so we append the account ID.

echo "═══ STEP 2: Creating S3 bucket ═══"
echo "  This bucket stores the source documents that our RAG system monitors."

aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}" 2>/dev/null || echo "  Bucket already exists (OK)"

echo "  ✓ Bucket: ${BUCKET_NAME}"
echo ""

# ─── STEP 3: Create DynamoDB Table ────────────────────────────────────
# DynamoDB is the "Knowledge" component of our MAPE-K loop.
# It stores: document hashes, chunk embeddings, drift scores, timestamps.
# PAY_PER_REQUEST = on-demand pricing (free tier: 25 WCU + 25 RCU).
# For 20 documents, we'll never exceed free tier.

echo "═══ STEP 3: Creating DynamoDB table ═══"
echo "  This table stores the MAPE-K knowledge base:"
echo "    - doc_id (primary key)"
echo "    - content_hash (SHA-256 for change detection)"
echo "    - chunk_embeddings (Titan Embeddings vectors)"
echo "    - drift_score, drift_type, heal_count, timestamps"

aws dynamodb create-table \
    --table-name "${TABLE_NAME}" \
    --attribute-definitions AttributeName=doc_id,AttributeType=S \
    --key-schema AttributeName=doc_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${REGION}" \
    2>/dev/null || echo "  Table already exists (OK)"

# Wait for table to be active (usually takes a few seconds)
echo "  Waiting for table to become active..."
aws dynamodb wait table-exists --table-name "${TABLE_NAME}" --region "${REGION}"
echo "  ✓ Table: ${TABLE_NAME}"
echo ""

# ─── STEP 4: Create SNS Topic ─────────────────────────────────────────
# SNS sends email/SMS alerts when the system detects knowledge drift.
# This is the "alert" output of our autonomic healing loop.
# After creating the topic, subscribe your email to receive alerts.

echo "═══ STEP 4: Creating SNS topic ═══"
echo "  SNS sends drift alerts when the monitor detects changes."

SNS_TOPIC_ARN=$(aws sns create-topic \
    --name "${SNS_TOPIC_NAME}" \
    --region "${REGION}" \
    --query "TopicArn" --output text)

echo "  ✓ Topic ARN: ${SNS_TOPIC_ARN}"
echo ""

# Subscribe your email (change this to your email address!)
echo "  To receive email alerts, run this command with YOUR email:"
echo "  aws sns subscribe --topic-arn ${SNS_TOPIC_ARN} --protocol email --notification-endpoint YOUR_EMAIL@example.com --region ${REGION}"
echo ""
echo "  Then check your email and click the confirmation link."
echo ""

# ─── STEP 5: Create IAM Role for Lambda ───────────────────────────────
# Lambda needs permissions to access S3, DynamoDB, Bedrock, CloudWatch, SNS.
# We create a role with a "trust policy" (who can assume it: Lambda)
# and an "inline policy" (what it can do: read S3, write DynamoDB, etc.)

echo "═══ STEP 5: Creating IAM role ═══"
echo "  Lambda needs an IAM role with permissions for:"
echo "    - S3: Read documents"
echo "    - DynamoDB: Read/write healing state"
echo "    - Bedrock: Invoke Titan Embeddings"
echo "    - CloudWatch: Publish custom metrics"
echo "    - SNS: Publish drift alerts"
echo "    - CloudWatch Logs: Write Lambda logs"

# 5a. Trust policy — allows Lambda service to assume this role
cat > /tmp/trust-policy.json << 'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
TRUST

aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file:///tmp/trust-policy.json \
    2>/dev/null || echo "  Role already exists (OK)"

# 5b. Inline policy — defines exactly what Lambda can access
# This follows least-privilege: only the specific resources we created
cat > /tmp/lambda-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadDocuments",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "DynamoDBReadWrite",
      "Effect": "Allow",
      "Action": [
        "dynamodb:Scan",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem"
      ],
      "Resource": "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${TABLE_NAME}"
    },
    {
      "Sid": "BedrockInvokeModel",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:${REGION}::foundation-model/amazon.titan-embed-text-v2:0"
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*"
    },
    {
      "Sid": "SNSPublish",
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "${SNS_TOPIC_ARN}"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    }
  ]
}
POLICY

aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name "self-healing-rag-permissions" \
    --policy-document file:///tmp/lambda-policy.json

echo "  ✓ Role: ${ROLE_NAME}"
echo ""

# IAM role propagation can take a few seconds — wait before creating Lambda
echo "  Waiting 10 seconds for IAM role to propagate..."
sleep 10

# ─── STEP 6: Package Lambda Function ──────────────────────────────────
# Lambda code must be uploaded as a ZIP file. Our function is a single
# Python file with no external dependencies (pure Python, no numpy).
# This is why we wrote cosine_distance() by hand instead of using numpy.

echo "═══ STEP 6: Packaging Lambda function ═══"
echo "  Zipping lambda_function.py (no dependencies needed — pure Python)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

zip -j /tmp/lambda_function.zip lambda_function.py

echo "  ✓ Package: /tmp/lambda_function.zip ($(du -h /tmp/lambda_function.zip | cut -f1))"
echo ""

# ─── STEP 7: Create Lambda Function ───────────────────────────────────
# The Lambda function is the core of our MAPE-K loop. It:
#   - Runs on Python 3.12 runtime
#   - Has 512MB memory (enough for JSON processing + API calls)
#   - 120s timeout (20 docs × ~5s each for Bedrock calls)
#   - Environment variables configure the S3 bucket, DynamoDB table, etc.

echo "═══ STEP 7: Creating Lambda function ═══"
echo "  Runtime: Python 3.12"
echo "  Memory: 512MB"
echo "  Timeout: 120 seconds"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

aws lambda create-function \
    --function-name "${FUNCTION_NAME}" \
    --runtime python3.12 \
    --role "${ROLE_ARN}" \
    --handler lambda_function.lambda_handler \
    --zip-file fileb:///tmp/lambda_function.zip \
    --timeout 120 \
    --memory-size 512 \
    --environment "Variables={S3_BUCKET=${BUCKET_NAME},DYNAMODB_TABLE=${TABLE_NAME},SNS_TOPIC_ARN=${SNS_TOPIC_ARN},DRIFT_THRESHOLD=0.08}" \
    --region "${REGION}" \
    2>/dev/null || {
        echo "  Function exists, updating code..."
        aws lambda update-function-code \
            --function-name "${FUNCTION_NAME}" \
            --zip-file fileb:///tmp/lambda_function.zip \
            --region "${REGION}"
    }

echo "  ✓ Lambda: ${FUNCTION_NAME}"
echo ""

# ─── STEP 8: Create EventBridge Schedule ──────────────────────────────
# EventBridge triggers the Lambda every 6 hours. This is the "Monitor"
# part of MAPE-K — continuous, automated checking.
# rate(6 hours) = 4 times per day. Well within free tier (1M events/month).

echo "═══ STEP 8: Creating EventBridge schedule ═══"
echo "  Schedule: Every 6 hours (rate(6 hours))"
echo "  This is the 'Monitor' trigger in our MAPE-K loop."

RULE_ARN=$(aws events put-rule \
    --name "${RULE_NAME}" \
    --schedule-expression "rate(6 hours)" \
    --state ENABLED \
    --description "Triggers self-healing RAG monitor every 6 hours" \
    --region "${REGION}" \
    --query "RuleArn" --output text)

echo "  ✓ Rule: ${RULE_NAME}"

# Allow EventBridge to invoke our Lambda
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"

aws lambda add-permission \
    --function-name "${FUNCTION_NAME}" \
    --statement-id "eventbridge-invoke" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "${RULE_ARN}" \
    --region "${REGION}" \
    2>/dev/null || echo "  Permission already exists (OK)"

# Point the EventBridge rule at our Lambda
aws events put-targets \
    --rule "${RULE_NAME}" \
    --targets "Id=self-healing-rag-target,Arn=${LAMBDA_ARN}" \
    --region "${REGION}"

echo "  ✓ EventBridge → Lambda connection established"
echo ""

# ─── STEP 9: Verify Deployment ────────────────────────────────────────
echo "═══ STEP 9: Verifying deployment ═══"

echo "  Checking S3 bucket..."
aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null && echo "    ✓ Bucket accessible" || echo "    ✓ Bucket exists (empty)"

echo "  Checking DynamoDB table..."
aws dynamodb describe-table --table-name "${TABLE_NAME}" --region "${REGION}" \
    --query "Table.TableStatus" --output text

echo "  Checking Lambda function..."
aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${REGION}" \
    --query "Configuration.{Runtime:Runtime,Memory:MemorySize,Timeout:Timeout}" --output table

echo "  Checking EventBridge rule..."
aws events describe-rule --name "${RULE_NAME}" --region "${REGION}" \
    --query "{State:State,Schedule:ScheduleExpression}" --output table

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "    1. Verify Bedrock Titan Embeddings (auto-enabled, no manual activation needed):"
echo "       aws bedrock-runtime invoke-model --model-id amazon.titan-embed-text-v2:0 \\"
echo "         --content-type application/json --accept application/json \\"
echo "         --body '{\"inputText\":\"test\",\"dimensions\":1024,\"normalize\":true}' \\"
echo "         --region ${REGION} /tmp/bedrock-test.json && echo 'Bedrock OK'"
echo ""
echo "    2. Subscribe your email to SNS alerts:"
echo "       aws sns subscribe --topic-arn ${SNS_TOPIC_ARN} --protocol email --notification-endpoint YOUR_EMAIL --region ${REGION}"
echo ""
echo "    3. Seed the database with test documents:"
echo "       python3 seed_data.py"
echo ""
echo "    4. Test drift detection:"
echo "       python3 test_drift.py"
echo ""
echo "  Resource summary:"
echo "    S3 Bucket:    ${BUCKET_NAME}"
echo "    DynamoDB:     ${TABLE_NAME}"
echo "    Lambda:       ${FUNCTION_NAME}"
echo "    SNS Topic:    ${SNS_TOPIC_ARN}"
echo "    EventBridge:  ${RULE_NAME} (every 6 hours)"
echo ""
echo "  To clean up (avoid charges):"
echo "    bash cleanup.sh"
echo ""
