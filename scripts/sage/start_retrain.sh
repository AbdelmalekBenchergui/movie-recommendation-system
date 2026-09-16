#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
BUCKET="${ML_BUCKET:-recsys-dev-bucket}"
STATE_MACHINE="${STATE_MACHINE_NAME:-recsys-dev-retrain}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID not set (see .env)}"
EPOCHS="${EPOCHS:-3}"
INSTANCE_TYPE="${INSTANCE_TYPE:-${SAGEMAKER_INSTANCE_TYPE:-ml.m5.large}}"
RUN_ID="${RUN_ID:-recsys-$(date +%Y%m%d-%H%M%S)}"

CODE_PREFIX="s3://${BUCKET}/${CODE_S3_PREFIX:-ml/retrain-code}"

echo "==> uploading code bundle to ${CODE_PREFIX}"
aws s3 cp scripts/config.py                 "${CODE_PREFIX}/scripts/config.py" --region "${REGION}"
aws s3 cp scripts/publish_ml.py            "${CODE_PREFIX}/scripts/publish_ml.py" --region "${REGION}"
aws s3 cp scripts/sage/two_tower_train.py  "${CODE_PREFIX}/scripts/sage/two_tower_train.py" --region "${REGION}"
aws s3 cp scripts/sage/processing_publish.py "${CODE_PREFIX}/scripts/sage/processing_publish.py" --region "${REGION}"

echo "==> bundling trainer entry point for the framework container"
TARBALL="$(mktemp)"
tar -czf "${TARBALL}" -C scripts/sage two_tower_train.py
aws s3 cp "${TARBALL}" "${CODE_PREFIX}/two_tower_train.tar.gz" --region "${REGION}"
rm -f "${TARBALL}"

INPUT_JSON=$(jq -n \
  --arg run "${RUN_ID}" \
  --arg epochs "${EPOCHS}" \
  --arg instance_type "${INSTANCE_TYPE}" \
  '{run: $run, epochs: ($epochs | tonumber), instance_type: $instance_type}')

echo "==> starting execution of ${STATE_MACHINE}"
EXEC_ARN=$(aws stepfunctions start-execution \
  --state-machine-arn "arn:aws:states:${REGION}:${ACCOUNT_ID}:stateMachine:${STATE_MACHINE}" \
  --name "${RUN_ID}" \
  --input "${INPUT_JSON}" \
  --region "${REGION}" \
  --query 'executionArn' --output text)

echo "Execution: ${EXEC_ARN}"
echo "Monitor:  aws stepfunctions get-execution-history --execution-arn '${EXEC_ARN}' --region ${REGION}"