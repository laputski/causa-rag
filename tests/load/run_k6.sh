#!/usr/bin/env bash
# Run k6 load test and save results to test-reports/k6/
set -e

TARGET="${TARGET:-http://localhost:8081}"
OUT_DIR="test-reports/k6"
mkdir -p "$OUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JSON_FILE="$OUT_DIR/results_${TIMESTAMP}.json"
SUMMARY_FILE="$OUT_DIR/summary_${TIMESTAMP}.json"

echo "Running k6 against $TARGET..."
echo "Output: $JSON_FILE"

k6 run \
  --env TARGET="$TARGET" \
  --out "json=$JSON_FILE" \
  --summary-export "$SUMMARY_FILE" \
  tests/load/k6_scenarios.js

echo ""
echo "Results saved:"
echo "  Raw metrics: $JSON_FILE"
echo "  Summary:     $SUMMARY_FILE"
