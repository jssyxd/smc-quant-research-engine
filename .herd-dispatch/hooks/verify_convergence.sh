#!/bin/bash
set -e

echo "=== [HERDR CAPTAIN HOOK] Checking Agent Convergence Protocol ==="

OUTPUT_FILE="$1"
if [ -z "$OUTPUT_FILE" ]; then
    echo "[!] No deliverable output path provided. Skipping check."
    exit 0
fi

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "[!] Deliverable $OUTPUT_FILE not found."
    exit 1
fi

python3 scripts/validate_agent_convergence.py "$OUTPUT_FILE"

echo "=== [HERDR CAPTAIN HOOK] Deliverable Passed Convergence Gate ==="
exit 0
