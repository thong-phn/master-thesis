#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_wandb_sweep_four_stage.sh [agent_count]
# Example:
#   ./scripts/run_wandb_sweep_four_stage.sh 20
#
# Environment variables (optional):
#   SWEEP_CONFIG  : path to sweep yaml (default: scripts/wandb_sweep_four_stage.yaml)
#   WANDB_PROJECT : wandb project name override
#   WANDB_ENTITY  : wandb entity/team override

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AGENT_COUNT="${1:-20}"
SWEEP_CONFIG="${SWEEP_CONFIG:-scripts/wandb_sweep_four_stage.yaml}"

if ! command -v wandb >/dev/null 2>&1; then
  echo "Error: wandb CLI not found. Install with: pip install wandb"
  exit 1
fi

if [[ ! -f "$SWEEP_CONFIG" ]]; then
  echo "Error: sweep config not found at $SWEEP_CONFIG"
  exit 1
fi

echo "Creating sweep from: $SWEEP_CONFIG"
SWEEP_OUTPUT="$(wandb sweep "$SWEEP_CONFIG" 2>&1)"
echo "$SWEEP_OUTPUT"

SWEEP_REF="$(
  echo "$SWEEP_OUTPUT" \
    | sed -E 's/\x1b\[[0-9;]*[mK]//g' \
    | sed -nE 's/.*wandb agent[[:space:]]+([^[:space:]]+).*/\1/p' \
    | tail -n 1
)"

if [[ -z "$SWEEP_REF" ]]; then
  echo "Error: failed to parse sweep id from wandb output"
  exit 1
fi

echo "Starting wandb agent for sweep: $SWEEP_REF"
wandb agent --count "$AGENT_COUNT" "$SWEEP_REF"
