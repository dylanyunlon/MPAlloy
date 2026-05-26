#!/bin/bash
# Alloy Experiment 2: Mixed-Precision Training Convergence
# FP8 (H100) / BF16 (A6000) / FP32 (CPU) gradient consistency
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp2_mixed_precision_convergence/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Alloy Exp2: Mixed-Precision Convergence          ║"
echo "╚═══════════════════════════════════════════════════╝"

# ── All-FP32 baseline ──
echo "━━━ Baseline: All FP32 ━━━"
python3 "$ROOT_DIR/experiments/exp2_mixed_precision_convergence/run_convergence.py" \
    --mode baseline_fp32 \
    --num-iters 2000 \
    --verify-interval 50 \
    --output "$RESULTS_DIR/baseline_fp32.json" \
    2>&1 | tee "$RESULTS_DIR/baseline_fp32.log"

# ── Mixed precision: FP8 + BF16 + FP32 ──
echo "━━━ Mixed: FP8/BF16/FP32 ━━━"
python3 "$ROOT_DIR/experiments/exp2_mixed_precision_convergence/run_convergence.py" \
    --mode mixed \
    --num-iters 2000 \
    --verify-interval 50 \
    --output "$RESULTS_DIR/mixed_precision.json" \
    2>&1 | tee "$RESULTS_DIR/mixed_precision.log"

# ── Plot loss curves and drift ──
python3 "$ROOT_DIR/experiments/exp2_mixed_precision_convergence/plot_convergence.py" \
    --baseline "$RESULTS_DIR/baseline_fp32.json" \
    --mixed "$RESULTS_DIR/mixed_precision.json" \
    --output-dir "$RESULTS_DIR"

echo "[DONE] Results saved to $RESULTS_DIR/"
