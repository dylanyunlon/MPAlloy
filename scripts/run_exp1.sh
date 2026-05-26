#!/bin/bash
# ═══════════════════════════════════════════════════════
# Alloy Experiment 1: Tiered Placement vs Homogeneous Baseline
# Hardware: A6000 × 2 + H100 × 1 + CPU
# ═══════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp1_tiered_placement/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Alloy Exp1: Tiered Placement vs Homogeneous     ║"
echo "╚═══════════════════════════════════════════════════╝"

# ── Detect GPUs ──
echo "[INFO] Detecting GPU configuration..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[INFO] Found $NUM_GPUS GPUs"

# ── Configuration ──
EMBEDDING_DIM=128
NUM_TABLES=26            # DLRM default
NUM_EMBEDDINGS=10000000  # 10M per table
BATCH_SIZE=65536
NUM_ITERS=500
WARMUP_ITERS=50

# ── Baseline: Uniform distribution (all embeddings split evenly across GPUs) ──
echo ""
echo "━━━ Running Baseline: Uniform Distribution ━━━"
python3 "$ROOT_DIR/experiments/exp1_tiered_placement/run_baseline.py" \
    --embedding-dim $EMBEDDING_DIM \
    --num-tables $NUM_TABLES \
    --num-embeddings $NUM_EMBEDDINGS \
    --batch-size $BATCH_SIZE \
    --num-iters $NUM_ITERS \
    --warmup-iters $WARMUP_ITERS \
    --distribution uniform \
    --output "$RESULTS_DIR/baseline_uniform.json" \
    2>&1 | tee "$RESULTS_DIR/baseline_uniform.log"

# ── Alloy: Frequency-based tiered placement ──
echo ""
echo "━━━ Running Alloy: Tiered Placement ━━━"
python3 "$ROOT_DIR/experiments/exp1_tiered_placement/run_baseline.py" \
    --embedding-dim $EMBEDDING_DIM \
    --num-tables $NUM_TABLES \
    --num-embeddings $NUM_EMBEDDINGS \
    --batch-size $BATCH_SIZE \
    --num-iters $NUM_ITERS \
    --warmup-iters $WARMUP_ITERS \
    --distribution tiered \
    --hot-ratio 0.1 \
    --warm-ratio 0.3 \
    --output "$RESULTS_DIR/alloy_tiered.json" \
    2>&1 | tee "$RESULTS_DIR/alloy_tiered.log"

# ── Generate comparison plots ──
echo ""
echo "━━━ Generating Comparison Plots ━━━"
python3 "$ROOT_DIR/experiments/exp1_tiered_placement/plot_results.py" \
    --baseline "$RESULTS_DIR/baseline_uniform.json" \
    --tiered "$RESULTS_DIR/alloy_tiered.json" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "[DONE] Results saved to $RESULTS_DIR/"
