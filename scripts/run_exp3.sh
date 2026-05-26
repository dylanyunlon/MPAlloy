#!/bin/bash
# Alloy Experiment 3: Elastic Tensor Dynamic Migration Overhead
# Measures gather/discard latency on PCIe Gen4 (A6000) vs Gen5 (H100)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/experiments/exp3_elastic_migration/results"
mkdir -p "$RESULTS_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  Alloy Exp3: Elastic Migration Overhead           ║"
echo "╚═══════════════════════════════════════════════════╝"

# Sweep tensor sizes to measure migration cost
for SIZE_MB in 1 4 16 64 256 1024; do
    echo "━━━ Tensor size: ${SIZE_MB} MB ━━━"
    python3 "$ROOT_DIR/experiments/exp3_elastic_migration/run_migration_bench.py" \
        --tensor-size-mb $SIZE_MB \
        --num-trials 100 \
        --output "$RESULTS_DIR/migration_${SIZE_MB}MB.json" \
        2>&1 | tee "$RESULTS_DIR/migration_${SIZE_MB}MB.log"
done

# ── Aggregate and plot ──
python3 "$ROOT_DIR/experiments/exp3_elastic_migration/plot_migration.py" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR"

echo "[DONE] Results saved to $RESULTS_DIR/"
