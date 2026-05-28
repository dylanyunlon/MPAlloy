#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Alloy Heterogeneous Cluster Test — Launch Script
#  Target: ags1 (A6000×2 + H100 NVL + EPYC 9354×2)
#
#  Usage:
#    bash scripts/run_hetero_test.sh              # Full run
#    bash scripts/run_hetero_test.sh --quick      # Smoke test
#    bash scripts/run_hetero_test.sh --exp exp3   # Single experiment
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${REPO_ROOT}/results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "═══════════════════════════════════════════════════"
echo "  Alloy Heterogeneous Integration Test"
echo "  Repo:    ${REPO_ROOT}"
echo "  Results: ${RESULTS_DIR}"
echo "  Time:    $(date)"
echo "═══════════════════════════════════════════════════"

# ── 1. Hardware probe ──
echo ""
echo ">>> Hardware Probe..."
echo "CPU: $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
echo "Sockets: $(lscpu | grep 'Socket(s)' | awk '{print $NF}')"
echo "Total cores: $(nproc)"

if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo ">>> GPU Topology:"
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current \
        --format=csv,noheader 2>/dev/null || true
    echo ""
    echo ">>> PCIe topology:"
    nvidia-smi topo -m 2>/dev/null | head -15 || true
fi

echo ""
echo ">>> Memory:"
free -g 2>/dev/null | head -2 || true

echo ""
echo ">>> NUMA:"
numactl --hardware 2>/dev/null | grep -E "node [0-9]+ (size|free)" || true

# ── 2. Environment ──
echo ""
echo ">>> Environment setup..."

# All GPUs are on NUMA node 1 — pin CPU affinity there
NUMA_NODE=1
NUMA_CPUS="32-63,96-127"

# Check CUDA version
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "PyTorch CUDA: ${CUDA_VER}"

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "PyTorch version: ${TORCH_VER}"

# Warn about PCIe Gen1 (should be Gen4)
PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader 2>/dev/null | head -1 || echo "?")
if [ "$PCIE_GEN" = "1" ]; then
    echo ""
    echo "⚠⚠⚠  WARNING: A6000 reports PCIe Gen1 — should be Gen4!  ⚠⚠⚠"
    echo "     This will severely bottleneck migration bandwidth."
    echo "     Check: BIOS settings, riser cables, slot population."
    echo "     Expected Gen4 x16: ~25 GB/s; Gen1 x16: ~4 GB/s"
    echo ""
fi

# ── 3. Ensure dependencies ──
echo ">>> Checking dependencies..."
python3 -c "import torch; import numpy; print('OK')" || {
    echo "ERROR: Missing torch or numpy. Install with:"
    echo "  pip install torch numpy"
    exit 1
}

# ── 4. Save hardware info ──
python3 -c "
import torch, json, os
hw = {
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpus': [],
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    c = torch.cuda.get_device_capability(i)
    hw['gpus'].append({
        'index': i,
        'name': torch.cuda.get_device_name(i),
        'mem_gb': round(p.total_mem / 1e9, 1),
        'sm': f'sm{c[0]}{c[1]}',
    })
with open('${RESULTS_DIR}/hardware.json', 'w') as f:
    json.dump(hw, f, indent=2)
print(f'Hardware info saved to ${RESULTS_DIR}/hardware.json')
"

# ── 5. Run experiment ──
echo ""
echo ">>> Launching experiment..."
echo ""

# Parse extra args
EXTRA_ARGS="$@"

# NUMA-pinned execution
# All GPUs on NUMA node 1, so pin CPU there too for best PCIe affinity
if command -v numactl &>/dev/null; then
    NUMA_PREFIX="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    echo "NUMA pinning: node ${NUMA_NODE} (CPUs ${NUMA_CPUS})"
else
    NUMA_PREFIX=""
    echo "⚠ numactl not found — running without NUMA pinning"
fi

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

${NUMA_PREFIX} python3 "${REPO_ROOT}/experiments/run_hetero_integration.py" \
    --output "${RESULTS_DIR}/results.json" \
    --seed 42 \
    ${EXTRA_ARGS}

EXIT_CODE=$?

# ── 6. Summary ──
echo ""
echo "═══════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ Experiment completed successfully"
else
    echo "  ✗ Experiment failed (exit code ${EXIT_CODE})"
fi
echo "  Results: ${RESULTS_DIR}/results.json"
echo "  Time:    $(date)"
echo "═══════════════════════════════════════════════════"

# Quick summary from JSON
python3 -c "
import json, sys
try:
    with open('${RESULTS_DIR}/results.json') as f:
        r = json.load(f)
    exps = r.get('experiments', {})

    if 'regression' in exps:
        reg = exps['regression']
        passed = sum(1 for t in reg['tests'].values() if t.get('passed'))
        total = len(reg['tests'])
        print(f'  Regression: {passed}/{total} passed')

    if 'exp1_tiered_placement' in exps:
        e1 = exps['exp1_tiered_placement']
        su = e1.get('speedup', 'N/A')
        print(f'  Exp1 Tiered speedup: {su}x')

    if 'exp2_convergence' in exps:
        e2 = exps['exp2_convergence']
        bl = e2.get('baseline_fp32', {}).get('final_loss', '?')
        mx = e2.get('mixed_precision', {}).get('final_loss', '?')
        print(f'  Exp2 Loss: baseline={bl}, mixed={mx}')

    if 'exp3_migration' in exps:
        e3 = exps['exp3_migration']
        xfers = e3.get('transfers', {}).get('64MB', {})
        for k, v in list(xfers.items())[:4]:
            if 'mean_bw_gbps' in v:
                print(f'  Exp3 {k}: {v[\"mean_bw_gbps\"]} GB/s')
except Exception as e:
    print(f'  (could not parse results: {e})')
" 2>/dev/null || true

exit $EXIT_CODE
