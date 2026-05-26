# Alloy: Mixed-Precision Elastic Embedding Training on Asymmetric Multi-Generation GPU Clusters

> **Alloy** (合金): Different metals (multi-generation GPUs) fused into a unified system with emergent properties beyond any individual component.

## Overview

Alloy is a distributed embedding training system designed for **asymmetric, multi-generation GPU clusters** (e.g., A6000 × 2 + H100 × 1 + CPU). It achieves training efficiency that surpasses homogeneous clusters of equivalent compute by exploiting the unique characteristics of each hardware tier.

### Key Contributions

1. **Four-Level Tiered Embedding Placement**: Extends two-level (GPU/CPU) hierarchical caching to four tiers:
   - **H100 HBM3** → Hot embeddings (FP8)
   - **A6000 GDDR6** → Warm embeddings (BF16)
   - **CPU DRAM** → Cold embeddings (FP32)
   - **SSD** → Archived embeddings

2. **Elastic Tensor Memory Scheduling**: Adapts mTuner's gather/discard/execute/checkpoint operations as GPU pool eviction policies for dynamic embedding migration across heterogeneous memory hierarchies.

3. **Cross-Precision Gradient Verification**: Uses FPRev's distinguishing input methodology to construct periodic diagnostic batches, verifying numerical drift in cross-precision gradient aggregation (FP8 ↔ BF16 ↔ FP32).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Alloy Runtime                        │
├──────────┬──────────────────┬───────────────────────────┤
│ HypeReca │  mTuner Elastic  │  FPRev Precision         │
│ Backbone │  Tensor Engine   │  Verification Oracle     │
├──────────┴──────────────────┴───────────────────────────┤
│  H100 (HBM3/FP8)  │  A6000×2 (GDDR6/BF16)  │  CPU/SSD │
└────────────────────┴────────────────────────┴──────────┘
```

## Repository Structure

```
MPAlloy/
├── core/
│   ├── hypereca/          # Distributed embedding system backbone (HypeReca)
│   ├── mtuner/            # Elastic tensor memory scheduling engine (mTuner)
│   └── fprev/             # Floating-point precision verification (FPRev)
├── alloy/
│   ├── scheduler/         # Four-level tiered placement scheduler
│   ├── mixed_precision/   # Cross-precision gradient aggregation
│   ├── elastic_tensor/    # Dynamic embedding migration manager
│   └── benchmarks/        # Micro-benchmarks
├── experiments/
│   ├── exp1_tiered_placement/           # Tiered vs homogeneous baseline
│   ├── exp2_mixed_precision_convergence/# FP8/BF16/FP32 convergence
│   └── exp3_elastic_migration/          # Migration overhead PCIe Gen4 vs Gen5
├── scripts/               # Launch & utility scripts
└── docs/                  # Documentation
```

## Hardware Requirements

- **Minimum**: 1× NVIDIA H100 + 2× NVIDIA A6000 + CPU with ≥128GB DRAM
- **Interconnect**: PCIe Gen4 (A6000) + PCIe Gen5 (H100)
- **Storage**: NVMe SSD for tier-4 archival

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Experiment 1: Tiered Placement vs Homogeneous Baseline
bash scripts/run_exp1.sh

# Run Experiment 2: Mixed-Precision Convergence
bash scripts/run_exp2.sh

# Run Experiment 3: Elastic Migration Overhead
bash scripts/run_exp3.sh
```

## Experiments

### Exp1: Tiered Placement vs Homogeneous Baseline
DLRM embedding training on A6000×2 + H100×1. Compares uniform distribution against frequency-based placement across H100/A6000/CPU tiers. Measures throughput and memory utilization.

### Exp2: Mixed-Precision Training Convergence
H100 embeddings updated in FP8, A6000 in BF16, CPU in FP32. FPRev verifies gradient consistency across the three precision paths after allreduce. Produces loss curve comparison.

### Exp3: Elastic Tensor Dynamic Migration Overhead
Online embedding GPU reassignment (simulating hotness changes). Measures mTuner gather/discard latency on PCIe Gen4 (A6000) vs PCIe Gen5 (H100).

## Related Work

- **Crucible** ([CASH](https://github.com/dylanyunlon/CASH)): Cross-architecture kernel correctness testing companion — verifies that Alloy's kernels behave correctly across sm86 (A6000) and sm90 (H100).

## Target Venue

OSDI / ATC / EuroSys

## License

See individual component licenses in `core/*/LICENSE`.
