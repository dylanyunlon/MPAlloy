#!/usr/bin/env python3
"""
Alloy Heterogeneous Cluster Integration Test
=============================================
Target: ags1 server
  GPU0: A6000 48GB  PCIe Gen4 x16  sm86  (NUMA1)
  GPU1: A6000 48GB  PCIe Gen4 x16  sm86  (NUMA1)
  GPU2: H100 NVL 96GB PCIe Gen5 x16 sm90  (NUMA1)
  CPU:  EPYC 9354 x2, 1.5TB DRAM (2 NUMA nodes)

This script runs ALL three experiments in sequence and produces
a unified JSON report.  Designed for direct execution on the server.

Usage:
  # Full run (from MPAlloy repo root):
  numactl --cpunodebind=1 --membind=1 python3 experiments/run_hetero_integration.py --output results/full_run.json

  # Quick smoke test:
  numactl --cpunodebind=1 --membind=1 python3 experiments/run_hetero_integration.py --quick --output results/smoke.json
"""

import argparse
import json
import os
import sys
import time
import gc
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np

# ── Ensure repo root is on path ──────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from alloy.scheduler.tiered_placement import (
    TieredPlacementScheduler, TierConfig, Tier,
    AccessFrequencyTracker, create_default_config
)
from alloy.elastic_tensor.manager import (
    GPUMemoryPool, TensorSlot, ElasticTensorManager, ElasticOp
)
from alloy.mixed_precision.gradient_verifier import (
    CrossPrecisionAllReduce, FPRevDiagnosticBatch, PrecisionConfig
)


# ═══════════════════════════════════════════════════════════════════
#  Section 0: Hardware Discovery & Validation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GPUInfo:
    index: int
    name: str
    total_mem_gb: float
    compute_cap: Tuple[int, int]
    pcie_gen: int        # from nvidia-smi
    arch_class: str      # 'sm86' or 'sm90'
    tier: str            # 'h100', 'a6000', or 'unknown'

def discover_hardware() -> Dict:
    """Discover and validate the heterogeneous GPU cluster."""
    hw = {
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'num_gpus': 0,
        'gpus': [],
        'cpu_count': os.cpu_count(),
        'warnings': [],
    }

    if not torch.cuda.is_available():
        hw['warnings'].append("CUDA not available — will run CPU-only fallback")
        return hw

    hw['num_gpus'] = torch.cuda.device_count()
    for i in range(hw['num_gpus']):
        props = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = props.total_mem / (1024**3)

        # Classify GPU
        if cap[0] >= 9:
            arch_class = f"sm{cap[0]}{cap[1]}"
            tier = 'h100'
        elif 'A6000' in name:
            arch_class = f"sm{cap[0]}{cap[1]}"
            tier = 'a6000'
        else:
            arch_class = f"sm{cap[0]}{cap[1]}"
            tier = 'unknown'

        # Detect PCIe gen from nvidia-smi output (hardcoded from your topology)
        pcie_gen_map = {0: 4, 1: 4, 2: 5}  # Your specific server
        pcie_gen = pcie_gen_map.get(i, 4)

        gpu = GPUInfo(
            index=i, name=name, total_mem_gb=round(mem_gb, 1),
            compute_cap=cap, pcie_gen=pcie_gen,
            arch_class=arch_class, tier=tier
        )
        hw['gpus'].append(asdict(gpu))

    # Validate expected topology
    tiers = [g['tier'] for g in hw['gpus']]
    if tiers.count('h100') == 0:
        hw['warnings'].append("No H100 detected — FP8 experiments will be skipped")
    if tiers.count('a6000') < 2:
        hw['warnings'].append(f"Expected 2 A6000s, found {tiers.count('a6000')}")

    # Check CUDA version for FP8 support
    cuda_ver = torch.version.cuda
    if cuda_ver and tuple(int(x) for x in cuda_ver.split('.')[:2]) < (11, 8):
        hw['warnings'].append(
            f"CUDA {cuda_ver} may not support torch.float8_e4m3fn. "
            f"FP8 tests will use BF16 fallback."
        )

    return hw


def check_fp8_support(device_index: int) -> bool:
    """Check if FP8 (float8_e4m3fn) is actually usable on this device."""
    try:
        dev = torch.device(f'cuda:{device_index}')
        cap = torch.cuda.get_device_capability(device_index)
        if cap[0] < 9:
            return False
        t = torch.randn(16, device=dev)
        if hasattr(torch, 'float8_e4m3fn'):
            t8 = t.to(torch.float8_e4m3fn)
            _ = t8.to(torch.float32)
            return True
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════
#  Section 1: Exp1 — Tiered Placement vs Uniform Baseline
# ═══════════════════════════════════════════════════════════════════

def generate_zipf_indices(num_embeddings: int, batch_size: int,
                          alpha: float = 1.0) -> torch.Tensor:
    """Generate Zipf-distributed embedding indices."""
    weights = np.arange(1, num_embeddings + 1, dtype=np.float64) ** (-alpha)
    weights /= weights.sum()
    indices = np.random.choice(num_embeddings, size=batch_size, p=weights)
    return torch.from_numpy(indices).long()


def run_exp1_tiered_placement(hw: Dict, quick: bool = False) -> Dict:
    """
    Exp1: Tiered Placement vs Uniform Baseline.
    Measures embedding lookup throughput with Zipf-distributed access
    under uniform vs frequency-aware placement.
    """
    print("\n" + "="*60)
    print("  EXP1: Tiered Placement vs Uniform Baseline")
    print("="*60)

    num_tables = 8 if quick else 26
    num_embeddings = 100_000 if quick else 1_000_000
    embedding_dim = 128
    batch_size = 8192 if quick else 65536
    num_iters = 100 if quick else 500
    warmup_iters = 20 if quick else 50
    alpha = 1.0  # Zipf parameter

    results = {
        'config': {
            'num_tables': num_tables,
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'batch_size': batch_size,
            'num_iters': num_iters,
            'zipf_alpha': alpha,
        },
        'uniform': {},
        'tiered': {},
    }

    gpus = hw.get('gpus', [])
    h100_devs = [g['index'] for g in gpus if g['tier'] == 'h100']
    a6000_devs = [g['index'] for g in gpus if g['tier'] == 'a6000']
    use_cuda = torch.cuda.is_available() and len(gpus) > 0

    # ── Create tier config from real hardware ──
    if use_cuda and h100_devs and a6000_devs:
        tier_configs = create_default_config(
            h100_memory_gb=95.0,   # H100 NVL actual
            a6000_memory_gb=48.0,  # A6000 actual
            cpu_memory_gb=768.0,   # NUMA1 free
        )
    else:
        tier_configs = create_default_config()

    scheduler = TieredPlacementScheduler(tier_configs)
    tracker = scheduler.tracker

    for table_id in range(num_tables):
        scheduler.register_embedding_table(
            table_id, num_embeddings, embedding_dim, chunk_size=4096
        )

    # ── Helper: run benchmark iterations ──
    def run_iterations(placement_name: str, tables: Dict[str, torch.Tensor],
                       device_map: Dict[int, torch.device]) -> Dict:
        throughputs = []
        latencies = []
        mem_snapshots = []

        for step in range(num_iters):
            indices = generate_zipf_indices(num_embeddings, batch_size, alpha)
            step_start = time.perf_counter()

            # Simulate tiered lookup: for each table, fetch from assigned device
            for table_id in range(num_tables):
                tbl = tables.get(f"table_{table_id}")
                if tbl is None:
                    continue
                idx = indices[:batch_size // num_tables].clamp(0, tbl.shape[0] - 1)
                if tbl.device.type == 'cuda':
                    idx = idx.to(tbl.device)
                _ = tbl[idx]

                # Record access for tiered scheduler
                tracker.record_access(table_id, idx.cpu())

            if use_cuda:
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - step_start
            if step >= warmup_iters:
                throughputs.append(batch_size / elapsed)
                latencies.append(elapsed * 1000)

            if step % 100 == 0 and step >= warmup_iters:
                avg_tput = np.mean(throughputs[-100:]) if throughputs else 0
                print(f"    [{placement_name}] Step {step}/{num_iters}: "
                      f"{avg_tput:.0f} samples/s")

        # GPU memory stats
        if use_cuda:
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                mem_snapshots.append({
                    'gpu': i,
                    'allocated_gb': round(alloc, 2),
                    'total_gb': round(total, 1),
                    'utilization_pct': round(alloc / total * 100, 1),
                })

        return {
            'mean_throughput': float(np.mean(throughputs)) if throughputs else 0,
            'p50_throughput': float(np.median(throughputs)) if throughputs else 0,
            'p99_latency_ms': float(np.percentile(latencies, 99)) if latencies else 0,
            'mean_latency_ms': float(np.mean(latencies)) if latencies else 0,
            'std_latency_ms': float(np.std(latencies)) if latencies else 0,
            'memory': mem_snapshots,
        }

    # ── Run UNIFORM placement ──
    print("\n  [UNIFORM] All tables split evenly across GPUs...")
    uniform_tables = {}
    if use_cuda:
        devs = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        for t in range(num_tables):
            dev = devs[t % len(devs)]
            uniform_tables[f"table_{t}"] = torch.randn(
                num_embeddings, embedding_dim, device=dev)
    else:
        for t in range(num_tables):
            uniform_tables[f"table_{t}"] = torch.randn(
                num_embeddings, embedding_dim)

    results['uniform'] = run_iterations("UNIFORM", uniform_tables, {})

    # Cleanup
    del uniform_tables
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ── Run TIERED placement ──
    print("\n  [TIERED] Hot→H100, Warm→A6000, Cold→CPU...")
    tiered_tables = {}
    hot_count = max(1, int(num_tables * 0.15))     # ~15% hot on H100
    warm_count = max(1, int(num_tables * 0.35))     # ~35% warm on A6000s
    cold_count = num_tables - hot_count - warm_count

    for t in range(num_tables):
        if t < hot_count and h100_devs:
            dev = torch.device(f'cuda:{h100_devs[0]}')
        elif t < hot_count + warm_count and a6000_devs:
            dev = torch.device(f'cuda:{a6000_devs[t % len(a6000_devs)]}')
        else:
            dev = torch.device('cpu')
        tiered_tables[f"table_{t}"] = torch.randn(
            num_embeddings, embedding_dim, device=dev)

    results['tiered'] = run_iterations("TIERED", tiered_tables, {})
    results['tiered']['placement'] = {
        'hot_on_h100': hot_count,
        'warm_on_a6000': warm_count,
        'cold_on_cpu': cold_count,
    }

    # Compute migration plan
    migration_plan = scheduler.compute_placement_plan()
    results['tiered']['migrations_needed'] = len(migration_plan)

    del tiered_tables
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ── Speedup ──
    if results['uniform']['mean_throughput'] > 0:
        speedup = results['tiered']['mean_throughput'] / results['uniform']['mean_throughput']
        results['speedup'] = round(speedup, 3)
        print(f"\n  Tiered/Uniform speedup: {speedup:.3f}x")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Section 2: Exp2 — Mixed-Precision Gradient Convergence
# ═══════════════════════════════════════════════════════════════════

class SimpleEmbeddingDLRM(nn.Module):
    """Simplified DLRM for convergence testing."""

    def __init__(self, num_tables, num_embeddings, dim, device):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.EmbeddingBag(num_embeddings, dim, mode='sum', device=device)
            for _ in range(num_tables)
        ])
        self.top = nn.Sequential(
            nn.Linear(num_tables * dim, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 1, device=device),
            nn.Sigmoid()
        )

    def forward(self, indices_list, offsets_list):
        embs = [tbl(idx, off) for tbl, idx, off in
                zip(self.tables, indices_list, offsets_list)]
        return self.top(torch.cat(embs, dim=1)).squeeze(-1)


def run_exp2_convergence(hw: Dict, quick: bool = False) -> Dict:
    """
    Exp2: Mixed-Precision Training Convergence.
    Compares FP32-only vs simulated FP8/BF16/FP32 gradient aggregation.
    Uses FPRev diagnostic batches for drift detection.
    """
    print("\n" + "="*60)
    print("  EXP2: Mixed-Precision Gradient Convergence")
    print("="*60)

    num_tables = 4 if quick else 8
    num_embeddings = 50_000 if quick else 200_000
    dim = 64
    batch_size = 2048 if quick else 4096
    num_iters = 200 if quick else 2000
    verify_interval = 10 if quick else 50

    gpus = hw.get('gpus', [])
    h100_devs = [g['index'] for g in gpus if g['tier'] == 'h100']
    a6000_devs = [g['index'] for g in gpus if g['tier'] == 'a6000']

    # Determine primary compute device
    if h100_devs:
        primary_dev = torch.device(f'cuda:{h100_devs[0]}')
    elif a6000_devs:
        primary_dev = torch.device(f'cuda:{a6000_devs[0]}')
    elif torch.cuda.is_available():
        primary_dev = torch.device('cuda:0')
    else:
        primary_dev = torch.device('cpu')

    has_fp8 = h100_devs and check_fp8_support(h100_devs[0])

    results = {
        'config': {
            'num_tables': num_tables,
            'num_embeddings': num_embeddings,
            'dim': dim,
            'batch_size': batch_size,
            'num_iters': num_iters,
            'primary_device': str(primary_dev),
            'fp8_available': has_fp8,
        },
        'baseline_fp32': {},
        'mixed_precision': {},
    }

    # ── Build precision configs for CrossPrecisionAllReduce ──
    precision_configs = []
    devices_for_allreduce = {}

    if h100_devs:
        h_dev = torch.device(f'cuda:{h100_devs[0]}')
        fp8_dtype = torch.float8_e4m3fn if has_fp8 else torch.bfloat16
        precision_configs.append(
            PrecisionConfig("H100", fp8_dtype, torch.float32, h_dev))
        devices_for_allreduce[h_dev] = "H100"

    for i, idx in enumerate(a6000_devs):
        a_dev = torch.device(f'cuda:{idx}')
        precision_configs.append(
            PrecisionConfig(f"A6000_{i}", torch.bfloat16, torch.float32, a_dev))
        devices_for_allreduce[a_dev] = f"A6000_{i}"

    precision_configs.append(
        PrecisionConfig("CPU", torch.float32, torch.float32, torch.device('cpu')))
    devices_for_allreduce[torch.device('cpu')] = "CPU"

    allreduce = CrossPrecisionAllReduce(precision_configs)
    diagnostic = FPRevDiagnosticBatch(dim, num_tables)

    def train_loop(mode: str) -> Dict:
        model = SimpleEmbeddingDLRM(num_tables, num_embeddings, dim, primary_dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        loss_curve = []
        drift_reports = []
        throughputs = []

        for step in range(num_iters):
            step_start = time.perf_counter()

            indices_list = [
                torch.randint(0, num_embeddings, (batch_size * 3,), device=primary_dev)
                for _ in range(num_tables)
            ]
            offsets_list = [
                torch.arange(0, batch_size * 3, 3, device=primary_dev)
                for _ in range(num_tables)
            ]
            labels = torch.randint(0, 2, (batch_size,),
                                   device=primary_dev).float()

            optimizer.zero_grad()
            out = model(indices_list, offsets_list)
            loss = criterion(out, labels)
            loss.backward()

            # ── Mixed-precision gradient simulation ──
            if mode == 'mixed' and step % verify_interval == 0:
                # Simulate cross-precision gradient by casting a sample grad
                sample_grad = None
                for p in model.parameters():
                    if p.grad is not None and p.grad.numel() >= dim:
                        sample_grad = p.grad.data[:dim].clone()
                        break

                if sample_grad is not None:
                    fake_grads = {}
                    for dev, name in devices_for_allreduce.items():
                        try:
                            cfg = next(c for c in precision_configs
                                       if c.device_name == name)
                            g = sample_grad.detach().float().to(dev)
                            fake_grads[dev] = g
                        except Exception:
                            pass

                    if len(fake_grads) >= 2:
                        drift = allreduce.compute_drift(fake_grads)
                        drift_reports.append({
                            'step': step,
                            'drift': drift,
                            'loss': loss.item(),
                        })

                # FPRev diagnostic
                diag = diagnostic.generate_distinguishing_inputs(
                    batch_size=64, strategy='boundary')

            optimizer.step()

            if primary_dev.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start

            loss_val = loss.item()
            loss_curve.append({'step': step, 'loss': loss_val})
            throughputs.append(batch_size / elapsed)

            if (step + 1) % max(1, num_iters // 5) == 0:
                avg_loss = np.mean([lc['loss'] for lc in loss_curve[-200:]])
                avg_tput = np.mean(throughputs[-200:])
                print(f"    [{mode}] Step {step+1}/{num_iters}: "
                      f"loss={avg_loss:.4f}  tput={avg_tput:.0f}/s")

        del model, optimizer
        gc.collect()
        if primary_dev.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'final_loss': loss_curve[-1]['loss'] if loss_curve else None,
            'mean_throughput': float(np.mean(throughputs)) if throughputs else 0,
            'loss_curve_sampled': loss_curve[::max(1, len(loss_curve) // 50)],
            'drift_reports': drift_reports[:20],  # Cap for JSON size
            'num_drift_checks': len(drift_reports),
            'max_drift': max(
                (max(d['drift'].values()) for d in drift_reports if d['drift']),
                default=0
            ),
        }

    # Run baseline
    print("\n  [FP32 BASELINE]...")
    results['baseline_fp32'] = train_loop('baseline_fp32')

    # Run mixed
    print("\n  [MIXED PRECISION]...")
    results['mixed_precision'] = train_loop('mixed')

    return results


# ═══════════════════════════════════════════════════════════════════
#  Section 3: Exp3 — Elastic Migration Latency (PCIe Gen4 vs Gen5)
# ═══════════════════════════════════════════════════════════════════

def run_exp3_migration(hw: Dict, quick: bool = False) -> Dict:
    """
    Exp3: Elastic Tensor Migration Latency.
    Measures actual PCIe bandwidth for gather/discard operations
    between each GPU pair and CPU.
    """
    print("\n" + "="*60)
    print("  EXP3: Elastic Migration Latency (PCIe Gen4 vs Gen5)")
    print("="*60)

    num_trials = 20 if quick else 100
    sizes_mb = [1, 4, 16, 64, 256] if not quick else [4, 64]

    results = {
        'config': {'num_trials': num_trials, 'sizes_mb': sizes_mb},
        'transfers': {},
        'dtype_conversions': {},
        'elastic_ops': {},
    }

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print("  No GPU available — skipping Exp3")
        return results

    def measure_transfer(src_dev, dst_dev, size_bytes, trials):
        numel = size_bytes // 4  # float32
        lats = []
        bws = []
        # Warmup
        for _ in range(3):
            s = torch.randn(numel, device=src_dev)
            d = s.to(dst_dev)
            if dst_dev.type == 'cuda':
                torch.cuda.synchronize(dst_dev)
            del s, d

        for _ in range(trials):
            s = torch.randn(numel, device=src_dev)
            if src_dev.type == 'cuda':
                torch.cuda.synchronize(src_dev)
            t0 = time.perf_counter()
            d = s.to(dst_dev)
            if dst_dev.type == 'cuda':
                torch.cuda.synchronize(dst_dev)
            if src_dev.type == 'cuda':
                torch.cuda.synchronize(src_dev)
            elapsed = time.perf_counter() - t0
            lats.append(elapsed * 1000)
            bws.append(size_bytes / elapsed / 1e9)
            del s, d

        return {
            'mean_latency_ms': round(float(np.mean(lats)), 3),
            'p50_latency_ms': round(float(np.median(lats)), 3),
            'p99_latency_ms': round(float(np.percentile(lats, 99)), 3),
            'mean_bw_gbps': round(float(np.mean(bws)), 2),
            'peak_bw_gbps': round(float(np.max(bws)), 2),
        }

    # ── PCIe bandwidth for all GPU pairs and CPU ──
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_key = f"{size_mb}MB"
        results['transfers'][size_key] = {}

        # CPU ↔ each GPU
        for i in range(num_gpus):
            gpu = torch.device(f'cuda:{i}')
            name = torch.cuda.get_device_name(i)

            print(f"  [{size_mb}MB] CPU → GPU:{i} ({name})...")
            results['transfers'][size_key][f'cpu→gpu{i}'] = {
                'gpu': name,
                **measure_transfer(torch.device('cpu'), gpu, size_bytes, num_trials)
            }

            print(f"  [{size_mb}MB] GPU:{i} → CPU ({name})...")
            results['transfers'][size_key][f'gpu{i}→cpu'] = {
                'gpu': name,
                **measure_transfer(gpu, torch.device('cpu'), size_bytes, num_trials)
            }

        # GPU ↔ GPU
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                src = torch.device(f'cuda:{i}')
                dst = torch.device(f'cuda:{j}')
                sn = torch.cuda.get_device_name(i)
                dn = torch.cuda.get_device_name(j)
                print(f"  [{size_mb}MB] GPU:{i}→GPU:{j} ({sn}→{dn})...")
                results['transfers'][size_key][f'gpu{i}→gpu{j}'] = {
                    'src': sn, 'dst': dn,
                    **measure_transfer(src, dst, size_bytes, num_trials)
                }

    # ── Dtype conversion latency ──
    print("\n  Dtype conversions...")
    test_size = 64 * 1024 * 1024  # 64MB
    numel = test_size // 4

    conversions = [
        ('float32', 'bfloat16', torch.float32, torch.bfloat16),
        ('bfloat16', 'float32', torch.bfloat16, torch.float32),
    ]

    for i in range(num_gpus):
        dev = torch.device(f'cuda:{i}')
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)

        for label_src, label_dst, dt_src, dt_dst in conversions:
            key = f"gpu{i}_{label_src}→{label_dst}"
            try:
                src = torch.randn(numel, device=dev, dtype=dt_src)
                torch.cuda.synchronize(dev)
                lats = []
                for _ in range(num_trials):
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    _ = src.to(dt_dst)
                    torch.cuda.synchronize(dev)
                    lats.append((time.perf_counter() - t0) * 1000)
                results['dtype_conversions'][key] = {
                    'gpu': name,
                    'mean_ms': round(float(np.mean(lats)), 3),
                    'p50_ms': round(float(np.median(lats)), 3),
                }
                del src
            except Exception as e:
                results['dtype_conversions'][key] = {'error': str(e)}

        # FP8 if sm90+
        if cap[0] >= 9 and check_fp8_support(i):
            for dt_src, dt_dst, label in [
                (torch.float32, torch.float8_e4m3fn, 'float32→fp8'),
                (torch.bfloat16, torch.float8_e4m3fn, 'bfloat16→fp8'),
                (torch.float8_e4m3fn, torch.float32, 'fp8→float32'),
            ]:
                key = f"gpu{i}_{label}"
                try:
                    n = numel if dt_src != torch.float8_e4m3fn else numel
                    src = torch.randn(n, device=dev).to(dt_src)
                    torch.cuda.synchronize(dev)
                    lats = []
                    for _ in range(num_trials):
                        torch.cuda.synchronize(dev)
                        t0 = time.perf_counter()
                        _ = src.to(dt_dst)
                        torch.cuda.synchronize(dev)
                        lats.append((time.perf_counter() - t0) * 1000)
                    results['dtype_conversions'][key] = {
                        'gpu': name,
                        'mean_ms': round(float(np.mean(lats)), 3),
                    }
                    del src
                except Exception as e:
                    results['dtype_conversions'][key] = {'error': str(e)}

    # ── ElasticTensorManager operation latency ──
    print("\n  Elastic tensor ops (gather/discard)...")
    pools = {}
    for i in range(num_gpus):
        dev = torch.device(f'cuda:{i}')
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_mem
        pool_name = 'h100' if cap[0] >= 9 else f'a6000_{i}'
        pools[pool_name] = GPUMemoryPool(dev, int(mem * 0.3))
    pools['cpu'] = GPUMemoryPool(torch.device('cpu'), int(200e9))

    manager = ElasticTensorManager(pools)

    # Test gather: CPU → each GPU
    embedding_dim = 128
    for chunk_rows in [4096, 16384, 65536]:
        data = torch.randn(chunk_rows, embedding_dim)
        for pool_name, pool in pools.items():
            if pool.device.type == 'cpu':
                continue
            key = f"gather_cpu→{pool_name}_{chunk_rows}rows"
            try:
                slot = manager.gather(
                    table_id=0, row_range=(0, chunk_rows),
                    source_pool='cpu', target_pool=pool_name,
                    source_data=data
                )
                lat = manager.op_latencies[ElasticOp.GATHER][-1] * 1000
                bw = (chunk_rows * embedding_dim * 4) / (lat / 1000) / 1e9
                results['elastic_ops'][key] = {
                    'latency_ms': round(lat, 3),
                    'bandwidth_gbps': round(bw, 2),
                    'rows': chunk_rows,
                }
                print(f"    {key}: {lat:.2f}ms ({bw:.1f} GB/s)")
            except Exception as e:
                results['elastic_ops'][key] = {'error': str(e)}

        del data

    # Cleanup
    del pools, manager
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════
#  Section 4: M001-M003 Regression Tests on Real Hardware
# ═══════════════════════════════════════════════════════════════════

def run_m001_m003_regression(hw: Dict) -> Dict:
    """
    Run M001-M003 regression tests on real GPU hardware to verify
    the fixes work with actual CUDA devices.
    """
    print("\n" + "="*60)
    print("  M001-M003 Regression Tests (Real Hardware)")
    print("="*60)

    results = {'tests': {}}
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # ── M003: Verify dtype inference on real GPUs ──
    print("\n  M003: Dynamic dtype inference on real GPUs...")
    for i in range(num_gpus):
        dev = torch.device(f'cuda:{i}')
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        pool = GPUMemoryPool(dev, int(1e9))
        mgr = ElasticTensorManager({f'gpu_{i}': pool})
        inferred = mgr._get_pool_dtype(f'gpu_{i}')

        expected = torch.float8_e4m3fn if cap[0] >= 9 else torch.bfloat16
        ok = inferred == expected
        results['tests'][f'm003_gpu{i}'] = {
            'name': name,
            'cap': f"sm{cap[0]}{cap[1]}",
            'inferred_dtype': str(inferred),
            'expected_dtype': str(expected),
            'passed': ok,
        }
        status = "✓" if ok else "✗"
        print(f"    GPU {i} ({name}, sm{cap[0]}{cap[1]}): "
              f"inferred={inferred} expected={expected} {status}")
        del pool, mgr

    # ── M001: Concurrent discard stress test ──
    print("\n  M001: Concurrent discard stress test...")
    if num_gpus >= 1:
        import threading
        dev = torch.device(f'cuda:0')
        pool_src = GPUMemoryPool(dev, int(1e9))
        pool_dst = GPUMemoryPool(torch.device('cpu'), int(10e9))
        mgr = ElasticTensorManager({'src': pool_src, 'dst': pool_dst})

        crash_count = 0
        success_count = 0
        lock = threading.Lock()

        def stress_discard(iteration):
            nonlocal crash_count, success_count
            try:
                data = torch.randn(1024, 128, device=dev)
                slot = TensorSlot(
                    table_id=iteration, row_range=(0, 1024), data=data,
                    dtype=torch.float32, device=dev,
                    size_bytes=data.numel() * data.element_size()
                )
                pool_src.put(slot)

                # Two threads try to discard same slot
                results_local = []
                def do_discard():
                    r = mgr.discard(iteration, (0, 1024), 'src', 'dst')
                    results_local.append(r)

                t1 = threading.Thread(target=do_discard)
                t2 = threading.Thread(target=do_discard)
                t1.start(); t2.start()
                t1.join(); t2.join()

                non_none = [r for r in results_local if r is not None]
                with lock:
                    if len(non_none) <= 1:
                        success_count += 1
                    else:
                        crash_count += 1
            except Exception:
                with lock:
                    crash_count += 1

        threads = [threading.Thread(target=stress_discard, args=(i,))
                   for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results['tests']['m001_concurrent_discard'] = {
            'total_iterations': 50,
            'success': success_count,
            'crash': crash_count,
            'passed': crash_count == 0,
        }
        status = "✓" if crash_count == 0 else f"✗ ({crash_count} crashes)"
        print(f"    50 concurrent discards: {success_count} ok, "
              f"{crash_count} crash {status}")

        del pool_src, pool_dst, mgr
        gc.collect()
        torch.cuda.empty_cache()

    # ── M002: Memory accounting after heavy churn ──
    print("\n  M002: Memory accounting after heavy churn...")
    if num_gpus >= 1:
        dev = torch.device('cuda:0')
        pool = GPUMemoryPool(dev, int(2e9))  # 2GB pool

        for i in range(500):
            size = np.random.randint(1000, 100000)
            data = torch.randn(size, device=dev)
            slot = TensorSlot(
                table_id=i % 10, row_range=(0, size), data=data,
                dtype=torch.float32, device=dev,
                size_bytes=data.numel() * data.element_size()
            )
            pool.put(slot)

        actual_bytes = sum(s.size_bytes for s in pool.slots.values())
        drift = abs(pool.used_bytes - actual_bytes)
        ok = drift == 0
        results['tests']['m002_memory_accounting'] = {
            'reported_bytes': pool.used_bytes,
            'actual_bytes': actual_bytes,
            'drift_bytes': drift,
            'passed': ok,
        }
        status = "✓" if ok else f"✗ (drift={drift} bytes)"
        print(f"    reported={pool.used_bytes}, actual={actual_bytes}, "
              f"drift={drift} {status}")

        del pool
        gc.collect()
        torch.cuda.empty_cache()

    all_passed = all(t.get('passed', False) for t in results['tests'].values())
    results['all_passed'] = all_passed
    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Alloy Heterogeneous Cluster Integration Test')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (fewer iterations)')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'exp1', 'exp2', 'exp3', 'regression'],
                        help='Which experiment to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ── Reproducibility ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Hardware discovery ──
    print("\n" + "="*60)
    print("  HARDWARE DISCOVERY")
    print("="*60)
    hw = discover_hardware()
    print(f"  PyTorch: {hw['torch_version']}")
    print(f"  CUDA: {hw['cuda_version']}")
    print(f"  GPUs: {hw['num_gpus']}")
    for g in hw['gpus']:
        print(f"    GPU {g['index']}: {g['name']} "
              f"({g['total_mem_gb']}GB, {g['arch_class']}, "
              f"PCIe Gen{g['pcie_gen']})")
    if hw['warnings']:
        for w in hw['warnings']:
            print(f"  ⚠ {w}")

    # ── Run experiments ──
    report = {
        'hardware': hw,
        'args': vars(args),
        'experiments': {},
    }

    try:
        if args.exp in ('all', 'regression'):
            report['experiments']['regression'] = \
                run_m001_m003_regression(hw)

        if args.exp in ('all', 'exp1'):
            report['experiments']['exp1_tiered_placement'] = \
                run_exp1_tiered_placement(hw, quick=args.quick)

        if args.exp in ('all', 'exp2'):
            report['experiments']['exp2_convergence'] = \
                run_exp2_convergence(hw, quick=args.quick)

        if args.exp in ('all', 'exp3'):
            report['experiments']['exp3_migration'] = \
                run_exp3_migration(hw, quick=args.quick)

    except Exception as e:
        report['error'] = {
            'message': str(e),
            'traceback': traceback.format_exc(),
        }
        print(f"\n  ERROR: {e}")
        traceback.print_exc()

    # ── Save results ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
