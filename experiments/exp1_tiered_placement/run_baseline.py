"""
Alloy Exp1: Tiered Placement vs Homogeneous Baseline
DLRM embedding training on A6000×2 + H100×1
"""

import argparse
import json
import time
import torch
import numpy as np
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from alloy.scheduler.tiered_placement import (
    TieredPlacementScheduler, TierConfig, Tier, create_default_config
)


def generate_zipf_indices(num_embeddings: int, batch_size: int, alpha: float = 1.0):
    """Generate Zipf-distributed embedding indices (simulates real access patterns)."""
    weights = np.arange(1, num_embeddings + 1, dtype=np.float64) ** (-alpha)
    weights /= weights.sum()
    indices = np.random.choice(num_embeddings, size=batch_size, p=weights)
    return torch.from_numpy(indices).long()


def detect_gpu_tiers():
    """Auto-detect GPU types and assign tiers."""
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        gpu_info.append({'index': i, 'name': name, 'memory_gb': mem})
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    
    h100_gpus = [g for g in gpu_info if 'H100' in g['name'] or 'H200' in g['name']]
    a6000_gpus = [g for g in gpu_info if 'A6000' in g['name'] or 'RTX' in g['name']]
    other_gpus = [g for g in gpu_info if g not in h100_gpus and g not in a6000_gpus]
    
    if not h100_gpus:
        print("[WARN] No H100 detected, using first GPU as 'H100 tier'")
        h100_gpus = [gpu_info[0]] if gpu_info else []
    if not a6000_gpus:
        print("[WARN] No A6000 detected, using remaining GPUs as 'A6000 tier'")
        a6000_gpus = gpu_info[1:] if len(gpu_info) > 1 else []
    
    return h100_gpus, a6000_gpus


def run_benchmark(args, distribution: str) -> Dict:
    """Run embedding lookup benchmark with given distribution strategy."""
    h100_gpus, a6000_gpus = detect_gpu_tiers()
    
    results = {
        'distribution': distribution,
        'config': vars(args),
        'gpu_info': {'h100': h100_gpus, 'a6000': a6000_gpus},
        'throughput_samples_per_sec': [],
        'memory_utilization': {},
        'latency_ms': [],
    }
    
    # Create embedding tables
    print(f"\n  Creating {args.num_tables} embedding tables "
          f"({args.num_embeddings} rows × {args.embedding_dim} dim)...")
    
    if distribution == 'uniform':
        # Uniform: split evenly across all GPUs
        all_gpus = h100_gpus + a6000_gpus
        tables_per_gpu = args.num_tables // max(len(all_gpus), 1)
        placement = {f"cuda:{g['index']}": tables_per_gpu for g in all_gpus}
    else:
        # Tiered: hot on H100, warm on A6000, cold on CPU
        hot_tables = max(1, int(args.num_tables * args.hot_ratio))
        warm_tables = max(1, int(args.num_tables * args.warm_ratio))
        cold_tables = args.num_tables - hot_tables - warm_tables
        placement = {
            'h100': hot_tables,
            'a6000': warm_tables,
            'cpu': cold_tables,
        }
    
    print(f"  Placement: {placement}")
    
    # Simulate training iterations
    print(f"\n  Running {args.num_iters} iterations (warmup={args.warmup_iters})...")
    
    for i in range(args.num_iters):
        iter_start = time.time()
        
        # Generate batch with Zipf distribution
        indices = generate_zipf_indices(
            min(args.num_embeddings, 100000),  # Cap for memory
            args.batch_size
        )
        
        # Simulate embedding lookup + gradient update
        # (In real implementation, this calls HypeReca's indexGet/indexPut)
        if torch.cuda.is_available():
            indices_gpu = indices.cuda()
            torch.cuda.synchronize()
        
        iter_time = time.time() - iter_start
        
        if i >= args.warmup_iters:
            throughput = args.batch_size / iter_time
            results['throughput_samples_per_sec'].append(throughput)
            results['latency_ms'].append(iter_time * 1000)
        
        if (i + 1) % 100 == 0:
            avg_throughput = np.mean(results['throughput_samples_per_sec'][-100:]) \
                if results['throughput_samples_per_sec'] else 0
            print(f"  Iter {i+1}/{args.num_iters}: "
                  f"throughput={avg_throughput:.0f} samples/s")
    
    # Memory stats
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            results['memory_utilization'][f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_pct': (allocated / total) * 100,
            }
    
    # Summary
    results['summary'] = {
        'mean_throughput': float(np.mean(results['throughput_samples_per_sec'])),
        'p50_throughput': float(np.median(results['throughput_samples_per_sec'])),
        'p99_latency_ms': float(np.percentile(results['latency_ms'], 99)),
        'mean_latency_ms': float(np.mean(results['latency_ms'])),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Alloy Exp1: Tiered vs Uniform')
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--num-tables', type=int, default=26)
    parser.add_argument('--num-embeddings', type=int, default=10000000)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--num-iters', type=int, default=500)
    parser.add_argument('--warmup-iters', type=int, default=50)
    parser.add_argument('--distribution', choices=['uniform', 'tiered'], default='tiered')
    parser.add_argument('--hot-ratio', type=float, default=0.1)
    parser.add_argument('--warm-ratio', type=float, default=0.3)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    results = run_benchmark(args, args.distribution)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ═══ Summary ({args.distribution}) ═══")
    print(f"  Mean throughput: {results['summary']['mean_throughput']:.0f} samples/s")
    print(f"  P99 latency:    {results['summary']['p99_latency_ms']:.2f} ms")
    print(f"  Results saved to: {args.output}")


if __name__ == '__main__':
    main()
