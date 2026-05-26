"""
Alloy Exp3: Elastic Tensor Dynamic Migration Overhead
Measures gather/discard latency across PCIe Gen4 (A6000) vs Gen5 (H100).
"""

import argparse
import json
import time
import torch
import numpy as np


def measure_transfer(src_device, dst_device, tensor_size_bytes, num_trials, dtype=torch.float32):
    """Measure CPU↔GPU or GPU↔GPU transfer latency."""
    numel = tensor_size_bytes // torch.tensor([], dtype=dtype).element_size()
    
    latencies = []
    bandwidths = []
    
    for _ in range(num_trials):
        src_tensor = torch.randn(numel, device=src_device, dtype=dtype)
        
        if src_device.type == 'cuda':
            torch.cuda.synchronize(src_device)
        if dst_device.type == 'cuda':
            torch.cuda.synchronize(dst_device)
        
        start = time.perf_counter()
        dst_tensor = src_tensor.to(device=dst_device)
        
        if dst_device.type == 'cuda':
            torch.cuda.synchronize(dst_device)
        
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms
        bandwidths.append(tensor_size_bytes / elapsed / 1e9)  # GB/s
        
        del src_tensor, dst_tensor
    
    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'p50_latency_ms': float(np.median(latencies)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'std_latency_ms': float(np.std(latencies)),
        'mean_bandwidth_gbps': float(np.mean(bandwidths)),
        'peak_bandwidth_gbps': float(np.max(bandwidths)),
    }


def measure_dtype_conversion(device, tensor_size_bytes, src_dtype, dst_dtype, num_trials):
    """Measure dtype conversion overhead (e.g., FP32→BF16, BF16→FP8)."""
    numel = tensor_size_bytes // torch.tensor([], dtype=src_dtype).element_size()
    
    latencies = []
    for _ in range(num_trials):
        src = torch.randn(numel, device=device, dtype=src_dtype)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        start = time.perf_counter()
        dst = src.to(dtype=dst_dtype)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed * 1000)
        del src, dst
    
    return {
        'conversion': f'{src_dtype}→{dst_dtype}',
        'mean_latency_ms': float(np.mean(latencies)),
        'p50_latency_ms': float(np.median(latencies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-size-mb', type=int, default=64)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    tensor_size_bytes = args.tensor_size_mb * 1024 * 1024
    
    results = {
        'tensor_size_mb': args.tensor_size_mb,
        'num_trials': args.num_trials,
        'transfers': {},
        'conversions': {},
    }
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  Tensor size: {args.tensor_size_mb} MB | GPUs: {num_gpus}")
    
    # CPU → GPU transfers (gather operations)
    for i in range(num_gpus):
        gpu = torch.device(f'cuda:{i}')
        gpu_name = torch.cuda.get_device_name(i)
        
        print(f"  Measuring CPU → GPU:{i} ({gpu_name})...")
        results['transfers'][f'cpu_to_gpu{i}'] = {
            'gpu_name': gpu_name,
            **measure_transfer(torch.device('cpu'), gpu, tensor_size_bytes, args.num_trials)
        }
        
        print(f"  Measuring GPU:{i} → CPU ({gpu_name})...")
        results['transfers'][f'gpu{i}_to_cpu'] = {
            'gpu_name': gpu_name,
            **measure_transfer(gpu, torch.device('cpu'), tensor_size_bytes, args.num_trials)
        }
    
    # GPU ↔ GPU transfers (cross-tier migration)
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue
            src_name = torch.cuda.get_device_name(i)
            dst_name = torch.cuda.get_device_name(j)
            print(f"  Measuring GPU:{i} ({src_name}) → GPU:{j} ({dst_name})...")
            results['transfers'][f'gpu{i}_to_gpu{j}'] = {
                'src_gpu': src_name,
                'dst_gpu': dst_name,
                **measure_transfer(
                    torch.device(f'cuda:{i}'), torch.device(f'cuda:{j}'),
                    tensor_size_bytes, args.num_trials
                )
            }
    
    # Dtype conversion overhead
    for i in range(num_gpus):
        gpu = torch.device(f'cuda:{i}')
        gpu_name = torch.cuda.get_device_name(i)
        
        conversions = [
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ]
        # FP8 only on H100+ (sm90)
        cap = torch.cuda.get_device_capability(i)
        if cap[0] >= 9:
            conversions.extend([
                (torch.float32, torch.float8_e4m3fn),
                (torch.bfloat16, torch.float8_e4m3fn),
                (torch.float8_e4m3fn, torch.float32),
            ])
        
        for src_dt, dst_dt in conversions:
            key = f"gpu{i}_{src_dt}_to_{dst_dt}"
            print(f"  Measuring {src_dt}→{dst_dt} on GPU:{i}...")
            try:
                results['conversions'][key] = {
                    'gpu_name': gpu_name,
                    **measure_dtype_conversion(gpu, tensor_size_bytes, src_dt, dst_dt, args.num_trials)
                }
            except Exception as e:
                results['conversions'][key] = {'error': str(e)}
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n  ═══ Transfer Summary ({args.tensor_size_mb} MB) ═══")
    for name, data in results['transfers'].items():
        if 'mean_latency_ms' in data:
            print(f"  {name}: {data['mean_latency_ms']:.2f} ms "
                  f"({data['mean_bandwidth_gbps']:.1f} GB/s)")


if __name__ == '__main__':
    main()
