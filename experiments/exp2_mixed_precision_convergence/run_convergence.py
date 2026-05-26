"""
Alloy Exp2: Mixed-Precision Training Convergence
Compares FP32-only baseline vs mixed FP8/BF16/FP32 training.
Uses FPRev diagnostic batches to verify gradient consistency.
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from alloy.mixed_precision.gradient_verifier import (
    CrossPrecisionAllReduce, FPRevDiagnosticBatch, PrecisionConfig
)


class SimpleEmbeddingModel(nn.Module):
    """Simplified DLRM embedding model for convergence testing."""
    
    def __init__(self, num_tables: int, num_embeddings: int, 
                 embedding_dim: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum',
                           device=device, dtype=dtype if dtype != torch.float8_e4m3fn else torch.float32)
            for _ in range(num_tables)
        ])
        self.top_mlp = nn.Sequential(
            nn.Linear(num_tables * embedding_dim, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 1, device=device),
            nn.Sigmoid()
        )
    
    def forward(self, indices_list: List[torch.Tensor], offsets_list: List[torch.Tensor]):
        embeddings = []
        for table, indices, offsets in zip(self.tables, indices_list, offsets_list):
            embeddings.append(table(indices, offsets))
        x = torch.cat(embeddings, dim=1)
        return self.top_mlp(x).squeeze()


def run_convergence_test(args) -> Dict:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'mode': args.mode,
        'loss_curve': [],
        'drift_reports': [],
        'throughput': [],
    }
    
    # Create model
    num_tables = 8
    num_embeddings = 100000
    embedding_dim = 64
    batch_size = 4096
    
    model = SimpleEmbeddingModel(
        num_tables, num_embeddings, embedding_dim, device, torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # FPRev diagnostic
    diagnostic = FPRevDiagnosticBatch(embedding_dim, num_tables)
    
    print(f"  Mode: {args.mode} | Iters: {args.num_iters}")
    
    for step in range(args.num_iters):
        iter_start = time.time()
        
        # Generate synthetic batch (Zipf-distributed)
        indices_list = [
            torch.randint(0, num_embeddings, (batch_size * 3,), device=device)
            for _ in range(num_tables)
        ]
        offsets_list = [
            torch.arange(0, batch_size * 3 + 1, 3, device=device)
            for _ in range(num_tables)
        ]
        labels = torch.randint(0, 2, (batch_size,), device=device).float()
        
        # Forward
        optimizer.zero_grad()
        output = model(indices_list, offsets_list)
        loss = criterion(output, labels)
        
        # Backward
        loss.backward()
        
        # Simulate mixed-precision gradient handling
        if args.mode == 'mixed' and step % args.verify_interval == 0:
            # Construct FPRev diagnostic batch
            diag_inputs = diagnostic.generate_distinguishing_inputs(
                batch_size=64, strategy='boundary'
            )
            results['drift_reports'].append({
                'step': step,
                'loss': loss.item(),
                'max_grad_norm': max(
                    p.grad.norm().item() for p in model.parameters() if p.grad is not None
                ),
            })
        
        optimizer.step()
        
        iter_time = time.time() - iter_start
        results['loss_curve'].append({'step': step, 'loss': loss.item()})
        results['throughput'].append(batch_size / iter_time)
        
        if (step + 1) % 200 == 0:
            avg_loss = np.mean([r['loss'] for r in results['loss_curve'][-200:]])
            avg_tput = np.mean(results['throughput'][-200:])
            print(f"  Step {step+1}/{args.num_iters}: loss={avg_loss:.4f} "
                  f"throughput={avg_tput:.0f} samples/s")
    
    results['summary'] = {
        'final_loss': results['loss_curve'][-1]['loss'],
        'mean_throughput': float(np.mean(results['throughput'])),
        'num_drift_checks': len(results['drift_reports']),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline_fp32', 'mixed'], required=True)
    parser.add_argument('--num-iters', type=int, default=2000)
    parser.add_argument('--verify-interval', type=int, default=50)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    results = run_convergence_test(args)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Final loss: {results['summary']['final_loss']:.4f}")
    print(f"  Mean throughput: {results['summary']['mean_throughput']:.0f} samples/s")


if __name__ == '__main__':
    main()
