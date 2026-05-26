"""
Alloy Mixed-Precision Gradient Aggregation & Verification
Integrates FPRev's distinguishing input methodology for cross-precision
gradient consistency verification across FP8 (H100) / BF16 (A6000) / FP32 (CPU).
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class PrecisionConfig:
    """Precision configuration per device tier."""
    device_name: str
    compute_dtype: torch.dtype
    accumulate_dtype: torch.dtype  # Accumulation may use higher precision
    device: torch.device
    
    @property
    def bits(self) -> int:
        dtype_bits = {
            torch.float8_e4m3fn: 8,
            torch.bfloat16: 16,
            torch.float16: 16,
            torch.float32: 32,
        }
        return dtype_bits.get(self.compute_dtype, 32)


class CrossPrecisionAllReduce:
    """
    Performs gradient allreduce across devices with different precisions.
    H100 computes in FP8, A6000 in BF16, CPU in FP32.
    Aggregation is done in FP32 to minimize numerical drift.
    """
    
    def __init__(self, precision_configs: List[PrecisionConfig]):
        self.configs = {pc.device: pc for pc in precision_configs}
        self._drift_history: List[float] = []
    
    def allreduce_gradients(
        self, 
        gradients: Dict[torch.device, torch.Tensor],
        reference_device: Optional[torch.device] = None
    ) -> Dict[torch.device, torch.Tensor]:
        """
        Perform cross-precision gradient allreduce.
        1. Upcast all gradients to FP32
        2. Sum and average
        3. Downcast back to each device's native precision
        """
        # Step 1: Collect all gradients in FP32 on CPU
        fp32_grads = []
        for device, grad in gradients.items():
            fp32_grad = grad.detach().float().cpu()
            fp32_grads.append(fp32_grad)
        
        # Step 2: Compute mean in FP32
        stacked = torch.stack(fp32_grads)
        mean_grad = stacked.mean(dim=0)
        
        # Step 3: Distribute back in native precision
        result = {}
        for device, grad in gradients.items():
            config = self.configs[device]
            result[device] = mean_grad.to(
                device=device,
                dtype=config.compute_dtype
            )
        
        return result
    
    def compute_drift(
        self,
        gradients: Dict[torch.device, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute numerical drift between precision paths.
        Returns max absolute and relative errors between each pair.
        """
        fp32_grads = {}
        for device, grad in gradients.items():
            config = self.configs[device]
            fp32_grads[config.device_name] = grad.detach().float().cpu()
        
        drift_report = {}
        names = list(fp32_grads.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = fp32_grads[names[i]], fp32_grads[names[j]]
                abs_err = (a - b).abs().max().item()
                rel_err = (abs_err / (a.abs().max().item() + 1e-10))
                key = f"{names[i]}_vs_{names[j]}"
                drift_report[f"{key}_abs"] = abs_err
                drift_report[f"{key}_rel"] = rel_err
        
        return drift_report


class FPRevDiagnosticBatch:
    """
    Constructs diagnostic batches using FPRev's distinguishing input method.
    These batches are designed to maximize numerical divergence between
    precision paths, enabling early detection of problematic drift.
    """
    
    def __init__(self, embedding_dim: int, num_tables: int):
        self.embedding_dim = embedding_dim
        self.num_tables = num_tables
    
    def generate_distinguishing_inputs(
        self,
        batch_size: int = 256,
        strategy: str = "boundary"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate inputs that maximize precision-dependent behavior differences.
        
        Strategies:
        - 'boundary': Values near precision boundaries (e.g., FP8 max/min)
        - 'cancellation': Values that cause catastrophic cancellation
        - 'accumulation': Long reduction chains that amplify rounding
        """
        inputs = {}
        
        if strategy == "boundary":
            # FP8 E4M3 range: max ≈ 448, min_normal ≈ 2^-6
            fp8_max = 448.0
            fp8_min_normal = 2 ** -6
            inputs['indices'] = torch.randint(0, 1000, (batch_size,))
            inputs['gradients'] = torch.tensor([
                fp8_max * (1 + 1e-3),   # Just above FP8 max → overflow in FP8
                fp8_min_normal * 0.5,    # Subnormal in FP8 but normal in BF16
                -fp8_max * 0.99,         # Near negative max
            ]).repeat(batch_size // 3 + 1)[:batch_size]
            
        elif strategy == "cancellation":
            # Create pairs that nearly cancel, exposing precision differences
            base = torch.randn(batch_size // 2, self.embedding_dim) * 100
            perturbation = torch.randn(batch_size // 2, self.embedding_dim) * 1e-4
            inputs['positive'] = base + perturbation
            inputs['negative'] = -base + perturbation  # Near cancellation
            inputs['indices'] = torch.randint(0, 1000, (batch_size,))
            
        elif strategy == "accumulation":
            # Long chains of small values that accumulate differently
            inputs['values'] = torch.full(
                (batch_size, self.embedding_dim), 
                1e-3  # Small enough that FP8 accumulation diverges
            )
            inputs['indices'] = torch.arange(batch_size)  # Sequential for ordered reduction
        
        return inputs
    
    def verify_consistency(
        self,
        allreduce_fn,
        max_relative_drift: float = 1e-2,
        num_diagnostic_batches: int = 10
    ) -> Tuple[bool, Dict]:
        """
        Run periodic diagnostic verification.
        Returns (is_consistent, detailed_report).
        """
        strategies = ["boundary", "cancellation", "accumulation"]
        all_drifts = []
        
        for i in range(num_diagnostic_batches):
            strategy = strategies[i % len(strategies)]
            inputs = self.generate_distinguishing_inputs(strategy=strategy)
            drift = allreduce_fn(inputs)
            all_drifts.append(drift)
        
        # Aggregate drift statistics
        max_drift = max(
            max(d.values()) for d in all_drifts if d
        ) if all_drifts else 0.0
        
        is_consistent = max_drift <= max_relative_drift
        
        report = {
            'is_consistent': is_consistent,
            'max_observed_drift': max_drift,
            'threshold': max_relative_drift,
            'num_batches_tested': num_diagnostic_batches,
            'per_batch_drifts': all_drifts,
        }
        
        if not is_consistent:
            logger.warning(
                f"Cross-precision drift {max_drift:.6f} exceeds threshold "
                f"{max_relative_drift:.6f}. Consider reducing FP8 usage."
            )
        
        return is_consistent, report


class MixedPrecisionTrainer:
    """
    Orchestrates mixed-precision embedding training across heterogeneous GPUs.
    """
    
    def __init__(
        self,
        h100_device: torch.device,
        a6000_devices: List[torch.device],
        verification_interval: int = 100,  # Verify every N steps
    ):
        self.precision_configs = [
            PrecisionConfig("H100", torch.float8_e4m3fn, torch.float32, h100_device),
        ] + [
            PrecisionConfig(f"A6000_{i}", torch.bfloat16, torch.float32, dev)
            for i, dev in enumerate(a6000_devices)
        ] + [
            PrecisionConfig("CPU", torch.float32, torch.float32, torch.device('cpu')),
        ]
        
        self.allreduce = CrossPrecisionAllReduce(self.precision_configs)
        self.diagnostic = FPRevDiagnosticBatch(embedding_dim=128, num_tables=26)
        self.verification_interval = verification_interval
        self.step = 0
        self.loss_history: List[float] = []
        self.drift_history: List[Dict] = []
    
    def train_step(self, batch) -> float:
        """Execute one training step with mixed-precision gradient sync."""
        self.step += 1
        
        # ... (forward/backward on each device with native precision)
        # Placeholder for actual training logic
        loss = 0.0
        
        # Periodic verification using FPRev diagnostic batches
        if self.step % self.verification_interval == 0:
            is_ok, report = self.diagnostic.verify_consistency(
                allreduce_fn=lambda inputs: self.allreduce.compute_drift(
                    {pc.device: torch.randn(128) for pc in self.precision_configs}
                ),
            )
            self.drift_history.append(report)
            if not is_ok:
                logger.warning(f"Step {self.step}: Precision drift detected!")
        
        self.loss_history.append(loss)
        return loss
