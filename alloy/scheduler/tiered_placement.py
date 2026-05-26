"""
Alloy Tiered Placement Scheduler
Extends HypeReca's 2-level (GPU/CPU) to 4-level hierarchy:
  Tier 0: H100 HBM3  (hot embeddings, FP8)
  Tier 1: A6000 GDDR6 (warm embeddings, BF16)
  Tier 2: CPU DRAM    (cold embeddings, FP32)
  Tier 3: SSD         (archived embeddings)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
import threading
import time
import logging

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    H100_HBM3 = 0    # Hot: FP8 on H100
    A6000_GDDR6 = 1  # Warm: BF16 on A6000
    CPU_DRAM = 2      # Cold: FP32 on CPU
    SSD_ARCHIVE = 3   # Archived: FP32 on SSD


@dataclass
class TierConfig:
    """Configuration for each memory tier."""
    tier: Tier
    capacity_bytes: int
    dtype: torch.dtype
    device: torch.device
    pcie_gen: int  # PCIe generation (4 for A6000, 5 for H100)
    bandwidth_gbps: float  # Measured PCIe bandwidth
    
    @property
    def capacity_gb(self) -> float:
        return self.capacity_bytes / (1024 ** 3)


@dataclass
class EmbeddingMeta:
    """Metadata for tracking embedding access patterns."""
    table_id: int
    row_start: int
    row_end: int
    current_tier: Tier
    access_count: int = 0
    last_access_time: float = 0.0
    cumulative_gradient_norm: float = 0.0
    migration_count: int = 0


class AccessFrequencyTracker:
    """
    Tracks embedding access frequency using exponential moving average.
    Used to decide tier placement based on hotness.
    """
    
    def __init__(self, decay_factor: float = 0.95, window_size: int = 1000):
        self.decay_factor = decay_factor
        self.window_size = window_size
        self._freq_map: Dict[Tuple[int, int], float] = {}  # (table_id, row_range) -> EMA freq
        self._step = 0
    
    def record_access(self, table_id: int, indices: torch.Tensor):
        """Record embedding access from a training batch."""
        self._step += 1
        unique_indices = indices.unique().cpu().numpy()
        
        for idx in unique_indices:
            key = (table_id, int(idx))
            old = self._freq_map.get(key, 0.0)
            self._freq_map[key] = old * self.decay_factor + (1.0 - self.decay_factor)
        
        # Decay all non-accessed entries periodically
        if self._step % self.window_size == 0:
            for k in self._freq_map:
                if k not in {(table_id, int(i)) for i in unique_indices}:
                    self._freq_map[k] *= self.decay_factor
    
    def get_frequency(self, table_id: int, row_idx: int) -> float:
        return self._freq_map.get((table_id, row_idx), 0.0)
    
    def classify_tier(self, table_id: int, row_idx: int,
                      hot_threshold: float = 0.3,
                      warm_threshold: float = 0.05) -> Tier:
        """Classify embedding row into a tier based on access frequency."""
        freq = self.get_frequency(table_id, row_idx)
        if freq >= hot_threshold:
            return Tier.H100_HBM3
        elif freq >= warm_threshold:
            return Tier.A6000_GDDR6
        else:
            return Tier.CPU_DRAM


class TieredPlacementScheduler:
    """
    Core scheduler that manages embedding placement across the 4-tier hierarchy.
    Integrates with HypeReca's embedding storage and mTuner's elastic operations.
    """
    
    def __init__(self, tier_configs: List[TierConfig]):
        self.tiers = {tc.tier: tc for tc in tier_configs}
        self.tracker = AccessFrequencyTracker()
        self.embeddings: Dict[Tuple[int, int], EmbeddingMeta] = {}
        self._tier_usage: Dict[Tier, int] = {t: 0 for t in Tier}
        self._lock = threading.Lock()
        self._migration_log: List[dict] = []
    
    def register_embedding_table(self, table_id: int, num_rows: int,
                                  embedding_dim: int, chunk_size: int = 4096):
        """Register an embedding table and do initial placement."""
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            # Initial placement: everything starts on CPU
            meta = EmbeddingMeta(
                table_id=table_id,
                row_start=start,
                row_end=end,
                current_tier=Tier.CPU_DRAM
            )
            self.embeddings[(table_id, start)] = meta
            chunk_bytes = (end - start) * embedding_dim * 4  # FP32 initially
            self._tier_usage[Tier.CPU_DRAM] += chunk_bytes
    
    def compute_placement_plan(self) -> List[Tuple[EmbeddingMeta, Tier]]:
        """
        Compute optimal placement plan based on current access frequencies.
        Returns list of (embedding_meta, target_tier) for migrations needed.
        """
        migrations = []
        
        for key, meta in self.embeddings.items():
            # Determine target tier based on frequency
            target_tier = self.tracker.classify_tier(
                meta.table_id, meta.row_start
            )
            
            # Check capacity constraints
            if not self._has_capacity(target_tier, meta):
                # Fall back to next tier
                target_tier = Tier(min(target_tier.value + 1, Tier.SSD_ARCHIVE.value))
            
            if target_tier != meta.current_tier:
                migrations.append((meta, target_tier))
        
        # Sort migrations: promote hot data first, then demote cold data
        migrations.sort(key=lambda x: (x[1].value, -self.tracker.get_frequency(
            x[0].table_id, x[0].row_start)))
        
        return migrations
    
    def execute_migration(self, meta: EmbeddingMeta, target_tier: Tier,
                          embedding_data: torch.Tensor) -> torch.Tensor:
        """
        Execute a single embedding migration between tiers.
        Handles dtype conversion (FP32 ↔ BF16 ↔ FP8).
        Returns the migrated tensor.
        """
        source_config = self.tiers[meta.current_tier]
        target_config = self.tiers[target_tier]
        
        start_time = time.time()
        
        # Cast to target precision
        migrated = embedding_data.to(
            device=target_config.device,
            dtype=target_config.dtype
        )
        
        elapsed = time.time() - start_time
        data_bytes = embedding_data.numel() * embedding_data.element_size()
        
        with self._lock:
            meta.current_tier = target_tier
            meta.migration_count += 1
            self._migration_log.append({
                'table_id': meta.table_id,
                'row_range': (meta.row_start, meta.row_end),
                'source_tier': source_config.tier.name,
                'target_tier': target_config.tier.name,
                'data_bytes': data_bytes,
                'latency_ms': elapsed * 1000,
                'bandwidth_gbps': (data_bytes / elapsed / 1e9) if elapsed > 0 else 0,
                'pcie_gen_source': source_config.pcie_gen,
                'pcie_gen_target': target_config.pcie_gen,
            })
        
        logger.info(
            f"Migrated table={meta.table_id} rows=[{meta.row_start}:{meta.row_end}] "
            f"{source_config.tier.name} → {target_config.tier.name} "
            f"latency={elapsed*1000:.2f}ms"
        )
        
        return migrated
    
    def _has_capacity(self, tier: Tier, meta: EmbeddingMeta) -> bool:
        """Check if a tier has capacity for the given embedding chunk."""
        if tier not in self.tiers:
            return False
        config = self.tiers[tier]
        chunk_size = (meta.row_end - meta.row_start) * 4  # rough estimate
        return self._tier_usage[tier] + chunk_size <= config.capacity_bytes
    
    def get_migration_stats(self) -> dict:
        """Return migration statistics for analysis."""
        if not self._migration_log:
            return {}
        
        pcie4_latencies = [m['latency_ms'] for m in self._migration_log 
                          if m['pcie_gen_target'] == 4]
        pcie5_latencies = [m['latency_ms'] for m in self._migration_log 
                          if m['pcie_gen_target'] == 5]
        
        return {
            'total_migrations': len(self._migration_log),
            'pcie4_avg_latency_ms': np.mean(pcie4_latencies) if pcie4_latencies else 0,
            'pcie5_avg_latency_ms': np.mean(pcie5_latencies) if pcie5_latencies else 0,
            'pcie4_count': len(pcie4_latencies),
            'pcie5_count': len(pcie5_latencies),
            'tier_distribution': {
                tier.name: sum(1 for m in self.embeddings.values() if m.current_tier == tier)
                for tier in Tier
            }
        }


def create_default_config(
    h100_memory_gb: float = 80.0,
    a6000_memory_gb: float = 48.0,
    cpu_memory_gb: float = 128.0,
    ssd_capacity_gb: float = 1000.0,
) -> List[TierConfig]:
    """Create default tier configuration for A6000×2 + H100×1 + CPU."""
    return [
        TierConfig(
            tier=Tier.H100_HBM3,
            capacity_bytes=int(h100_memory_gb * 0.7 * 1024**3),  # Reserve 30% for model
            dtype=torch.float8_e4m3fn,  # FP8 on H100
            device=torch.device('cuda:0'),  # Assuming H100 is cuda:0
            pcie_gen=5,
            bandwidth_gbps=64.0,  # PCIe Gen5 x16
        ),
        TierConfig(
            tier=Tier.A6000_GDDR6,
            capacity_bytes=int(a6000_memory_gb * 0.7 * 1024**3),
            dtype=torch.bfloat16,  # BF16 on A6000
            device=torch.device('cuda:1'),  # First A6000
            pcie_gen=4,
            bandwidth_gbps=32.0,  # PCIe Gen4 x16
        ),
        TierConfig(
            tier=Tier.CPU_DRAM,
            capacity_bytes=int(cpu_memory_gb * 0.8 * 1024**3),
            dtype=torch.float32,  # FP32 on CPU
            device=torch.device('cpu'),
            pcie_gen=0,
            bandwidth_gbps=25.6,  # DDR4-3200 dual channel
        ),
        TierConfig(
            tier=Tier.SSD_ARCHIVE,
            capacity_bytes=int(ssd_capacity_gb * 1024**3),
            dtype=torch.float32,
            device=torch.device('cpu'),  # Staged via CPU
            pcie_gen=0,
            bandwidth_gbps=7.0,  # NVMe Gen4
        ),
    ]
