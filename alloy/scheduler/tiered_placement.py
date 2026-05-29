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
    embedding_dim: int = 0  # needed for dtype-aware byte accounting (R10 fix)
    access_count: int = 0
    last_access_time: float = 0.0
    cumulative_gradient_norm: float = 0.0
    migration_count: int = 0
    last_migration_time: float = 0.0  # M008: wall-clock of last tier change
    pending_target: Optional[Tier] = None  # M007: tier proposed but not yet confirmed
    pending_streak: int = 0  # M007: consecutive plans agreeing on pending_target


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

    def classify_tier_hysteresis(self, table_id: int, row_idx: int,
                                 current_tier: Tier,
                                 hot_threshold: float = 0.3,
                                 warm_threshold: float = 0.05,
                                 archive_threshold: float = 0.001,
                                 band: float = 0.2) -> Tier:
        """
        M007: Hysteresis-aware classification to prevent hot<->warm ping-pong.

        A row sitting near a boundary (e.g. freq ~0.30 against hot_threshold)
        would otherwise flip tier every step as EMA jitters by a few percent.
        We split each boundary into an upper (promote) and lower (demote) edge
        separated by a dead band. Promotion requires crossing the *higher*
        edge; demotion requires falling below the *lower* edge. Inside the band
        the row keeps its current tier, so jitter no longer triggers migration.

        band is the fractional width of the dead zone relative to each
        threshold (0.2 → ±10% around the nominal threshold).

        R11 fix: the original classifier could only return H100/A6000/CPU, so
        the fourth tier (SSD_ARCHIVE) of this explicitly 4-tier design was
        unreachable — rows that effectively went idle still pinned CPU DRAM.
        We add an archive_threshold below which a (truly cold) row is demoted
        to SSD, with the same hysteresis discipline as the other boundaries.
        """
        freq = self.get_frequency(table_id, row_idx)

        hot_up = hot_threshold * (1.0 + band / 2.0)
        hot_down = hot_threshold * (1.0 - band / 2.0)
        warm_up = warm_threshold * (1.0 + band / 2.0)
        warm_down = warm_threshold * (1.0 - band / 2.0)
        arch_up = archive_threshold * (1.0 + band / 2.0)
        arch_down = archive_threshold * (1.0 - band / 2.0)

        # Resolve the "naive" tier the row would want at each edge.
        if freq >= hot_up:
            desired = Tier.H100_HBM3
        elif freq >= warm_up:
            desired = Tier.A6000_GDDR6
        elif freq >= arch_up:
            desired = Tier.CPU_DRAM
        else:
            desired = Tier.SSD_ARCHIVE

        # If the row is already hotter than desired, only demote once it has
        # truly fallen below the lower edge; otherwise hold (dead band).
        if current_tier == Tier.H100_HBM3:
            if freq >= hot_down:
                return Tier.H100_HBM3
        elif current_tier == Tier.A6000_GDDR6:
            if warm_down <= freq < hot_up:
                return Tier.A6000_GDDR6
        elif current_tier == Tier.CPU_DRAM:
            if arch_down <= freq < warm_up:
                return Tier.CPU_DRAM

        return desired


class TieredPlacementScheduler:
    """
    Core scheduler that manages embedding placement across the 4-tier hierarchy.
    Integrates with HypeReca's embedding storage and mTuner's elastic operations.
    """
    
    def __init__(self, tier_configs: List[TierConfig],
                 migration_cooldown_s: float = 5.0,
                 confirm_streak: int = 3,
                 hysteresis_band: float = 0.2):
        self.tiers = {tc.tier: tc for tc in tier_configs}
        self.tracker = AccessFrequencyTracker()
        self.embeddings: Dict[Tuple[int, int], EmbeddingMeta] = {}
        self._tier_usage: Dict[Tier, int] = {t: 0 for t in Tier}
        self._lock = threading.Lock()
        self._migration_log: List[dict] = []
        # M008: minimum wall-clock seconds a chunk must rest before it may
        # migrate again. Caps churn at 1/cooldown migrations per chunk.
        self.migration_cooldown_s = migration_cooldown_s
        # M007/M009: how many consecutive plans must agree on a new target
        # before the migration actually fires. Filters transient hotness spikes.
        self.confirm_streak = max(1, confirm_streak)
        self.hysteresis_band = hysteresis_band
        self._suppressed_migrations = 0  # diagnostics: how many we damped
    
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
                current_tier=Tier.CPU_DRAM,
                embedding_dim=embedding_dim,
            )
            self.embeddings[(table_id, start)] = meta
            # R10: dtype-aware byte size for the row's *current* tier.
            chunk_bytes = self._chunk_bytes(meta, Tier.CPU_DRAM)
            self._tier_usage[Tier.CPU_DRAM] += chunk_bytes
    
    def compute_placement_plan(self) -> List[Tuple[EmbeddingMeta, Tier]]:
        """
        Compute optimal placement plan based on current access frequencies.
        Returns list of (embedding_meta, target_tier) for migrations needed.

        M007: tier selection now uses a hysteresis band so a row hovering near
              a threshold no longer flips every step.
        M008: a chunk that migrated within migration_cooldown_s is skipped,
              capping per-chunk migration rate.
        M009: a new target must be confirmed by confirm_streak consecutive
              plans before it fires, filtering transient spikes. Confirmation
              state lives on the meta so it survives across calls.
        """
        now = time.time()
        migrations = []

        with self._lock:
            # In-plan capacity reservation. compute_placement_plan() decides a
            # whole batch of migrations against a snapshot of _tier_usage, but
            # those migrations have not executed yet. Without reserving, every
            # candidate sees the same free space and we over-admit (e.g. 3 hot
            # chunks all "fit" a 2-chunk H100, then all migrate → OOM). We
            # track bytes provisionally consumed/released by earlier decisions
            # in this same pass and fold them into the capacity check. This is
            # the resource-reservation discipline Ray's PlacementGroup and the
            # K8s scheduler use during a scheduling cycle.
            reserved = {t: 0 for t in Tier}

            def has_capacity_with_reservations(tier, meta):
                cfg = self.tiers.get(tier)
                if cfg is None:
                    return False
                incoming = 0 if meta.current_tier == tier else self._chunk_bytes(meta, tier)
                return (self._tier_usage[tier] + reserved[tier] + incoming
                        <= cfg.capacity_bytes)

            for key, meta in self.embeddings.items():
                # M007: hysteresis-aware desired tier.
                target_tier = self.tracker.classify_tier_hysteresis(
                    meta.table_id, meta.row_start, meta.current_tier,
                    band=self.hysteresis_band,
                )

                # Capacity fallback: walk DOWN the tier hierarchy until one has
                # room (accounting for in-plan reservations), bottoming out at
                # SSD. The previous single-step fallback could still pick a tier
                # that was already full from earlier same-plan admissions.
                while (target_tier != Tier.SSD_ARCHIVE and
                       not has_capacity_with_reservations(target_tier, meta)):
                    target_tier = Tier(target_tier.value + 1)

                if target_tier == meta.current_tier:
                    # Stable: clear any half-built confirmation streak.
                    meta.pending_target = None
                    meta.pending_streak = 0
                    continue

                # M009: require the same target across confirm_streak plans.
                if meta.pending_target == target_tier:
                    meta.pending_streak += 1
                else:
                    meta.pending_target = target_tier
                    meta.pending_streak = 1

                if meta.pending_streak < self.confirm_streak:
                    self._suppressed_migrations += 1
                    continue

                # M008: cooldown gate — respect minimum rest between moves.
                if (meta.last_migration_time > 0.0 and
                        now - meta.last_migration_time < self.migration_cooldown_s):
                    self._suppressed_migrations += 1
                    continue

                # Reserve target capacity and release the source's footprint
                # for the remainder of this planning pass.
                reserved[target_tier] += self._chunk_bytes(meta, target_tier)
                reserved[meta.current_tier] -= self._chunk_bytes(meta, meta.current_tier)
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
            # R9: maintain the tier occupancy invariant that _has_capacity
            # reads. Without this, _tier_usage stays frozen at registration
            # values and the capacity guard never sees H100/A6000 fill up,
            # leading to over-admission and OOM on real hardware. We account
            # by the row's *native* byte size in each tier (R10): the source
            # tier releases its (possibly different-dtype) footprint, the
            # target tier gains its own. Mirrors vLLM's block-pool occupancy
            # counter and DeepSpeed's partition accounting.
            src_bytes = self._chunk_bytes(meta, meta.current_tier)
            dst_bytes = self._chunk_bytes(meta, target_tier)
            self._tier_usage[meta.current_tier] = max(
                0, self._tier_usage[meta.current_tier] - src_bytes)
            self._tier_usage[target_tier] += dst_bytes

            meta.current_tier = target_tier
            meta.migration_count += 1
            meta.last_migration_time = time.time()  # M008: start cooldown clock
            meta.pending_target = None  # M009: streak consumed
            meta.pending_streak = 0
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
    
    def _tier_dtype_bytes(self, tier: Tier) -> int:
        """Bytes-per-element for a tier's native dtype.

        R10: the previous code hardcoded 4 (FP32) everywhere, a 4x
        overestimate for the FP8 hot tier and 2x for the BF16 warm tier,
        which made _has_capacity reject placements that would actually fit.
        We derive the size from the tier's configured dtype instead.
        """
        cfg = self.tiers.get(tier)
        if cfg is None:
            return 4
        try:
            # torch dtypes expose itemsize via a zero-dim tensor; cache-free
            # lookup is fine since this is called O(migrations), not O(rows).
            return torch.empty(0, dtype=cfg.dtype).element_size()
        except Exception:
            return 4

    def _chunk_bytes(self, meta: EmbeddingMeta, tier: Tier) -> int:
        """Footprint of an embedding chunk *in a specific tier's dtype*."""
        rows = meta.row_end - meta.row_start
        dim = meta.embedding_dim if meta.embedding_dim > 0 else 1
        return rows * dim * self._tier_dtype_bytes(tier)

    def _has_capacity(self, tier: Tier, meta: EmbeddingMeta) -> bool:
        """Check if a tier has capacity for the given embedding chunk."""
        if tier not in self.tiers:
            return False
        config = self.tiers[tier]
        # R10: account for the chunk's footprint *in this tier's dtype*.
        # If the row already lives in this tier, it does not double-count.
        incoming = 0 if meta.current_tier == tier else self._chunk_bytes(meta, tier)
        return self._tier_usage[tier] + incoming <= config.capacity_bytes
    
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
            'suppressed_migrations': self._suppressed_migrations,
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
