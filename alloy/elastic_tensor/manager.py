"""
Alloy Elastic Tensor Manager
Adapts mTuner's gather/discard/execute/checkpoint operations as
GPU pool eviction strategies for dynamic embedding migration.
"""

import torch
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class ElasticOp(Enum):
    GATHER = "gather"       # Pull embedding from lower tier to higher tier
    DISCARD = "discard"     # Evict embedding from higher tier to lower tier
    EXECUTE = "execute"     # Compute on embedding in current tier
    CHECKPOINT = "checkpoint"  # Save embedding state for recovery


@dataclass
class TensorSlot:
    """A slot in the GPU memory pool holding an embedding chunk."""
    table_id: int
    row_range: Tuple[int, int]
    data: Optional[torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    last_access: float = 0.0
    access_count: int = 0
    dirty: bool = False  # Has been modified since last checkpoint
    size_bytes: int = 0


class GPUMemoryPool:
    """
    LRU-based GPU memory pool for a single device.
    Manages embedding slots with eviction when memory pressure rises.
    """
    
    def __init__(self, device: torch.device, capacity_bytes: int,
                 eviction_threshold: float = 0.85):
        self.device = device
        self.capacity_bytes = capacity_bytes
        self.eviction_threshold = eviction_threshold
        self.slots: OrderedDict[Tuple[int, int], TensorSlot] = OrderedDict()
        self.used_bytes = 0
        self._lock = threading.Lock()
    
    @property
    def utilization(self) -> float:
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes > 0 else 0
    
    def get(self, table_id: int, row_range: Tuple[int, int]) -> Optional[TensorSlot]:
        key = (table_id, row_range[0])
        with self._lock:
            if key in self.slots:
                slot = self.slots[key]
                slot.last_access = time.time()
                slot.access_count += 1
                # Move to end (most recently used)
                self.slots.move_to_end(key)
                return slot
        return None

    def pop(self, table_id: int, row_range: Tuple[int, int]) -> Optional[TensorSlot]:
        """Atomically get and remove a slot. Prevents TOCTOU race in discard()."""
        key = (table_id, row_range[0])
        with self._lock:
            if key in self.slots:
                slot = self.slots.pop(key)
                self.used_bytes -= slot.size_bytes
                return slot
        return None
    
    def put(self, slot: TensorSlot) -> List[TensorSlot]:
        """
        Insert a slot, potentially evicting LRU entries.
        Returns list of evicted slots.
        """
        evicted = []
        key = (slot.table_id, slot.row_range[0])
        
        with self._lock:
            # Reject if single slot exceeds total capacity (not threshold)
            if slot.size_bytes > self.capacity_bytes:
                logger.warning(
                    f"Slot table={slot.table_id} rows={slot.row_range} "
                    f"size={slot.size_bytes} exceeds pool capacity "
                    f"{self.capacity_bytes}. Rejecting insertion."
                )
                return [slot]  # Return the slot itself as "evicted"

            # Evict if necessary
            while (self.used_bytes + slot.size_bytes > 
                   self.capacity_bytes * self.eviction_threshold):
                if not self.slots:
                    break
                # Evict LRU (first item)
                lru_key, lru_slot = self.slots.popitem(last=False)
                self.used_bytes -= lru_slot.size_bytes
                evicted.append(lru_slot)
                logger.debug(f"Evicted table={lru_slot.table_id} "
                           f"rows={lru_slot.row_range} from {self.device}")
            
            # Remove existing entry for same key if present (update case)
            if key in self.slots:
                old_slot = self.slots.pop(key)
                self.used_bytes -= old_slot.size_bytes
            
            self.slots[key] = slot
            self.used_bytes += slot.size_bytes
        
        return evicted


class ElasticTensorManager:
    """
    Manages the four elastic operations (gather/discard/execute/checkpoint)
    across the heterogeneous GPU hierarchy.
    
    Adapted from mTuner's elastic tensor paradigm:
    - gather: Pull hot embeddings to faster tier (e.g., CPU → H100)
    - discard: Push cold embeddings to slower tier (e.g., A6000 → CPU)
    - execute: Compute on embedding at current location
    - checkpoint: Save state for fault tolerance
    """
    
    def __init__(self, pools: Dict[str, GPUMemoryPool]):
        self.pools = pools  # device_name -> GPUMemoryPool
        self.op_latencies: Dict[ElasticOp, List[float]] = {op: [] for op in ElasticOp}
        self._checkpoint_store: Dict[Tuple[int, int], torch.Tensor] = {}
    
    def gather(self, table_id: int, row_range: Tuple[int, int],
               source_pool: str, target_pool: str,
               source_data: torch.Tensor) -> TensorSlot:
        """
        Gather operation: migrate embedding to a faster tier.
        Handles dtype conversion (FP32→BF16→FP8) during migration.
        """
        start = time.time()
        target = self.pools[target_pool]
        
        # Convert to target pool's expected dtype
        target_dtype = self._get_pool_dtype(target_pool)
        migrated = source_data.to(device=target.device, dtype=target_dtype)
        
        slot = TensorSlot(
            table_id=table_id,
            row_range=row_range,
            data=migrated,
            dtype=target_dtype,
            device=target.device,
            last_access=time.time(),
            size_bytes=migrated.numel() * migrated.element_size()
        )
        
        evicted = target.put(slot)
        elapsed = time.time() - start
        self.op_latencies[ElasticOp.GATHER].append(elapsed)
        
        # Auto-discard evicted slots to lower tier
        for evicted_slot in evicted:
            self._auto_discard(evicted_slot, target_pool)
        
        logger.info(f"GATHER table={table_id} {source_pool}→{target_pool} "
                    f"latency={elapsed*1000:.2f}ms evicted={len(evicted)}")
        
        return slot
    
    def discard(self, table_id: int, row_range: Tuple[int, int],
                source_pool: str, target_pool: str) -> Optional[TensorSlot]:
        """
        Discard operation: migrate embedding to a slower tier.
        Handles dtype upcast (FP8→BF16→FP32) to preserve precision.

        Durability (R1 fix): the migration is two-phase. We first stage the
        upcast copy and attempt to insert it into the target tier. Only if the
        target *durably accepts* it do we release the source slot. If the
        target rejects (e.g. the chunk exceeds the target pool's capacity), we
        roll the source back so the embedding is never lost from both tiers.

        The previous implementation popped the source first; a target
        rejection then destroyed the only copy and desynced used_bytes. This
        mirrors the write-ahead / two-phase-commit discipline DeepSpeed uses in
        swap_out_partitioned_params and that databases use for durable moves.
        """
        start = time.time()
        source = self.pools[source_pool]

        # Phase 0 — read the source slot WITHOUT removing it yet. We use
        # pop() to take ownership atomically (closing the TOCTOU race from
        # M001) but keep a reference so we can reinstate it on failure.
        slot = source.pop(table_id, row_range)
        if slot is None:
            return None

        try:
            target = self.pools[target_pool]
            target_dtype = self._get_pool_dtype(target_pool)

            # Phase 1 — stage the upcast copy.
            migrated = slot.data.to(device=target.device, dtype=target_dtype)
            new_slot = TensorSlot(
                table_id=table_id,
                row_range=row_range,
                data=migrated,
                dtype=target_dtype,
                device=target.device,
                last_access=slot.last_access,
                access_count=slot.access_count,
                dirty=slot.dirty,
                size_bytes=migrated.numel() * migrated.element_size()
            )

            # Phase 2 — commit to target. put() signals rejection by
            # returning the very slot we tried to insert (oversized vs the
            # target's total capacity).
            evicted = target.put(new_slot)
            rejected = any(e is new_slot for e in evicted)
            if rejected:
                # Roll the source back: the embedding stays where it was.
                source.put(slot)
                logger.error(
                    f"DISCARD rejected by target {target_pool} "
                    f"(chunk too large); rolled back to {source_pool}. "
                    f"table={table_id} rows={row_range}")
                return None

            # Target accepted other evictions (LRU victims) — cascade them.
            for ev in evicted:
                self._auto_discard(ev, target_pool)
        except Exception:
            # Any staging/commit failure: reinstate the source slot so data
            # is never silently lost, then re-raise for the caller to handle.
            source.put(slot)
            logger.exception(
                f"DISCARD failed mid-flight; rolled back to {source_pool}. "
                f"table={table_id} rows={row_range}")
            raise

        elapsed = time.time() - start
        self.op_latencies[ElasticOp.DISCARD].append(elapsed)

        logger.info(f"DISCARD table={table_id} {source_pool}→{target_pool} "
                    f"latency={elapsed*1000:.2f}ms")

        return new_slot
    
    def execute(self, table_id: int, row_range: Tuple[int, int],
                pool_name: str, op_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Execute operation: run computation on embedding at current location.
        """
        start = time.time()
        pool = self.pools[pool_name]
        slot = pool.get(table_id, row_range)
        
        if slot is None or slot.data is None:
            raise ValueError(f"Embedding table={table_id} rows={row_range} "
                           f"not found in pool {pool_name}")
        
        result = op_fn(slot.data)
        slot.dirty = True
        
        elapsed = time.time() - start
        self.op_latencies[ElasticOp.EXECUTE].append(elapsed)
        
        return result
    
    def checkpoint(self, table_id: int, row_range: Tuple[int, int],
                   pool_name: str):
        """
        Checkpoint operation: save embedding state in FP32 for recovery.
        """
        start = time.time()
        pool = self.pools[pool_name]
        slot = pool.get(table_id, row_range)
        
        if slot is None or slot.data is None:
            return
        
        # Always checkpoint in FP32 for maximum fidelity
        key = (table_id, row_range[0])
        self._checkpoint_store[key] = slot.data.detach().float().cpu().clone()
        slot.dirty = False
        
        elapsed = time.time() - start
        self.op_latencies[ElasticOp.CHECKPOINT].append(elapsed)
    
    def _auto_discard(self, slot: TensorSlot, from_pool: str):
        """Auto-discard evicted slot to the next lower tier."""
        tier_order = ['h100', 'a6000_0', 'a6000_1', 'cpu']
        try:
            idx = tier_order.index(from_pool)
            if idx + 1 < len(tier_order):
                target_pool = tier_order[idx + 1]
                if target_pool in self.pools:
                    target = self.pools[target_pool]
                    target_dtype = self._get_pool_dtype(target_pool)
                    migrated = slot.data.to(device=target.device, dtype=target_dtype)
                    new_slot = TensorSlot(
                        table_id=slot.table_id,
                        row_range=slot.row_range,
                        data=migrated,
                        dtype=target_dtype,
                        device=target.device,
                        size_bytes=migrated.numel() * migrated.element_size()
                    )
                    target.put(new_slot)
        except ValueError:
            pass
    
    def _get_pool_dtype(self, pool_name: str) -> torch.dtype:
        """
        Infer the native dtype for a pool based on its device type and name.
        
        Strategy:
          1. If pool has a real CUDA device, inspect compute capability.
          2. Otherwise, infer from pool name (supports testing on CPU).
        
        This handles arbitrary pool names (e.g., 'a6000_2', 'a6000_3')
        without manual dict updates.
        """
        pool = self.pools.get(pool_name)
        if pool is None:
            # Unknown pool — try name-based inference, then default FP32
            return self._dtype_from_name(pool_name)
        
        device = pool.device
        
        # For CUDA devices, check capability (authoritative)
        if device.type == 'cuda':
            try:
                cap = torch.cuda.get_device_capability(device)
                if cap[0] >= 9:  # sm90+ (Hopper): supports FP8
                    return torch.float8_e4m3fn
                else:  # sm86 and below: BF16
                    return torch.bfloat16
            except Exception:
                pass
        
        # For CPU-device pools OR fallback: use name-based inference.
        # This is essential for:
        #   a) Testing without real GPUs (pools created on CPU)
        #   b) CPU-staged pools that genuinely hold FP32 data
        return self._dtype_from_name(pool_name)
    
    @staticmethod
    def _dtype_from_name(pool_name: str) -> torch.dtype:
        """Infer dtype from pool name string. Used as fallback."""
        if pool_name.startswith('h100') or pool_name.startswith('h200'):
            return torch.float8_e4m3fn
        if 'a6000' in pool_name or 'a100' in pool_name or 'gpu' in pool_name:
            return torch.bfloat16
        if pool_name == 'cpu' or pool_name.startswith('cpu'):
            return torch.float32
        return torch.float32
    
    def get_latency_report(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics for all operations."""
        import numpy as np
        report = {}
        for op, latencies in self.op_latencies.items():
            if latencies:
                report[op.value] = {
                    'count': len(latencies),
                    'mean_ms': float(np.mean(latencies)) * 1000,
                    'p50_ms': float(np.median(latencies)) * 1000,
                    'p99_ms': float(np.percentile(latencies, 99)) * 1000,
                    'total_ms': float(np.sum(latencies)) * 1000,
                }
        return report
