# Claude #1 — NeurIPS Review: X-axis数据设计 + 大厂Infra函数级映射 + 38-Claude开发计划

## Part I: Demo Data X-axis维度分析

从 `data.zip` (commit 94b7c6 / 7f92b5) 提取的四份数据文件，X-axis维度如下：

| 文件 | X-axis | 长度 | 范围 | 间距 | Y-axis | 方法数 | Seed数 |
|------|--------|------|------|------|--------|--------|--------|
| `reversed_figure_data.json` | Sequential Steps | 3000 | [0, 3000) | 1.0 | Perplexity | 5 (DDP, DES-LOC, Local Adam, FAVG+OPT, FAVG-OPT) | 3 |
| `gradient_norm_24k_data.json` | Sequential Steps | 2000 | [0.0, 40960.0] | 20.49 | Gradient Norm (L2) | 4 (AdamW-DDP, DES-LOC-Nesterov, Local Adam, DiLoCo) | 3 |
| `ppl_vs_time_1B_30k_data.json` | Wall Time (hours) | 2000 | [0.0, 17.1] | 0.00855 | Perplexity | 5 (DDP, DES-LOC, Local Adam, FAVG+OPT, FAVG-OPT) | 3 |
| `reversed_figure18_data.json` | Sequential Steps (1-20000) | ~2000 | [1, 20000] | ~10 | Momentum/Parameter Norm (L2) | 多panel | 3 |

### 数据模式总结

每份数据共享统一schema:
```json
{
  "metadata": {"panel": "...", "source": "...", "n_per_seed": 2000, "n_seeds": 3},
  "methods": {
    "MethodName": {
      "seed_0": [2000 floats],
      "seed_1": [2000 floats],
      "seed_2": [2000 floats],
      "mean": [2000 floats],
      "std": [2000 floats],
      "reported_final": "value±std"
    }
  }
}
```

### 我们需要生成的X-axis维度数据

参照demo格式，MPAlloy/CASH实验需要产出如下X-axis维度的数据：

| 实验 | X-axis | 预期长度 | 范围 | Methods (对标demo) | 度量 |
|------|--------|---------|------|-------------------|------|
| **Exp1: Tiered Placement** | Training Steps | 2000 | [0, 40000] | Uniform, Tiered-Alloy, Oracle | Throughput (samples/s) |
| **Exp2: Mixed Precision Convergence** | Training Steps | 2000 | [0, 40000] | FP32-only, Mixed-FP8/BF16/FP32, BF16-only | Loss / Perplexity |
| **Exp2b: Drift Over Time** | Training Steps | 2000 | [0, 40000] | Boundary, Cancellation, Accumulation | Relative Drift |
| **Exp3: Migration Latency** | Transfer Size (MB) | 20 | [1, 1024] | CPU→H100, CPU→A6000, H100→A6000, A6000→CPU | Bandwidth (GB/s) |
| **Exp3b: Elastic Ops** | Chunk Size (rows) | 10 | [1024, 262144] | Gather, Discard, Execute, Checkpoint | Latency (ms) |
| **CASH-Exp1: Differential Fuzzing** | Fuzz Configs | 500 | [1, 500] | indexGet, indexPut, indexCopy | Divergence Count (cumulative) |
| **CASH-Exp3: Pareto Frontier** | Tolerance (ULP) | 8 | [0, 128] | sm86-strict, sm90-relaxed, sm90-strict | Throughput (Gops/s) |

---

## Part II: 大厂Infra函数级映射 — "从好例子开始"

### Pattern 1: 从 NCCL `ncclFuncAllReduce` + CUB `DoubleBuffer` 开始

**NCCL 真实代码** (grep自 `infra_refs/nccl/src/graph/tuning.cc`):
```cpp
// tuning.cc:276
int nsteps = coll == ncclFuncAllReduce ? 2*(nRanks-1) : ...

// tuning.cc:299-300
// AllReduce pipelines two operations.
if (coll == ncclFuncAllReduce) {
    busBw = std::min(busBw*.92, graphs[a]->nChannels*perChMaxTreeBw);
}

// tuning.cc:601
if (algorithm == NCCL_ALGO_TREE && coll == ncclFuncAllReduce && logSize >= 0 && logSize < 23)
    bw *= treeCorrectionFactor[protocol][logSize];
```

**CUB 真实代码** (grep自 `infra_refs/cccl/cub/test/catch2_radix_sort_helper.cuh`):
```cpp
// catch2_radix_sort_helper.cuh:40-41
// pass references to a __global__ function. The DoubleBuffer object
cub::DoubleBuffer<KeyType> d_keys(...);

// catch2_test_device_topk_common.cuh:338
cub::DoubleBuffer<KeyT> d_keys(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()));
```

然后，遵循该模式实现一个新的 **`CrossPrecisionAllReduce`** (MPAlloy `alloy/mixed_precision/gradient_verifier.py:47-78`):

```python
class CrossPrecisionAllReduce:
    """
    Performs gradient allreduce across devices with different precisions.
    H100 computes in FP8, A6000 in BF16, CPU in FP32.
    Aggregation is done in FP32 to minimize numerical drift.
    """
    def __init__(self, precision_configs: List[PrecisionConfig]):
        self.configs = {pc.device: pc for pc in precision_configs}
        self._drift_history: List[float] = []

    def allreduce_gradients(self, gradients, reference_device=None):
        # Step 1: Upcast all gradients to FP32
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
            result[device] = mean_grad.to(device=device, dtype=config.compute_dtype)
        return result
```

让 **tiered embedding gradients** 可以 **在 FP8/BF16/FP32 三精度路径之间一致聚合**，并能 **通过三阶段管线 `GatherUpcastKernel` → FP32 sum → `ScatterDowncastKernel`** (CUDA: `alloy/include/alloy/mixed_precision_kernels.cuh:110-136`) **保持数值稳定**:

```cuda
// mixed_precision_kernels.cuh:110-122
template <typename TierType>
__global__ void GatherUpcastKernel(
    const TierType* __restrict__ d_tier_grad,
    float*          __restrict__ d_accum,
    size_t                       num_elements)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        atomicAdd(d_accum + i, static_cast<float>(d_tier_grad[i]));
    }
}
```

### Pattern 2: 接着 `ClassifyAndScatterKernel` 引入 CUB三lambda分发

**CUB DeviceTopK原始模式**: `f_early_stop` / `f_with_out_buf` / `f_no_out_buf` — 根据运行时状态选择不同的处理lambda。

**Alloy实现** (`alloy/include/alloy/tiered_cache.cuh:148-214`):
```cuda
// 三lambda: f_in_place / f_serve_enqueue / f_serve_only
// f_in_place: 已在正确tier → 只serve
auto f_in_place = [&](size_t row, size_t out_idx) {
    const Tier current = static_cast<Tier>(d_row_tier[row]);
    gather_row(row, out_idx, current);
};

// f_serve_enqueue: 在错误tier + migration queue有容量 → serve + enqueue
auto f_serve_enqueue = [&](size_t row, size_t out_idx) {
    const float freq = static_cast<float>(d_freq_hist[row]);
    const Tier target = classify_row(freq, thresholds);
    const Tier current = static_cast<Tier>(d_row_tier[row]);
    gather_row(row, out_idx, current);
    if (target != current) {
        const uint32_t slot = atomicAdd(d_migration_count, 1u);
        d_migration_queue[slot] = {row, current, target};
    }
};

// f_serve_only: 在错误tier + budget耗尽 → 只serve不迁移
auto f_serve_only = [&](size_t row, size_t out_idx) {
    const Tier current = static_cast<Tier>(d_row_tier[row]);
    gather_row(row, out_idx, current);
};
```

使 **lookup阶段** 能够 **同时完成频率直方图统计、tier分类和migration入队**，同时 **`MigrationQueueDoubleBuffer`** (`tiered_cache.cuh:120-138`) 优化 **producer/consumer解耦**:
```cuda
struct MigrationQueueDoubleBuffer {
    MigrationEntry* bufs[2];
    uint32_t*       counts[2];
    int             selector;
    __host__ __device__ MigrationEntry* Current()   { return bufs[selector]; }
    __host__ __device__ MigrationEntry* Alternate() { return bufs[selector ^ 1]; }
};
```

### Pattern 3: 随后 `AsyncPipeline` 整合 CUDA多流事件同步

**类比 Megatron-LM** (grep自 `infra_refs/Megatron-LM/megatron/training/checkpointing.py:495`):
```python
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...)
```
**类比 PyTorch FSDP** (grep自 `infra_refs/pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_api.py:14`):
```python
class MixedPrecisionPolicy:  # param_dtype, reduce_dtype, output_dtype
```

**Alloy 三流管线** (`alloy/include/alloy/async_pipeline.cuh:62-150`):
```cuda
class AsyncPipeline {
    // Stream 0: compute (forward + backward + optim)
    // Stream 1: analytics (histogram + classify)
    // Stream 2: migration (gather + discard + checkpoint)
    void step(KernelCallback fn_histogram, KernelCallback fn_classify,
              KernelCallback fn_forward_bwd, KernelCallback fn_migration, void* user_data)
    {
        // Wait for previous migration
        if (step_ > 0)
            cudaStreamWaitEvent(s_compute, prev.migration_done);
        // Phase 1: Analytics
        fn_histogram(s_analytics, user_data);
        cudaEventRecord(cur.histogram_done, s_analytics);
        // Phase 2: Compute (waits on classify)
        cudaStreamWaitEvent(s_compute, cur.classify_done);
        fn_forward_bwd(s_compute, user_data);
        // Phase 3: Migration (waits on compute)
        cudaStreamWaitEvent(s_migration, cur.compute_done);
        fn_migration(s_migration, user_data);
    }
};
```

令 **三路流水线(compute/analytics/migration)** 支持 **跨step重叠执行**，进而 **`MultiDevicePipeline`** (`async_pipeline.cuh:173-225`) 增强 **P2P peer access + 全局barrier**:
```cuda
struct MultiDevicePipeline {
    AsyncPipeline pipelines[MAX_DEVICES];
    void barrier_allreduce() {
        for (int i = 0; i < num_devices; ++i)
            cudaEventRecord(allreduce_ready[i], pipelines[i].stream(COMPUTE));
        for (int i = 0; i < num_devices; ++i)
            for (int j = 0; j < num_devices; ++j)
                if (i != j)
                    cudaStreamWaitEvent(pipelines[i].stream(COMPUTE), allreduce_ready[j]);
    }
};
```

### Pattern 4: 最终 `CrucibleRuntime` 完善 cross-architecture验证

**类比 OpenAI/Triton** (grep自 `infra_refs/triton/python/triton/runtime/autotuner.py:19`):
```python
class Autotuner(KernelInterface):  # auto-tune kernel configs
```
**类比 vLLM eviction** (grep自 `infra_refs/vllm/vllm/v1/core/block_pool.py:435`):
```python
def evict_blocks(self, block_ids: set[int]) -> None:
```

**Crucible实现** (`CASH/crucible/src/crucible_runtime.cu:104-140`):
```cuda
int fuzz_round(size_t max_configs = 100) {
    // Generate boundary configs via BoundaryTestSuite
    std::vector<IndexKernelTestConfig> configs(max_configs);
    const size_t num_configs = BoundaryTestSuite::generate(configs.data(), max_configs);
    // Three kernels: indexGet, indexPut, indexCopy
    for (const auto& kern : kernels) {
        for (size_t ci = 0; ci < num_configs; ++ci) {
            DivergenceReport report =
                IndexKernelHarness::differential_test<float>(dev_a, dev_b, kern.type, cfg);
            influence_.update("shape_rows", diverged);
        }
    }
}
```

确保 **sm86(A6000) ↔ sm90(H100) kernel行为** 兼容 **所有Alloy依赖的indexGet/indexPut/indexCopy内核**，全面 **`InfluenceGuide`** (`CASH/crucible/fuzzer/kernel_fuzzer.py:67-107`) 升级 **boundary-aware fuzzing覆盖**:
```python
class InfluenceGuide:
    def update(self, config: KernelConfig, caused_divergence: bool):
        if caused_divergence:
            for key, val in params.items():
                if self._is_boundary_value(key, val):
                    self.param_scores[key] = min(1.0, self.param_scores[key] + 0.1)
    def _is_boundary_value(self, param_name, value):
        boundaries = {
            'shared_mem': [0, 1024, 4096, 16384, 49152, 65536, 100352, 163840, 227328],
            'block_x': [32, 64, 128, 256, 512, 1024],
            ...
        }
```

以达成 **跨架构bitwise/relaxed一致性的Pareto最优** (`CASH/crucible/src/crucible_runtime.cu:145-240`, `sweep_pareto()`):
```cuda
std::vector<ParetoPoint> sweep_pareto(...) {
    for (int lvl = 0; lvl < ToleranceLevel::NUM_PRESETS; ++lvl) {
        if (tol.require_bitwise)
            StrictLookupKernel<float><<<batch, 32>>>(d_table_b, d_idx_b, d_out_b, batch, dim);
        else
            RelaxedLookupKernel<float, 256, 8><<<(batch+7)/8, dim3(32,8)>>>(d_table_b, d_idx_b, d_out_b, batch, dim);
        ParetoVerifyKernel<<<...>>>(d_out_a, d_out_b_copy, d_max_ulp, d_max_rel, out_elems);
    }
}
```

### 其余15个大厂仓库的关键函数映射

| # | 仓库 | grep得到的真实函数 | Alloy/CASH对应 |
|---|------|-------------------|---------------|
| 5 | **TransformerEngine** | `fake_quantize(tensor, fp8_format, out=None)` (fake_quant.py:25), `DisableFP8Layer` | `FPRevDiagnosticBatch.generate_distinguishing_inputs()` |
| 6 | **CUTLASS** | `MixedInputUtils` (mixed_input_utils.hpp:539), `Sm100MixedInputBlockwiseScaleConfig` (sm100_mixed_dtype_blockwise_layout.hpp:49) | `GatherUpcastKernel`/`ScatterDowncastKernel` 精度转换 |
| 7 | **DeepSpeed** | `ZeROOptimizer` (base_optimizer.py:235), `partition_grads()` (superoffload_stage3.py:134), `swap_out_partitioned_params()` (partitioned_param_swapper.py:401) | `ElasticTensorManager.discard()`, SSD tier归档 |
| 8 | **vLLM** | `evict_blocks()` (block_pool.py:435), `_allocate_blocks()` (manager.py:69), ARC policy (arc.py:104), LRU policy (lru.py:34) | `GPUMemoryPool.put()` LRU eviction |
| 9 | **PyTorch FSDP** | `MixedPrecisionPolicy` (_fsdp_api.py:14), `all_reduce()` (distributed_c10d.py:3156), `set_all_reduce_hook()` (_fully_shard.py:591) | 跨精度allreduce管线 |
| 10 | **FasterTransformer** | Multi-GPU pipeline parallelism CUDA kernels | `AsyncPipeline` 三流overlap |
| 11 | **JAX** | `jax.distributed` sharding API | `TieredPlacementScheduler.compute_placement_plan()` |
| 12 | **Triton** | `JITFunction` (jit.py:622), `Autotuner` (autotuner.py:19) | `VectorizedLaunchConfig.compute()` |
| 13 | **Ray** | `PlacementGroup` (placement_group.py:22), `PlacementGroupSchedulingStrategy` (scheduling_strategies.py:17) | `LoadAwarePartitioner.partition_dynamic()` |
| 14 | **Accelerate** | `FP8RecipeKwargs` (dataclasses.py:456), `AcceleratorState` (state.py:871), `set_mixed_precision()` (dataclasses.py:1412) | Device-tier state管理 |
| 15 | **FlashInfer** | `cudnn_batch_prefill_with_kv_cache()` (prefill.py:563), `cudnn_batch_decode_with_kv_cache()` (decode.py:258) | Batched embedding lookup模式 |
| 16 | **xformers** | `BlockSparseAttentionWrapper` (sparse.py:69) | Vectorized memory access |
| 17 | **FairScale** | Fully sharded data parallel | Tier-aware gradient partitioning |
| 18 | **ONNX Runtime** | Graph optimization, mixed precision execution | Compute graph scheduling |
| 19 | **MaxText** | TPU mesh sharding | Multi-tier device assignment |
| 20 | **TensorRT** | FP8 calibration, quantization-aware inference | `MeasureDriftKernel` drift验证 |

---

## Part III: 38-Claude开发计划

| Claude # | Milestones | 具体任务 | 产出数据X-axis |
|----------|-----------|---------|---------------|
| **#1 (当前)** | M001–M003 | `ElasticTensorManager` race condition修复 + capacity guard + dtype推断 | 回归测试 pass/fail |
| **#2** | M004–M006 | `CrossPrecisionAllReduce` 加权平均 + NaN/Inf sentinel + error bound | Drift vs Steps (2000点) |
| **#3** | M007–M009 | `TieredPlacementScheduler` hysteresis防乒乓 + migration cooldown + frequency histogram优化 | Migration count vs Steps |
| **#4** | M010–M012 | `CrucibleFuzzer` FP8 dtype fuzzing + `InfluenceGuide` Bayesian采样 + config space覆盖 | Divergence vs Fuzz configs (500点) |
| **#5** | M013–M015 | `tiered_embedding.cu` CUDA error checking + stream同步guard + kernel launch bounds | Error rate vs Steps |
| **#6** | M016–M018 | `AsyncPipeline` profiling hooks + `MultiDevicePipeline` 错误恢复 + event timing | Pipeline utilization vs Steps |
| **#7** | M019–M021 | 实验reproducibility: seed固定 + deterministic mode + checkpoint/resume | Variance across seeds |
| **#8** | M022–M024 | `LoadAwarePartitioner` 通信开销建模 + PCIe latency × data volume + dynamic rebalance | Partition ratio vs workload |
| **#9** | M025–M027 | Exp1 Criteo/Avazu真实数据集loader + feature preprocessing + Zipf validation | Throughput vs Steps (2000点, 3 seeds) |
| **#10** | M028–M030 | `VectorizedTieredGetKernel` warp-shuffle优化 + shared memory staging + L2 cache hint | Bandwidth vs embedding_dim |
| **#11** | M031–M033 | FP8 E5M2 backward support + forward E4M3 / backward E5M2 split + dynamic loss scaling | Loss vs Steps (E4M3 vs E5M2) |
| **#12** | M034–M036 | `CheckpointCompactKernel` CRC32校验 + incremental snapshot + async IO | Checkpoint latency vs dirty ratio |
| **#13** | M037–M039 | Cross-node NCCL integration + multi-node allreduce + ring/tree topology | Allreduce latency vs node count |
| **#14** | M040–M042 | `BitwiseConsistencyChecker` ULP距离量化 + histogram binning + statistical testing | ULP histogram (64 bins) |
| **#15** | M043–M045 | SSD tier实现: mmap + direct I/O + prefetch heuristic | SSD bandwidth vs transfer size |
| **#16** | M046–M048 | `AccessFrequencyTracker` count-min sketch替换 + 内存从O(n)降到O(1) | Memory usage vs table size |
| **#17** | M049–M051 | Pareto frontier可视化 + LaTeX figure生成 + error bar计算 | Throughput vs Tolerance (8点, 3 seeds) |
| **#18** | M052–M054 | CI/CD: multi-arch compilation test + sm86/sm90 cross-compile + binary size | Build time vs config |
| **#19** | M055–M057 | Per-tier gradient clipping + FP8 overflow protection + dynamic scale factor | Clip ratio vs Steps |
| **#20** | M058–M060 | `MigrationBatchConfig` 运行时PCIe bandwidth calibration + adaptive budget | Actual vs predicted bandwidth |
| **#21** | M061–M063 | 嵌入表dynamic resizing + online feature expansion + hot-add rows | Table size vs Steps |
| **#22** | M064–M066 | `DeviceCalibrator` memory bandwidth micro-benchmark + roofline model | Roofline: ops/byte vs FLOPS |
| **#23** | M067–M069 | 多租户embedding serving + tier isolation + fair scheduling | Per-tenant throughput |
| **#24** | M070–M072 | Async checkpoint to NFS/S3 + compression + dedup | Checkpoint size vs compression |
| **#25** | M073–M075 | `TieredSGDKernel` → Adam/LAMB + per-tier momentum + second moment tracking | Convergence: Adam vs SGD vs LAMB |
| **#26** | M076–M078 | Exp2 convergence统计显著性检验 + confidence interval + Wilcoxon test | P-value vs sample size |
| **#27** | M079–M081 | CUDA Graph capture + steady-state optimization + graph replay | Kernel launch overhead reduction |
| **#28** | M082–M084 | Fault injection: GPU OOM / PCIe error / NaN injection + recovery | Recovery time vs fault type |
| **#29** | M085–M087 | gRPC serving endpoint + batched inference + embedding cache | Serving latency vs batch size |
| **#30** | M088–M090 | `FrequencyHistogramKernel` shared memory histogram + warp-level reduction | Histogram kernel time vs batch |
| **#31** | M091–M093 | Multi-table fusion: batch across tables + shared index buffer | Throughput vs num_tables |
| **#32** | M094–M096 | Dynamic learning rate per-tier + warmup schedule + cosine decay | LR schedule vs Steps |
| **#33** | M097–M099 | `MeasureDriftKernel` warp-cooperative reduction替换block-level | Drift measurement overhead |
| **#34** | M100–M102 | Feature hashing + collision resolution for cold tier + MurmurHash3 | Collision rate vs hash table size |
| **#35** | M103–M105 | Telemetry: Prometheus + Grafana + per-tier metric export | Dashboard JSON schema |
| **#36** | M106–M108 | Paper figure reproduction scripts + matplotlib + pgfplots | All 4 figure types |
| **#37** | M109–M111 | Ablation study automation + config sweep + result aggregation | Ablation table (8×4) |
| **#38** | M112–M114 | End-to-end integration test suite + regression CI + performance gates | Test matrix (all experiments) |

---

## Part IV: Claude #1 已完成的M001-M003修改

**文件**: `MPAlloy/alloy/elastic_tensor/manager.py`

### M001: `discard()` TOCTOU Race Condition
- 新增 `GPUMemoryPool.pop()` 原子方法
- `discard()` 改用 `pop()` 替代 `get()` + 手动 `del`
- 消除并发discard的double-free风险

### M002: `put()` Capacity Overflow Guard
- 超大slot被拒绝（返回为evicted而非静默插入）
- 同key重复插入时先移除旧entry再插入（防止used_bytes double-count）

### M003: `_get_pool_dtype()` 动态推断
- CUDA设备: 通过 `get_device_capability()` 自动判断 sm90+ → FP8, 其他 → BF16
- 非CUDA设备: 通过pool name substring匹配（支持a6000_2, gpu_custom等）
- 消除硬编码dict的O(n)维护问题

**测试**: 5/5 通过 (pop atomicity, discard no-race, capacity overflow, duplicate key, dynamic dtype)

**实验脚本**: `experiments/run_hetero_integration.py` (978行) + `scripts/run_hetero_test.sh`
- 自动发现GPU拓扑 (A6000×2 + H100 NVL)
- NUMA node 1 pinning
- 4个实验模块: regression, exp1, exp2, exp3
- 输出格式与demo data对齐 (2000点, 3 seeds, mean/std)
