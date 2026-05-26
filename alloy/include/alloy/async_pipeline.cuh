#pragma once

#ifndef ALLOY_ASYNC_PIPELINE_CUH
#define ALLOY_ASYNC_PIPELINE_CUH

// Alloy Asynchronous Training Pipeline
//
// The single-stream execution in tiered_embedding.cu serialises five
// phases per training step:
//
//   decay → histogram → classify+scatter → training → migration
//
// This is wasteful: the histogram of batch N has no dependency on
// the migration of batch N-1, and the migration of batch N has no
// dependency on the training of batch N+1's forward pass (different
// embedding rows in steady state).
//
// This module implements a three-stream pipeline:
//
//   Stream 0 (compute):   forward → backward → optimizer step
//   Stream 1 (analytics): decay → histogram → classify
//   Stream 2 (migration): gather/discard → checkpoint
//
// CUDA events enforce the true data dependencies:
//
//   histogram[N] must complete before classify[N] starts
//   classify[N]  must complete before forward[N] starts
//   forward[N]   must complete before migration[N] starts
//   migration[N] must complete before forward[N+1] reads migrated rows
//
// This gives us:
//
//   Time →  ┃ Step N              ┃ Step N+1            ┃
//   ────────┃─────────────────────┃─────────────────────┃
//   Stream0 ┃  forward  backward  ┃  forward  backward  ┃
//   Stream1 ┃  hist classify      ┃  hist classify      ┃
//   Stream2 ┃     migration       ┃     migration       ┃
//           ┃     └── overlap ──┘ ┃     └── overlap ──┘ ┃

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace Alloy {

// ─────────────────────────────────────────────────────────────────
//  Stream roles
// ─────────────────────────────────────────────────────────────────

enum class StreamRole : int {
    COMPUTE    = 0,   // forward + backward + optim
    ANALYTICS  = 1,   // histogram + classify
    MIGRATION  = 2,   // gather + discard + checkpoint
    NUM_STREAMS = 3
};

// ─────────────────────────────────────────────────────────────────
//  Pipeline event graph
//
//  Each step produces 4 events; the next step's kernels wait on
//  the relevant events before launching.
// ─────────────────────────────────────────────────────────────────

struct PipelineEvents {
    cudaEvent_t histogram_done;     // analytics stream signals histogram complete
    cudaEvent_t classify_done;      // analytics stream signals classify complete
    cudaEvent_t compute_done;       // compute stream signals fwd+bwd complete
    cudaEvent_t migration_done;     // migration stream signals gather/discard done

    static PipelineEvents create()
    {
        PipelineEvents ev;
        cudaEventCreateWithFlags(&ev.histogram_done,  cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev.classify_done,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev.compute_done,    cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev.migration_done,  cudaEventDisableTiming);
        return ev;
    }

    void destroy()
    {
        cudaEventDestroy(histogram_done);
        cudaEventDestroy(classify_done);
        cudaEventDestroy(compute_done);
        cudaEventDestroy(migration_done);
    }
};

// ─────────────────────────────────────────────────────────────────
//  AsyncPipeline
//
//  Owns the three streams and the double-buffered event sets.
//  The caller registers kernel launch callbacks for each phase;
//  AsyncPipeline handles the event synchronisation.
// ─────────────────────────────────────────────────────────────────

class AsyncPipeline {
public:
    using KernelCallback = void (*)(cudaStream_t stream, void* user_data);

    AsyncPipeline()
        : step_(0)
        , current_ev_(0)
    {
        for (int i = 0; i < static_cast<int>(StreamRole::NUM_STREAMS); ++i)
            cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);

        events_[0] = PipelineEvents::create();
        events_[1] = PipelineEvents::create();
    }

    ~AsyncPipeline()
    {
        for (int i = 0; i < static_cast<int>(StreamRole::NUM_STREAMS); ++i)
            cudaStreamDestroy(streams_[i]);

        events_[0].destroy();
        events_[1].destroy();
    }

    cudaStream_t stream(StreamRole role) const
    {
        return streams_[static_cast<int>(role)];
    }

    // Execute one pipeline step.
    //
    // @param fn_histogram   Launches decay + histogram on analytics stream
    // @param fn_classify    Launches classify+scatter on analytics stream
    // @param fn_forward_bwd Launches forward + backward on compute stream
    // @param fn_migration   Launches gather/discard on migration stream
    // @param user_data      Passed to all callbacks
    void step(
        KernelCallback fn_histogram,
        KernelCallback fn_classify,
        KernelCallback fn_forward_bwd,
        KernelCallback fn_migration,
        void*          user_data)
    {
        PipelineEvents& cur  = events_[current_ev_];
        PipelineEvents& prev = events_[current_ev_ ^ 1];

        cudaStream_t s_compute   = streams_[0];
        cudaStream_t s_analytics = streams_[1];
        cudaStream_t s_migration = streams_[2];

        // ── Wait for previous step's migration to finish ──
        // (so we don't read stale tier buffers in forward pass)
        if (step_ > 0)
        {
            cudaStreamWaitEvent(s_compute,   prev.migration_done);
            cudaStreamWaitEvent(s_analytics, prev.migration_done);
        }

        // ── Phase 1: Analytics (histogram + classify) ──
        fn_histogram(s_analytics, user_data);
        cudaEventRecord(cur.histogram_done, s_analytics);

        // Classify depends on histogram
        cudaStreamWaitEvent(s_analytics, cur.histogram_done);
        fn_classify(s_analytics, user_data);
        cudaEventRecord(cur.classify_done, s_analytics);

        // ── Phase 2: Compute (forward + backward) ──
        // Wait for classify to know tier assignments
        cudaStreamWaitEvent(s_compute, cur.classify_done);
        fn_forward_bwd(s_compute, user_data);
        cudaEventRecord(cur.compute_done, s_compute);

        // ── Phase 3: Migration ──
        // Wait for compute to finish (migration reads dirty flags)
        cudaStreamWaitEvent(s_migration, cur.compute_done);
        fn_migration(s_migration, user_data);
        cudaEventRecord(cur.migration_done, s_migration);

        // Flip double-buffer
        current_ev_ ^= 1;
        ++step_;
    }

    // Wait for all streams to drain.
    void synchronize()
    {
        for (int i = 0; i < static_cast<int>(StreamRole::NUM_STREAMS); ++i)
            cudaStreamSynchronize(streams_[i]);
    }

    size_t step_count() const { return step_; }

private:
    cudaStream_t    streams_[3];
    PipelineEvents  events_[2];    // double-buffered
    size_t          step_;
    int             current_ev_;   // 0 or 1
};

// ─────────────────────────────────────────────────────────────────
//  Multi-device stream manager
//
//  For the full A6000×2 + H100×1 cluster, each device has its own
//  AsyncPipeline.  Cross-device synchronisation (for allreduce)
//  uses peer-to-peer events.
// ─────────────────────────────────────────────────────────────────

struct MultiDevicePipeline {
    static constexpr int MAX_DEVICES = 4;

    AsyncPipeline  pipelines[MAX_DEVICES];
    cudaEvent_t    allreduce_ready[MAX_DEVICES];  // device i is ready for allreduce
    cudaEvent_t    allreduce_done[MAX_DEVICES];   // allreduce complete on device i
    int            num_devices;

    void init(int n_devices)
    {
        num_devices = n_devices;
        for (int i = 0; i < n_devices; ++i)
        {
            cudaSetDevice(i);
            cudaEventCreateWithFlags(&allreduce_ready[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&allreduce_done[i],  cudaEventDisableTiming);
        }

        // Enable P2P access between all device pairs
        for (int i = 0; i < n_devices; ++i)
        {
            cudaSetDevice(i);
            for (int j = 0; j < n_devices; ++j)
            {
                if (i == j) continue;
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access)
                    cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    void destroy()
    {
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(i);
            cudaEventDestroy(allreduce_ready[i]);
            cudaEventDestroy(allreduce_done[i]);
        }
    }

    // Barrier: all devices signal ready, then all wait for everyone.
    // Used before and after NCCL allreduce.
    void barrier_allreduce()
    {
        // Each device records its ready event
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(i);
            cudaEventRecord(allreduce_ready[i],
                            pipelines[i].stream(StreamRole::COMPUTE));
        }

        // Each device waits for all others
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(i);
            for (int j = 0; j < num_devices; ++j)
            {
                if (i == j) continue;
                cudaStreamWaitEvent(
                    pipelines[i].stream(StreamRole::COMPUTE),
                    allreduce_ready[j]);
            }
        }
    }
};

}  // namespace Alloy

#endif  // ALLOY_ASYNC_PIPELINE_CUH
