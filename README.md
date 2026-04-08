## Optimization Attempts

### Logging and Profiling

**Targeted issue:** Limited visibility into where end-to-end latency was being spent, especially during synchronization-heavy decode steps.

Runtime instrumentation was expanded to improve diagnosis and benchmarking across configurations. Added measurements included compute time, synchronization time, wait time, per-segment timing, per-thread blocking behavior, network skew, and structured summaries for slow compute and synchronization paths. End-of-run metrics were also aggregated to support consistent comparison of throughput, latency, and total runtime.

Profiling showed that the dominant bottleneck was not local layer computation, but synchronization overhead. The largest latency spikes occurred in the synchronization path, particularly in a late-stage segment around `seg[65]`, where worker threads spent substantial time blocked waiting at synchronization barriers. Root-side merge computation was not the primary limiter; instead, end-to-end latency was driven mainly by imbalance and waiting during inter-thread and inter-node synchronization.

### Code Change Attempts

**Targeted issue:** Excessive thread-management overhead and synchronization cost during repeated forward passes.

A deeper execution-model refactor was attempted to reduce thread-management and synchronization overhead. This replaced the existing per-forward threading behavior with a persistent executor-thread design using a condition-variable-based barrier. The change was not retained, as incorrect barrier participation accounting introduced deadlock and increased wait overhead under distributed execution. 

Another refactor was attempted to separate the orchestration and compute in the root node to to prevent the slowing down of root node's compute. First attempt was to have a completely separate orchestration node that is not involved in compute whatsoever, which failed due to significant need of refactoring. The second less aggressive attempt was to separate the two operations into different threads so the efficiency cores of our root node could work on the orchestration while performance cores would focus soley on compute like worker nodes. This again failed due to the significant dependence of the original codebase in the existing method of threading.

### Communication Changes

**Targeted issue:** Communication overhead from fixed small socket transfer granularity in the send/read path.

The fixed socket transfer granularity was replaced with a configurable network chunk size from Makefile. Multiple chunk sizes were evaluated, with 256 KB selected as the best-performing setting in the tested configurations. This change was retained.

### Build and Makefile Changes

**Targeted issue:** Suboptimal local compute performance on the target ARM hardware.

More aggressive hardware-targeted compiler settings were evaluated to improve local compute performance. The final selected build configuration was:

```make
-mcpu=cortex-a76+fp16+dotprod -Ofast -flto -fomit-frame-pointer
```

This configuration delivered the best overall result among the tested builds and was adopted as the preferred compilation target for the optimized branch.

### Stability Adjustments

**Targeted issue:** Runtime instability on the root node when using dot-product acceleration.

To ensure stable execution with hardware dot-product support enabled, the root node CPU frequency was capped at 1992 MHz after trial and error testing.

### Configuration Sweeps

As one of the final stages of optimization, controlled sweeps were run across key runtime parameters to identify stable and high-performing operating points. These sweeps covered:

- `nthreads`
- `max_seq_len`
- batch size
- network chunk size
- prompt length

These experiments helped establish the preferred deployment regime for the current codebase and confirmed that synchronization overhead remained the dominant performance constraint across workload shapes.
