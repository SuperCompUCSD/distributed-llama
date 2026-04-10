## Optimization Attempts

### Thread Tuning

**Targeted issue:** Significant ~3000 ms synchronization spikes during runs at maximum thread count, likely caused by the large compute-performance gap between performance and efficiency cores on the root node.

The thread count was reduced to 4 so execution remained on the performance cores. This eliminated the large synchronization spikes and became the preferred runtime setting for the tested configuration.

### Hardware Accelerators

Vulkan driver and SDK support were installed and tested with a DLLAMA Vulkan build, but this path was rolled back after observing limited practical benefit from the Rock 5B GPU’s compute capability. NPU-based acceleration was also considered, but was not pursued further due to the scale of system-level and codebase changes required.

### Logging and Profiling

**Targeted issue:** Limited visibility into where end-to-end latency was being spent, especially during synchronization-heavy decode steps.

Runtime instrumentation was expanded to improve diagnosis and benchmarking across configurations. Added measurements included compute time, synchronization time, wait time, per-segment timing, per-thread blocking behavior, network skew, and structured summaries for slow compute and synchronization paths. End-of-run metrics were also aggregated to support consistent comparison of throughput, latency, and total runtime.

Profiling showed that the dominant bottleneck was not local layer computation, but synchronization overhead. The largest latency spikes occurred in the synchronization path, particularly in a late-stage segment around `seg[65]`, where worker threads spent substantial time blocked at synchronization barriers. Root-side merge computation was not the primary limiter; instead, end-to-end latency was driven mainly by imbalance and waiting during inter-thread and inter-node synchronization.

### Code Change Attempts

**Targeted issue:** Excessive thread-management overhead and synchronization cost during repeated forward passes.

A deeper execution-model refactor was attempted to reduce thread-management and synchronization overhead. This replaced the existing per-forward threading behavior with a persistent executor-thread design using a condition-variable-based barrier. The change was not retained, as incorrect barrier participation accounting introduced deadlock and increased wait overhead under distributed execution.

Another refactor was attempted to separate orchestration and compute responsibilities on the root node to avoid interference with root-side compute performance. The first approach introduced a dedicated orchestration-only node with no compute role, but was not pursued further due to the scale of refactoring required. A second, less aggressive approach attempted to split orchestration and compute into separate threads so that efficiency cores could handle orchestration while performance cores remained focused on compute. This was also not retained due to the original codebase’s dependence on the existing threading model.

### Communication Changes

**Targeted issue:** Communication overhead from fixed small socket transfer granularity in the send/read path.

The fixed socket transfer granularity was replaced with a configurable network chunk size controlled through the Makefile. Multiple chunk sizes were evaluated, with 256 KB selected as the best-performing setting in the tested configurations. This change was retained.

TCP buffer size tuning was also evaluated as a possible way to reduce transfer overhead and improve synchronization behavior. However, these changes did not produce meaningful or consistent performance gains in the tested setup and were not retained.

### Build and Makefile Changes

**Targeted issue:** Suboptimal local compute performance on the target ARM hardware.

More aggressive hardware-targeted compiler settings were evaluated to improve local compute performance. The ARM Cortex-A76 performance cores have support for the dotprod extension, which enables the UDOT and SDOT integer dot product instructions. These instructions allow the CPU to compute 4 multiply-accumulate operations in a single cycle, making them significantly more efficient than standard NEON SIMD for Q40 quantized matrix multiplication. Enabling dotprod effectively doubled evaluation throughput and improved prediction throughput by nearly 1.5 tok/s. We also used and tested other flag configurations to improve performance, and the final selected build configuration was: 

```make
-mcpu=cortex-a76+fp16+dotprod -Ofast -flto -fomit-frame-pointer
```

This configuration delivered the best overall result among the tested builds and was adopted as the preferred compilation target for the optimized branch.

### Stability Adjustments

**Targeted issue:** Runtime instability on the root node when using dot-product acceleration.

Enabling dotprod at the default 2.4GHz clock frequency caused the head node (rock0) to crash. Through systematic frequency testing, we identified that rock0's cannot sustain dotprod workloads above 1992MHz, while worker nodes remained stable at all frequencies. So we capped the root node CPU frequency to 1992MHz.  

### Configuration Sweeps

As one of the final stages of optimization, controlled sweeps were run across key runtime parameters using `/tools/sweep.sh` to identify stable and high-performing operating points. These sweeps covered:

- `nthreads`
- `max_seq_len`
- batch size
- network chunk size
- prompt length

These experiments helped establish the preferred deployment regime for the current codebase and confirmed that synchronization overhead remained the dominant performance constraint across workload shapes.

### Final Outcome

The retained attempts of this optimization cycle were expanded profiling, improved structured benchmarking, a configurable network chunk size with 256 KB selected as the preferred setting, hardware-targeted build configuration, and a root-node CPU frequency cap for stability.
