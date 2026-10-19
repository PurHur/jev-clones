# Question-count scaling

Lux-9B uses the same architecture and runtime measured here with earlier weights, on one otherwise idle AMD gfx942 GPU (SKU M3250101, PCI device 0x74b9, 274,542,362,624 bytes VRAM). These measurements are specific to this host; no cross-hardware speed ranking is claimed.

Each point contains 30 measured requests across six independently loaded processes, with three warmups and five timed calls per cell per process. Input state and each question’s length remain fixed while the number of questions grows. The primary condition uses distinct instructions; repeated instructions are reported separately. Default batching is eight independent questions.

The timer starts after pre-request GPU synchronization and includes request validation, rendering, tokenization, model execution, response construction and final GPU synchronization. It excludes model loading, warmup, network transport and result serialization. Private process caches warm all shapes first; autotune configuration hashes remain unchanged during timing.

## Distinct questions · 499 tokens per question

| Questions | Median (ms) | p95 (ms) | Peak allocated (GiB) |
|---:|---:|---:|---:|
| 1 | 33.17 | 33.52 | 14.95 |
| 2 | 49.33 | 49.72 | 15.02 |
| 4 | 82.01 | 82.69 | 15.17 |
| 8 | 150.41 | 150.92 | 15.46 |
| 16 | 301.06 | 302.22 | 15.46 |
| 32 | 600.74 | 605.47 | 15.46 |

## Repeated questions · 309 tokens per question

| Questions | Median (ms) | p95 (ms) | Peak allocated (GiB) |
|---:|---:|---:|---:|
| 1 | 32.27 | 32.78 | 14.92 |
| 2 | 36.25 | 36.76 | 14.96 |
| 4 | 56.30 | 56.55 | 15.06 |
| 8 | 98.17 | 98.68 | 15.24 |
| 16 | 195.26 | 196.56 | 15.24 |
| 32 | 390.27 | 393.29 | 15.24 |

Every timed output preserved exact response values and key order; each workload also produced one identical output hash across all six processes. Whole-host GPU process checks observed no unrelated GPU work. Reported peaks are PyTorch allocated memory, not total physical reservation.

This is warm local-request latency for two fixed short-input workloads, not concurrent HTTP throughput or long-context performance. [Raw summary and process-block uncertainty](question-scaling.json).
