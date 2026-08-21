# Repository Guidelines

## Project Structure & Module Organization

This repository is an AI training infrastructure and Megatron performance lab. The current top-level layout is:

- `AI_INFRA_CONTEXT.md`: long-term learning goals, project rules, and measurement standards.
- `docs/`: architecture notes, RFCs, design docs, and experiment writeups.
- `configs/`: reproducible training, benchmark, and profiling configurations.
- `scripts/`: runnable utilities for setup, benchmarks, profiling, and result processing.
- `benchmarks/`: benchmark entry points and workload definitions.
- `profiles/`: profiler outputs such as Nsight Systems, Nsight Compute, and PyTorch Profiler traces.
- `results/`: measured results, summaries, and before/after comparisons.
- `patches/`: experimental framework patches or vendor changes.

Add `src/` and `tests/` when implementation code begins. Keep benchmark code separate from analysis artifacts.

## Build, Test, and Development Commands

There is no repo-wide build system yet. When adding executable workflows, prefer explicit scripts and document them in `README.md`.

- `python -m pytest tests/`: run the test suite once tests exist.
- `python scripts/<name>.py --config configs/<run>.yaml`: run a reproducible experiment or utility.
- `nsys profile -o profiles/<run-name> <command>`: collect timeline and NCCL/CPU-GPU overlap data.
- `ncu --set full -o profiles/<kernel-run> <command>`: inspect important CUDA kernels.

Do not commit generated heavyweight profiler traces unless they are intentionally part of a documented result.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, type hints for public APIs, and `snake_case` for modules, functions, configs, and scripts. Use clear metric names such as `tokens_per_sec`, `step_time_ms`, and `mfu`. Name RFCs as `docs/design/RFC-001-short-topic.md`. Name benchmark outputs with date, workload, GPU count, and key parallelism settings when practical.

## Testing Guidelines

Correctness comes before optimization. Add tests for tensor shapes, numerical equivalence, gradients, checkpoint save/load, and distributed synchronization where applicable. Compare low-precision results against a higher-precision reference and report max absolute, mean absolute, and relative error.

## Benchmarking & Profiling Rules

Never fabricate performance results. Every benchmark must record hardware, GPU count, CUDA, driver, PyTorch, NCCL, precision, batch size, sequence length, parallelism configuration, timestamp, and commit hash when available. Performance changes should include `Before`, `After`, `Delta`, and an explanation of the bottleneck.

## Commit & Pull Request Guidelines

This repository has no commit history yet, so no local convention is established. Use concise Conventional Commit-style messages, for example `feat: add tensor parallel benchmark` or `docs: record profiling plan`. Pull requests should include purpose, commands run, correctness evidence, benchmark/profiling evidence when relevant, limitations, and linked issues or RFCs.

