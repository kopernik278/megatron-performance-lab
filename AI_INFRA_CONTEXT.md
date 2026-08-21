# AI Infra Learning & Project Context

> Purpose: This file is the long-term context for AI coding agents (especially Cursor) working with this repository.
> Target role: AI Training Infrastructure / Distributed Training Engineer.
> Working principle: understand the system first, implement second, benchmark third, profile fourth, optimize fifth.

---

## 1. Career Goal

### Primary target

AI Training Infrastructure Engineer / Distributed Training Engineer.

### Secondary directions

- GPU Kernel / CUDA Optimization Engineer
- AI Systems Engineer
- AI Compiler Engineer
- LLM Training Systems Engineer
- LLM Inference Infrastructure Engineer

### Priority

1. Distributed LLM Training Infrastructure
2. GPU Kernel Optimization
3. NCCL / RDMA / RoCE / InfiniBand
4. Training Performance Engineering
5. AI Compiler
6. LLM Inference Infrastructure
7. MoE / Expert Parallelism

The goal is not merely to use existing frameworks. The goal is to understand, implement, profile, debug, and optimize the underlying systems.

---

## 2. Learning Philosophy

Use the following loop for every major topic:

```text
Theory
  ↓
Architecture
  ↓
Design
  ↓
Implementation
  ↓
Baseline Benchmark
  ↓
Profiling
  ↓
Bottleneck Analysis
  ↓
Optimization
  ↓
Benchmark Again
  ↓
Documentation
```

Do not optimize without measurements.

Do not accept code that cannot be explained.

For important components, prefer:

1. Explain the underlying principle.
2. Design the data flow and API.
3. Implement a minimal version.
4. Test correctness.
5. Establish a baseline.
6. Profile.
7. Identify the bottleneck.
8. Implement one optimization.
9. Compare before/after.
10. Document the trade-offs.

---

## 3. Current Knowledge

### 3.1 Distributed Training

Already studied:

- DDP
- FSDP
- ZeRO
- ZeRO-Offload
- ZeRO-Infinity
- Tensor Parallelism
- Pipeline Parallelism
- Sequence Parallelism
- Context Parallelism
- Expert Parallelism
- Activation Checkpointing
- Megatron-LM
- DeepSpeed
- Distributed Checkpoint concepts

Important concepts:

- AllReduce
- AllGather
- ReduceScatter
- Point-to-point communication
- 1F1B pipeline scheduling
- Communication/computation overlap
- Gradient synchronization
- Sharding
- Parameter / gradient / optimizer-state memory

---

### 3.2 CUDA / GPU Architecture

Already studied:

- GPU architecture
- SM
- Warp
- Thread
- Block
- Grid
- CUDA memory hierarchy
- Global Memory
- Shared Memory
- Register
- Coalesced Memory Access
- Shared Memory Bank Conflict
- Occupancy
- Register Pressure
- Tiling
- Tensor Core
- MMA / `mma.sync`
- CUDA Graph
- Kernel Fusion
- Software Pipeline
- Double Buffering
- Roofline Analysis

Performance analysis tools studied:

- Nsight Systems
- Nsight Compute
- PyTorch Profiler
- NCCL Profiling

---

### 3.3 GPU Kernel Optimization

Already studied:

- GEMM optimization
- CUTLASS architecture
- Threadblock Tile
- Warp Tile
- Thread Tile
- Tensor Core Tile
- Global → Shared → Register → Tensor Core data flow
- Software pipelining
- Double buffering
- FlashAttention architecture

---

### 3.4 FlashAttention

Already studied:

- Standard Attention memory bottleneck
- Tiling
- Online Softmax
- FlashAttention architecture
- FlashAttention-1 / 2 / 3 high-level evolution
- Shared Memory
- Register Fragment
- Tensor Core
- `mma.sync`
- Software Pipeline
- Double Buffer
- Hopper-oriented optimization concepts such as TMA / Warp Group at a high level

Online Softmax was intentionally skipped later because it had already been learned in depth.

---

### 3.5 Networking / Distributed Communication

Already studied:

- MPI
- NCCL
- RDMA
- RoCE
- RoCE v2
- InfiniBand
- PFC
- ECN
- GPUDirect RDMA concepts
- Ring AllReduce concepts
- Collective communication

Important distinction:

```text
RDMA
├── InfiniBand
└── RoCE
```

NCCL is the main GPU collective communication library in modern GPU training.

MPI is a general HPC message-passing standard and may also be used for process launch / initialization, while NCCL typically handles GPU communication.

---

### 3.6 Quantization / Low Precision

Already studied:

- FP32
- FP16
- BF16
- TF32
- FP8
- E4M3
- E5M2
- INT8
- Scale
- Zero Point
- Symmetric Quantization
- Asymmetric Quantization
- Per-Tensor Quantization
- Per-Channel Quantization
- PTQ
- QAT
- GPTQ
- AWQ
- TensorRT-LLM quantization concepts

Important principle:

Low precision is useful only when the hardware and kernels actually exploit it.

---

### 3.7 LLM Inference / Acceleration

Already studied at a high level:

- TensorRT-LLM
- KV Cache
- Paged KV Cache concepts
- Continuous Batching
- CUDA Graph
- FlashAttention
- Kernel Fusion
- INT4 / INT8 / FP8
- Tensor Core acceleration

vLLM comparison was intentionally skipped for now.

---

## 5. Recommended Project Portfolio

The target portfolio is a connected AI Training Infrastructure platform rather than several unrelated toy projects.

### Project 1 — Mini Distributed LLM Training Engine

Priority: S / Highest

Goal:

Build a simplified training engine inspired by Megatron-LM / DeepSpeed.

Required stages:

```text
Stage 1:
Single GPU baseline

Stage 2:
DDP + NCCL

Stage 3:
Tensor Parallelism

Stage 4:
Pipeline Parallelism

Stage 5:
1F1B scheduling

Stage 6:
Distributed Checkpoint

Stage 7:
Profiling

Stage 8:
Communication/Computation overlap
```

Important features:

- Config-driven training
- Dataset pipeline
- Distributed initialization
- DDP
- Tensor Parallel Linear layers
- Pipeline Parallel
- NCCL collectives
- Checkpoint save/load
- Resharding if feasible
- Profiling hooks
- Benchmark scripts

---

### Project 2 — Megatron-LM Performance Optimization Lab

Priority: S / Highest

Goal:

Use a real LLM training framework and perform measurable performance optimization.

Baseline metrics:

- tokens/sec
- samples/sec
- step time
- MFU
- GPU utilization
- memory utilization
- communication time
- computation time
- NCCL time

Profiling:

```text
PyTorch Profiler
      ↓
Nsight Systems
      ↓
Nsight Compute
```

Investigate:

- communication bottlenecks
- NCCL overhead
- synchronization
- idle GPU periods
- kernel launch overhead
- memory bandwidth
- Tensor Core utilization
- kernel efficiency
- TP/PP configuration

Every optimization must include:

```text
Before
After
Delta
Explanation
```

---

### Project 3 — NCCL / RoCE Communication Benchmark

Priority: S

Goal:

Understand GPU collective communication at the systems level.

Benchmark:

- Point-to-point bandwidth if useful
- AllReduce
- AllGather
- ReduceScatter
- Broadcast

Variables:

- message size
- number of GPUs
- number of nodes
- topology
- protocol
- network configuration

Measure:

- latency
- algorithmic bandwidth
- bus bandwidth
- GPU utilization
- CPU utilization
- scaling efficiency

Study:

- Ring AllReduce
- Tree-based communication
- NCCL topology awareness
- RDMA
- RoCE
- InfiniBand
- communication/computation overlap

If actual RoCE/IB hardware is unavailable, use a realistic local/simulated benchmark only for conceptual experiments and clearly label it. Do not fabricate RoCE/IB measurements.

---

### Project 4 — CUDA / Triton Transformer Kernel Optimization

Priority: A+

Goal:

Implement and optimize transformer-related GPU kernels.

Possible progression:

```text
Vector operation
  ↓
Reduction
  ↓
RMSNorm / LayerNorm
  ↓
Fused RMSNorm + Linear
  ↓
Fused MLP component
  ↓
GEMM-related kernel
  ↓
Attention component
```

Compare:

- PyTorch baseline
- Triton implementation
- CUDA implementation
- optimized implementation

Measure:

- latency
- effective bandwidth
- TFLOPS
- occupancy
- register usage
- shared-memory usage
- Tensor Core utilization

Use Nsight Compute for important kernels.

---

### Project 5 — Distributed MoE Training System

Priority: A+

Goal:

Understand MoE training infrastructure.

Implement:

```text
Router
  ↓
Token Routing
  ↓
Dispatch
  ↓
All-to-All
  ↓
Expert Computation
  ↓
All-to-All
  ↓
Combine
```

Study:

- Expert Parallelism
- All-to-All
- Token routing
- capacity factor
- load balancing
- expert imbalance
- communication overhead
- communication/computation overlap

Advanced extension:

- fused routing
- grouped GEMM
- expert load balancing
- DeepEP concepts
- optimized dispatch

---

## 6. Recommended Repository Structure

Use a consistent structure:

```text
ai-infra/
│
├── README.md
├── AI_INFRA_CONTEXT.md
│
├── docs/
│   ├── architecture/
│   ├── design/
│   ├── learning-notes/
│   └── experiments/
│
├── projects/
│   ├── mini-training-engine/
│   ├── megatron-performance-lab/
│   ├── nccl-roce-benchmark/
│   ├── triton-transformer-kernel/
│   └── moe-training-system/
│
├── benchmarks/
├── profiles/
├── results/
├── scripts/
└── tests/
```

For each project:

```text
project/
├── README.md
├── docs/
├── src/
├── tests/
├── configs/
├── scripts/
├── benchmarks/
├── profiles/
└── results/
```

---

## 7. Benchmarking Rules

Never report fabricated performance numbers.

All performance claims must come from actual runs or be explicitly marked as theoretical expectations.

Every benchmark should record:

- hardware
- GPU model
- GPU count
- CPU
- RAM
- CUDA version
- driver version
- PyTorch version
- NCCL version
- compiler version
- batch size
- sequence length
- hidden size
- precision
- parallelism configuration
- software commit/hash
- timestamp

Example:

```text
GPU: NVIDIA H100 80GB
GPUs: 4
Precision: BF16
Sequence Length: 4096
Global Batch Size: ...
Tensor Parallel: 4
Pipeline Parallel: 1
CUDA: ...
PyTorch: ...
NCCL: ...
Commit: ...
```

---

## 8. Profiling Rules

### Nsight Systems

Use for:

- CPU/GPU timeline
- kernel launch gaps
- CUDA Graph behavior
- NCCL communication
- communication/computation overlap
- GPU idle time

### Nsight Compute

Use for:

- individual kernel analysis
- occupancy
- register pressure
- shared memory
- memory throughput
- Tensor Core utilization
- instruction mix
- roofline analysis

### PyTorch Profiler

Use for:

- operator-level bottlenecks
- model-level profiling
- PyTorch execution breakdown

---

## 9. AI Coding Agent Rules

The AI coding agent should behave like a senior AI Infra engineer and tutor.

### Before coding

Explain:

1. Problem
2. Requirements
3. Architecture
4. Data flow
5. API/interface
6. Correctness considerations
7. Performance considerations
8. Expected bottlenecks

### During coding

- Prefer small incremental changes.
- Do not rewrite large parts of the repository unnecessarily.
- Preserve working baselines.
- Add tests for correctness.
- Keep performance-sensitive code isolated and measurable.
- Explain non-obvious CUDA/NCCL behavior.
- Do not hide complexity behind unnecessary abstractions.

### After coding

Always provide:

1. What changed
2. Why it changed
3. How to test
4. Expected result
5. Performance experiment
6. Possible next optimization

### Important

Never claim a benchmark result that was not actually measured.

If GPU hardware is unavailable:

- provide code that should run on supported hardware;
- provide expected qualitative behavior;
- clearly label any numbers as estimates;
- do not present estimates as measurements.

---

## 10. ChatGPT ↔ Cursor Workflow

Use ChatGPT primarily for:

- learning theory
- architecture design
- system decomposition
- algorithm explanation
- performance reasoning
- experiment design
- code review
- interpreting profiling results

Use Cursor primarily for:

- repository navigation
- implementation
- refactoring
- tests
- debugging
- incremental code changes
- Git workflow

Recommended loop:

```text
ChatGPT
  ↓
Understand concept
  ↓
Design RFC
  ↓
Cursor
  ↓
Implement
  ↓
GPU server
  ↓
Benchmark
  ↓
Nsight / Profiler
  ↓
Results
  ↓
ChatGPT
  ↓
Analyze bottleneck
  ↓
Cursor
  ↓
Optimize
  ↓
Benchmark again
```

---

## 11. RFC Requirement

Before implementing a major feature, create a short RFC.

Example:

```text
docs/design/RFC-001-tensor-parallel-linear.md
```

Structure:

```text
# RFC: Tensor Parallel Linear

## Motivation

## Goals

## Non-goals

## Architecture

## Data Flow

## Communication

## Memory Behavior

## Correctness

## Performance Hypothesis

## Benchmark Plan

## Risks

## Future Work
```

This trains the ability to communicate system design in interviews.

---

## 12. Correctness First

For distributed and GPU projects, always verify correctness before optimization.

Required checks where applicable:

- numerical equivalence
- gradient equivalence
- distributed synchronization correctness
- deterministic tests where feasible
- checkpoint correctness
- restart correctness
- tensor shape validation
- communication correctness

For low precision:

Compare against a higher precision reference and report:

- max absolute error
- mean absolute error
- relative error
- task/model accuracy where appropriate

---

## 13. Performance Optimization Framework

When a workload is slow, classify the bottleneck:

```text
Compute Bound
Memory Bound
Launch Bound
Communication Bound
Synchronization Bound
```

Then investigate:

```text
GPU Utilization
↓
SM Utilization
↓
Tensor Core Utilization
↓
Memory Throughput
↓
Occupancy
↓
Register Pressure
↓
Kernel Launch
↓
NCCL
↓
CPU/GPU overlap
```

Do not assume the bottleneck.

Measure it.

---

## 14. Current Recommended Learning Sequence

### Completed / largely completed

```text
Distributed Training fundamentals
        ↓
GPU architecture
        ↓
CUDA fundamentals
        ↓
NCCL fundamentals
        ↓
RDMA / RoCE / InfiniBand
        ↓
GPU Kernel Optimization
        ↓
CUTLASS architecture
        ↓
FlashAttention
        ↓
Quantization
        ↓
GPTQ / AWQ
        ↓
FP8
        ↓
TensorRT-LLM acceleration
        ↓
Profiling fundamentals
```

### Next major topics

```text
AI Compiler
    ↓
TorchDynamo
    ↓
Torch FX
    ↓
TorchInductor
    ↓
Triton
    ↓
MLIR / AI compiler concepts
```

At the same time, begin project implementation rather than waiting to finish every theory topic.

---

## 15. Interview-Oriented Skills

The candidate should eventually be able to explain and implement:

### Distributed Training

- Why DDP scales
- Why communication becomes a bottleneck
- AllReduce algorithms
- Ring vs Tree
- Tensor Parallel
- Pipeline Parallel
- Expert Parallel
- Gradient synchronization
- Checkpointing

### GPU

- SM / Warp
- Tensor Core
- memory hierarchy
- occupancy
- roofline
- shared memory bank conflicts
- register pressure
- kernel fusion
- CUDA Graph

### Networking

- RDMA
- RoCE v2
- InfiniBand
- GPUDirect RDMA
- PFC
- ECN
- NCCL over RDMA

### Performance

- tokens/sec
- MFU
- scaling efficiency
- communication/computation overlap
- Nsight Systems
- Nsight Compute

### LLM Training

- data pipeline
- tokenizer
- distributed optimizer
- checkpoint
- mixed precision
- activation checkpointing
- parallelism strategies
- MoE

---

## 16. Project Quality Standard

A project is considered complete only when it has:

```text
[ ] Correct implementation
[ ] Unit tests
[ ] Distributed tests where applicable
[ ] Baseline
[ ] Benchmark script
[ ] Real benchmark results
[ ] Profiling evidence
[ ] Bottleneck analysis
[ ] At least one optimization
[ ] Before/after comparison
[ ] Architecture diagram
[ ] Design document
[ ] README
[ ] Reproducible environment
[ ] Clear limitations
[ ] Future work
```

A project with only source code is incomplete.

---

## 17. Final Portfolio Goal

The final portfolio should demonstrate:

```text
Python / C++ / CUDA
        +
PyTorch
        +
Distributed Training
        +
NCCL
        +
RDMA / RoCE / IB
        +
GPU Kernel Optimization
        +
Profiling
        +
LLM Training
        +
MoE
        +
AI Compiler
```

The strongest evidence is not the number of repositories.

The strongest evidence is:

> "I identified a real systems bottleneck, measured it, changed the implementation, and demonstrated the performance improvement with reproducible experiments."

---

## 18. Working Agreement with AI

When the user asks to "enter the next section", continue from the current learning roadmap instead of restarting from the beginning.

When the user says a topic has already been learned, skip it.

When the user explicitly says to skip a framework/topic, do not repeatedly teach it unless requested.

Prefer compact formatting and dense technical explanations when requested.

For difficult topics, use:

```text
Concept
→ Intuition
→ Architecture
→ Mathematical / systems model
→ Implementation
→ Profiling
→ Optimization
→ Interview questions
```

The ultimate goal is independent engineering ability, not merely completing lessons.
