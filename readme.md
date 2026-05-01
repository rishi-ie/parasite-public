# Parasite: A Modular Architecture for Structured Reasoning

## Overview

Parasite is a reasoning system that separates knowledge, reasoning machinery, and generation into independent, composable components connected by a clean protocol.

Rather than conflating factual knowledge, reasoning circuits, and language generation into a single dense model weight matrix, Parasite distributes these concerns across three substrates:

- **Knowledge Layer (Vindex)**: A queryable graph database extracted from a frontier transformer, stored on SSD, never fully loaded into RAM.
- **Reasoning Layer (Spark)**: A 27M-parameter Hierarchical Reasoning Model that owns all cognition and iteration logic.
- **Protocol Layer (LQL)**: A structured query language that enables the reasoning module to retrieve exactly the knowledge slice it needs at each step.

## Key Properties

| Property               | Dense LLM                    | Parasite                             |
| ---------------------- | ---------------------------- | ------------------------------------ |
| Compute per query      | ~140 trillion FLOPs          | ~600 million FLOPs (~240× reduction) |
| Storage per specialist | Model replica (100+ GB)      | 54 MB                                |
| Knowledge updates      | Full retrain (months, $M)    | Millisecond patches (< 1ms, ~$0)     |
| Hallucination          | 5–15% on multi-hop reasoning | Structural zero (100% provenance)    |
| Expertise cost         | Fine-tune per domain ($K–$M) | Train one specialist (~$100, hours)  |
| Hardware to deploy     | High-end datacenter          | Single consumer GPU ($2K)            |

## Core Objectives

### 1. **Eliminate Hallucination Through Structural Design**

Every factual claim in an answer traces directly to a vindex edge with a confidence score. Generation hallucination is impossible by construction.

### 2. **Achieve 240× Compute Reduction**

Reasoning is 27M parameters doing iterative retrieval, not 70B+ parameters doing dense forward passes. Per-query compute drops from ~140T to ~600M FLOPs.

### 3. **Make Knowledge Updates Instant and Cheap**

Patch system applies knowledge updates in <1ms without touching the base knowledge store or retraining anything. Facts stay current without infrastructure cost.

### 4. **Enable Edge Deployment**

A 27M reasoner + SSD knowledge store runs fully on consumer hardware — phones, laptops, air-gapped systems — with no API dependency.

### 5. **Create a Specialist Factory**

Marginal cost of a new reasoning specialist (math, logic, domain-specific) collapses to ~$100 and hours of training. Each 54MB specialist is independently swappable.

### 6. **Provide Full Provenance and Auditability**

Every step of reasoning is logged, every fact is traced to a source, every confidence score is recorded. No black box.

### 7. **Support Multi-Hop Factual Reasoning at Frontier Quality**

Match or exceed 70B+ models on structured, knowledge-grounded tasks (HotpotQA, 2WikiMultiHopQA, TriviaQA, fact verification) while maintaining 100% factual traceability.

## Target Capability Map

### Bucket A — Will match or exceed frontier models

- Multi-hop factual QA (HotpotQA, 2WikiMultiHopQA, MuSiQue)
- Single-hop factual recall (TriviaQA, NaturalQuestions)
- Knowledge-grounded entity reasoning
- Structured fact verification (FEVER, FActScore)
- Constraint satisfaction with explicit rules

### Bucket B — Will be competitive (60–90% of frontier)

- Reading comprehension over passages
- Commonsense QA (when encodable as edges)
- Simple arithmetic and word problems
- Structured extraction and classification

### Bucket C — Will see meaningful loss

- Broad academic QA (MMLU): ~60–70% of frontier
- Long-context reasoning
- Multi-step math (GSM8K, MATH)

### Bucket D — Structurally out of scope

- Fluent prose generation, creative writing
- Open-ended dialogue, pragmatic inference
- In-context learning from long examples
- Genuine novelty and synthesis

## Architecture Layers

```
Query Input
    ↓
Spark (27M params) — Reasoning Layer
    ↓
LQL Protocol — Structured Query Interface
    ↓
Vindex (SSD-based) — Knowledge Layer
    ↓
Answer + Provenance Trace
```

Each component is independently replaceable and auditable.

## Research Questions (Phase-Gated)

The architecture rests on several claims that will be validated in phased experiments:

1. **Edge quality at depth** (Phase 0): Can the extracted knowledge graph maintain ≥0.85 confidence across 3+ hops?
2. **Reasoning under polysemy** (Phase 1): Can Spark reliably disambiguate entity senses using accumulated context?
3. **Query head learning** (Phase 1): Can a small reasoner learn to generate valid, productive queries via supervised training?
4. **Self-improvement stability** (Phase 4+): Do self-patched knowledge improvements compound or eventually degrade?

Kill gates are explicit; failed experiments pivot rather than push through.

## Build Roadmap

| Phase        | Scope                                              | Timeline | Gate                                                  |
| ------------ | -------------------------------------------------- | -------- | ----------------------------------------------------- |
| **Phase 0**  | Edge validation: 20-query manual test on vindex    | 1 week   | ≥0.85 confidence at 3-hop, ≥80% top-3 recall          |
| **Phase 1**  | Spark + end-to-end pipeline on TriviaQA + HotpotQA | 9 weeks  | TriviaQA EM > 60%, HotpotQA F1 within 10% of baseline |
| **Phase 2**  | First specialist (HRM-Math) on GSM8K + MATH        | 8 weeks  | GSM8K > 50%                                           |
| **Phase 3**  | Router + multi-specialist composition              | 6 weeks  | Routing accuracy > 90%                                |
| **Phase 4+** | Specialist pipeline automation                     | Ongoing  | End-to-end training without manual intervention       |

## Hardware & Economics

- **Spark**: ~2 GB f16, runs on RTX 4090 or equivalent
- **Vindex**: ~10 GB f16 for Qwen 2.5 72B, mmap'd on NVMe ($150 SSD)
- **Parallelism**: 100+ stateless Spark instances on a single machine (read-only vindex)
- **Per-query cost**: ~$0.002 (electricity only, no cloud API dependency)

## Source Model

Built on Qwen 2.5 72B, extracted via LARQL into a queryable knowledge graph. The vindex format is compatible with standard HuggingFace safetensors/GGUF for distribution.

## Open Source & Licensing

Architecture and protocol are designed to be openly specified. Final licensing and open-source strategy TBD.

---

**Parasite v1** — Pre-build specification
**Mumbrane | April 2026**

_Build Phase 0 first. Everything depends on it._
