# MANGO vLLM Inference Optimization
## Experimental Journey Log

**Project:** Serving `CMKL/MANGO1.5-Qwen3.5-9B` with vLLM  
**Environment:** NVIDIA GPU A100 inference environment  
**Image:** `vllm/vllm-openai:latest` in the recorded experiments  
**Primary objective:** Improve benchmark performance while balancing throughput, latency, accuracy, and SLO compliance.

---

## 1. Research Objectives

The purpose of these experiments is to understand how vLLM serving configurations affect real-world benchmark performance.

The investigation focuses on:

- **Benchmark score:** Overall evaluation performance.
- **Accuracy:** Whether serving optimizations preserve answer quality.
- **P95 TTFT:** Time to first token at the 95th percentile.
- **TPOT:** Time per output token.
- **Throughput and efficiency:** How efficiently the server processes concurrent requests.
- **SLO compliance:** The proportion of requests meeting latency requirements.
- **Request queuing:** Whether the server can handle incoming requests without excessive waiting.
- **Memory efficiency:** GPU memory consumption and KV-cache capacity.
- **Prefix caching:** Whether reusing shared prompt prefixes improves performance.
- **Speculative decoding:** Whether Multi-Token Prediction (MTP) improves inference efficiency.

### Experimental principle

Change a small number of variables at a time, record the results, and distinguish between measured outcomes and explanations that still require testing.

A higher benchmark score alone does not necessarily prove that a configuration is better. The score must be interpreted alongside latency, accuracy, SLO compliance, and experimental conditions.

---

## 2. Experiment Timeline

### Phase 1 — Establishing the Serving Configuration

#### Observation 1: Small `max-num-seqs` can create a large queue

**Observed configuration:** `max-num-seqs=8`

**Finding:** A large request queue was observed with this setting.

**Initial hypothesis:** The server's effective scheduling capacity may have been insufficient for the incoming workload.

**Interpretation:** Limiting the number of active sequences can constrain concurrency. However, the actual effect depends on request lengths, batching behavior, GPU capacity, and workload arrival patterns.

**Lesson learned:** A low sequence limit should not automatically be treated as a latency optimization.

**Next experiment:**
- Compare `max-num-seqs=8`, `32`, and `128`.
- Keep the workload and other serving parameters constant.
- Record queue length, P95 TTFT, throughput, and GPU utilization.

---

### Phase 2 — Investigating Sequence Concurrency and Batching

#### Observation 2: Increasing sequence capacity did not behave as expected

**Recorded conditions:**
- `max-num-batched-tokens=8192`
- MTP enabled with one speculative token
- Comparison between `max-num-seqs=32` and `128`

**Finding:** The configuration with `max-num-seqs=32` produced approximately **5× worse P95 TTFT** than the configuration with `max-num-seqs=128`.

**Initial hypothesis:** The lower sequence limit may have constrained scheduling flexibility or caused requests to wait longer before being admitted into execution.

**Important qualification:** This result does not establish that `128` is universally optimal. It indicates that, under the tested workload and configuration, the lower limit performed substantially worse.

**Lesson learned:** Increasing concurrency capacity can improve request latency when the existing limit is restrictive.

**Follow-up experiments:**
1. Compare sequence limits of `32`, `64`, `96`, and `128`.
2. Record the actual number of active and waiting requests.
3. Check whether the GPU is compute-bound, memory-bound, or underutilized.
4. Repeat each configuration with the same request workload.

---

### Phase 3 — Testing Quantization and MTP

#### Experiment #17 — AWQ-W4A16 with MTP

**Date:** October 24, 2026

**Configuration:**
- Quantization: AWQ-W4A16
- MTP: Enabled, one speculative token
- `max-num-seqs=160`
- `max-num-batched-tokens=2048`

**Measured results:**

| Metric | Result |
|---|---:|
| Benchmark score | 27.5 |
| Speculative decoding efficiency, η | 1.39× |
| Energy per request | 247 J/request |
| P95 TTFT | 1.66 s |
| TPOT | 60 ms |
| SLO score | 53/100 |
| Accuracy | 87.8% |

**Findings:**

- Speculative decoding efficiency was reported as 1.39×.
- P95 TTFT was 1.66 seconds.
- Accuracy was 87.8%.
- SLO performance remained an important limitation to investigate.

**Initial hypothesis:** MTP was providing some decoding benefit, but that benefit alone might not be sufficient to produce a high overall benchmark score.

**What remains unknown:** The recorded results do not isolate how much of the score came from quantization, MTP, scheduling, latency constraints, or other benchmark components.

**Next experiment:** Compare against an otherwise equivalent configuration with MTP disabled. Keep the model, quantization, workload, and scheduling settings fixed.

**Research question:** Does MTP improve the final benchmark outcome, or does it mainly improve a subset of inference metrics?

---

#### Experiment #18 — AWQ-W4A16 with MTP and FP8 KV Cache

**Date:** October 24, 2026

**Configuration:**
- Quantization: AWQ-W4A16
- `max-num-seqs=128`
- `max-num-batched-tokens=2048`
- KV-cache dtype: FP8
- MTP: Enabled, one speculative token

**Recorded benchmark score:** 41

**Change from Experiment #17:**
- Sequence capacity decreased from 160 to 128.
- FP8 KV cache was explicitly enabled.

**Finding:** The recorded score increased from 27.5 to 41.

**Critical limitation:** Multiple configuration differences and potentially different benchmark conditions prevent attributing this improvement to FP8 KV cache alone.

**Hypotheses to investigate:**

1. **KV-cache capacity hypothesis:** FP8 KV cache may reduce memory usage and allow more effective memory utilization.
2. **Scheduling hypothesis:** Changing sequence capacity may have improved the interaction between request concurrency and batching.
3. **Run-variance hypothesis:** Differences in workload execution, cache state, or benchmark conditions may have contributed to the score change.

**Next experiment:** Repeat Experiment #18 without changing anything, then disable FP8 KV cache while keeping the other settings fixed.

**Decision rule:** Attribute an improvement to FP8 KV cache only if repeated, controlled comparisons support that conclusion.

---

### Phase 4 — Investigating Prefix Caching and Benchmark Variability

#### Observation 3: Validation scores differed substantially between runs

**Observed results:**
- One validation run scored approximately 27.
- A subsequent run scored approximately 89.
- A test run returned a score of 0.0 under conditions involving approximately 20–60 requests and a waiting queue of roughly 20–40 requests.

**Finding:** The observed results were highly variable.

**Initial hypotheses:**

- Prefix-cache state differed between runs.
- Request queuing or latency SLO violations affected benchmark performance.
- Benchmark execution or evaluation behavior differed between runs.
- Errors or timeouts may have affected the test run.

**Important distinction:** Prefix caching can improve performance when requests share reusable prefixes, but it should not inherently improve the model's underlying reasoning ability or answer correctness.

A score difference of this magnitude should not be attributed to prefix caching without further evidence.

**Next experiments:**

1. Run the benchmark with a cold prefix cache.
2. Repeat it with a deliberately warmed prefix cache.
3. Record prefix-cache hit rate, P95 TTFT, TPOT, errors, and SLO compliance.
4. Compare the actual generated answers and accuracy.
5. Confirm that all runs use the same evaluation dataset and scoring procedure.

**Key question:** Is the score measuring model quality, serving performance, or a combination of both?

---

### Phase 5 — Comparing Benchmark Configurations

#### Observation 4: Experiment #32 versus Experiment #24

**Recorded results:**
- Experiment #32: Score 49
- Experiment #24: Score 46

**Finding:** Experiment #32 produced a score three points higher than Experiment #24.

**Initial interpretation:** Experiment #32 may represent an improvement, but the evidence is insufficient to establish a reliable performance gain.

**Unresolved questions:**
- Were both experiments run with the same cache state?
- Were the request workload and concurrency identical?
- Were model parameters and serving flags identical except for the intended variable?
- Was the difference larger than normal run-to-run variation?
- Were accuracy and SLO compliance comparable?

**Next experiment:** Repeat both configurations under identical conditions and compare their average scores and variability.

**Decision rule:** Do not declare Experiment #32 the winner based on one score alone.

---

## 3. Current Hypothesis Register

| ID | Hypothesis | Supporting observation | Confidence | Next validation |
|---|---|---|---|---|
| H1 | Low `max-num-seqs` can create excessive queuing under the tested workload. | `max-num-seqs=8` produced a large queue. | Moderate | Repeat with controlled concurrency. |
| H2 | Increasing sequence capacity can improve P95 TTFT when the lower limit is restrictive. | `32` produced approximately 5× worse P95 TTFT than `128` in the recorded comparison. | Moderate | Repeat with identical requests and other settings. |
| H3 | FP8 KV cache contributed to the improvement from Experiment #17 to #18. | Score increased from 27.5 to 41, and FP8 KV was enabled in #18. | Low | Isolate FP8 KV cache in an A/B experiment. |
| H4 | Prefix-cache state contributes to run-to-run performance differences. | Prefix-cache hits of approximately 20–30% were observed alongside differing benchmark outcomes. | Plausible, unconfirmed | Compare controlled cold-cache and warm-cache runs. |
| H5 | SLO violations or request queuing contributed to poor benchmark results. | Queues were observed, and one test run scored 0.0. | Plausible, unconfirmed | Inspect request-level latency, timeouts, and evaluation logs. |
| H6 | MTP improves decoding efficiency but may not maximize the overall benchmark score. | Experiment #17 reported η = 1.39× alongside a score of 27.5. | Unconfirmed | Compare MTP on/off under identical conditions. |

*Confidence describes the strength of the available evidence, not a statistical probability.*

---

## 4. Lessons Learned So Far

### Lesson 1 — Serving parameters interact

`max-num-seqs`, `max-num-batched-tokens`, MTP, KV-cache dtype, and prefix caching should not be treated as independent performance switches.

A configuration that improves one metric can worsen another.

### Lesson 2 — Benchmark scores need context

A score is useful only when its evaluation conditions are understood. Latency, accuracy, queueing, timeouts, and cache state can all matter, depending on the benchmark's scoring method.

### Lesson 3 — Separate model quality from serving performance

Accuracy and inference performance should be tracked separately.

If accuracy changes unexpectedly, inspect generated outputs and evaluation behavior rather than assuming that a scheduling or caching optimization changed the model's reasoning quality.

### Lesson 4 — One experiment should answer one main question

When several parameters change at once, a better score reveals that the configuration changed successfully, but not which change caused the improvement.

### Lesson 5 — Reproducibility is part of optimization

Repeat promising configurations before accepting them as improvements. Record the exact flags, model revision, runtime image, workload, cache state, and benchmark results.

---

