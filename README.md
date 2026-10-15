# Eviction Policies for vLLM's CPU KV-Cache Offload Tier

Prefix KV-cache reuse skips prefill for repeated context (system prompts, multi-turn chat,
agent loops). This project studies **which KV chunks to keep** when the cache is full,
using **public production traces** (Mooncake/Kimi, FAST'25) and **vLLM 0.30's real
offload code**, and contributes **DIP-KV**, an adaptive-admission policy that plugs into
vLLM through its official out-of-tree `CachePolicy` hook (no fork).

## Key findings

1. **On a single GPU's KV cache, the eviction policy barely matters.** At HBM-scale
   capacity, LRU, LFU and cost-aware GDSF give identical hit rates on all traces, because
   ~85% of written chunks are never reused and reuse arrives a median of ~70 s later. The
   lever is the larger **CPU offload tier** (vLLM `OffloadingConnector`, LMCache-style).
2. **vLLM's ARC policy collapses to LRU.** The connector touches a request's keys again
   after storing them, so ARC promotes every new chunk to its "frequent" list (T1 stays
   empty). Reproduced inside vLLM's own `CPUOffloadingManager` (`tests/test_real_manager.py`).
3. **vLLM's `store_threshold=2` admission filter helps at small capacity but hurts at large
   capacity** (chat, 16k chunks: 27.3% → 21.6% hit rate).
4. **Prefix closure dominates eviction quality.** A policy that evicts mid-chain chunks
   strands every descendant. Making LHD prefix-aware (leaf-only eviction) lifted its hit
   rate from 11% to 24% on the synthetic trace.
5. **DIP-KV** (probation insertion + ghost list, switched adaptively against LRU by sampled
   shadow caches) matches or beats LRU on both real traces at every capacity. Small and
   mid capacity give up to ~2× LRU's hit rate; at large capacity it falls back to LRU.

### CPU-tier token hit rate, full 1-hour traces (512-token chunks)

| Trace | Chunks | LRU (vLLM) | ARC (vLLM) | LRU + admission (vLLM) | LHD (prefix-aware) | **DIP-KV** | Belady (oracle) |
|---|---|---|---|---|---|---|---|
| chat | 1,000 | 4.6% | 4.6% | 5.9% | 6.4% | **8.0%** | 17.5% |
| chat | 2,000 | 5.6% | 5.6% | 9.6% | 8.2% | **11.3%** | 25.2% |
| chat | 4,000 | 9.2% | 9.2% | 15.4% | 12.3% | **15.6%** | 32.4% |
| chat | 16,000 | 27.3% | 27.3% | 21.6% | 26.6% | **27.3%** | 37.3% |
| tool-agent | 1,000 | 35.2% | 35.2% | 36.2% | 35.7% | **37.8%** | 44.4% |
| tool-agent | 4,000 | 39.1% | 39.1% | 43.3% | 39.1% | **43.1%** | 55.0% |
| tool-agent | 16,000 | 52.2% | 52.2% | 47.0% | 48.8% | **52.1%** | 57.0% |

The full tables are in `results/tier_*.csv`. On the 4k-request synthetic trace DIP-KV can
trail LRU by up to ~2 points (adaptive switch noise on a short trace); see the CSV.

**Simulator fidelity:** `tests/test_real_manager.py` replays the traces through vLLM 0.30's
real `CPUOffloadingManager`. All policies match the simulator within 0.15 points, and four
decimals for vLLM's own LRU, ARC and admission.

## Measured on a real GPU (vLLM 0.30, A100-80GB)

Qwen2.5-1.5B-Instruct served by `vllm serve` with the `OffloadingConnector` CPU tier (500 chunks
of 512 tokens, about 7.3 GB of pinned host memory), with a deliberately small GPU KV pool (~98k tokens).
Each trace is replayed **open-loop at its real arrival pattern** (1/4 session sample, time-compressed
by a speed chosen by calibration). Every policy sees identical traffic. The hit rate comes from vLLM's own
Prometheus counters, and per-request `cached_tokens` agrees.

| | Chat (moderate load) | Tool-agent (high load) |
|---|---|---|
| Prefix hit rate, LRU → **DIP-KV** | 6.6% → **12.7%** | 39.6% → **43.6%** |
| CPU-tier hits, LRU → DIP-KV | 1.4% → 7.5% (**5.4×**) | 4.7% → 8.5% (**1.8×**) |
| p50 TTFT vs vLLM LRU | 806 → **686 ms (−15%)** | 2,023 → **1,209 ms (−40%)** |
| p95 TTFT vs vLLM LRU | 2,979 → **2,591 ms (−13%)** | 6,083 → **4,550 ms (−25%)** |
| p95 TTFT vs vLLM `store_threshold=2` | −4.5% | −3.0% |
| p95 TTFT vs GPU-only prefix cache | −12% | −58% |
| Simulator-predicted vs measured hit rate | within 0.7 pts | within 0.4 pts |

* **With vLLM's default LRU, the CPU tier barely helps chat traffic** (+1.4 pts hit rate, same
  latency as GPU-only). DIP-KV makes the same tier 5.4× more productive.
* **Latency gains grow with load.** Tool-agent ran near saturation: TTFT drifted upward over the
  run for GPU-only (2.7 s → 9.2 s median, first vs last 10%) and slightly for LRU (1.4 → 2.1 s),
  while DIP-KV stayed flat (1.1 → 1.3 s). Near capacity, the prefill work a better cache saves
  also shortens every queue behind it.
* With no prefix caching at all, tool-agent at this load does not keep up (p50 TTFT 133 s).
* Failed requests were at most 0.2% per run (≤10 of 5,810), all under peak load.
* Setup caveats: 1.5B model on a single GPU; 1/4 session sample with CPU capacity scaled by 1/4
  (offline check: hit-rate curves are preserved); prompts truncated to 30k tokens (prefix kept);
  outputs capped at 64 tokens; token contents synthetic (the traces publish block hashes only), with
  the real sharing structure. Raw per-request results: `results/gpu/`.

## Layout
```
kvpolicy/      vLLM CachePolicy plugins: dip.py (DIP-KV, PG ablation), lhd.py, probe.py (logging)
kvsim/         simulators: tier_sim.py (CPU tier, drives real vLLM policy classes), run.py/cache.py (GPU radix cache)
vllm_stub/     verbatim vLLM 0.30 LRU/ARC policy files + 10-line stub, for offline runs
bench/         GPU benchmark: prep.py (session sampling), run.py, calibrate.py, predict.py
colab/         kv_offload_benchmark.ipynb (A100, ~3 h core grid, resumable on Drive)
tests/         real-vLLM manager replay, fake vLLM server for harness tests
results/       CSVs from the simulations
```

## Reproduce
```bash
./scripts_get_traces.sh
python kvsim/tier_sim.py --trace traces/conversation_trace.jsonl --out results/tier_conversation.csv
# GPU: open colab/kv_offload_benchmark.ipynb in Colab (A100, High-RAM)
```
Use in vLLM:
```bash
PYTHONPATH=. vllm serve <model> --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"cpu_bytes_to_use":8e9,"block_size":512,
  "eviction_policy":"DIPCachePolicy","cache_policy_module_path":"kvpolicy.dip"}}'
```

## Status
- [x] Trace analysis, simulators, policies, validation against vLLM's real offload manager
- [x] GPU runs (TTFT p50/p95 + measured hit rate): `colab/kv_offload_benchmark.ipynb`
- [ ] Confirmation run on a 7B model
- [ ] Report the ARC/re-touch interaction upstream (vLLM issue)
