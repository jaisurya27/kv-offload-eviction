"""One benchmark run: start vLLM with a KV-cache configuration, replay a prepared
Mooncake trace open-loop (real arrival times, scaled), record per-request TTFT
and the server's prefix-cache counters, then stop the server.

Resumable: a run whose summary.json exists is skipped.

Configs (--config):
  nocache   prefix caching off (recompute everything)
  gpu       vLLM GPU prefix cache only (stock LRU), no CPU tier
  lru       + CPU offload tier, vLLM's LRU                (LRUProbe)
  arc       + CPU offload tier, vLLM's ARC                (ARCProbe)
  adm2      + CPU offload tier, LRU with store_threshold=2 (vLLM admission filter)
  dip       + CPU offload tier, DIP-KV (ours)              (DIPProbe)
  pg        + CPU offload tier, probation+ghost (ablation) (PGProbe)
  lhd       + CPU offload tier, prefix-aware LHD (ours)    (LHDProbe)
"""
import argparse, asyncio, json, math, os, re, signal, subprocess, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.prep import prepare, stats, BLOCK

TIER = {"lru": ("LRUProbe", 0), "arc": ("ARCProbe", 0), "adm2": ("LRUProbe", 2),
        "dip": ("DIPProbe", 0), "pg": ("PGProbe", 0), "lhd": ("LHDProbe", 0)}
METRICS = ["vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
           "vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_hits"]


# ---------------------------------------------------------------- prompts
class Prompter:
    """Deterministic token ids per trace block: identical hash_id => identical tokens,
    so vLLM's prefix hashing sees exactly the trace's sharing structure."""
    def __init__(self, vocab_lo=1000, vocab_hi=150000):
        self.lo, self.hi, self.cache = vocab_lo, vocab_hi, {}

    def block(self, h):
        b = self.cache.get(h)
        if b is None:
            b = np.random.default_rng(h).integers(self.lo, self.hi, BLOCK, dtype=np.int32)
            self.cache[h] = b
        return b

    def prompt(self, r):
        L = r["input_length"]
        ids = np.concatenate([self.block(h) for h in r["hash_ids"]])[:L]
        return ids.tolist()


def kv_bytes_per_chunk(model):
    if os.environ.get("KV_BYTES_PER_CHUNK"):  # testing without transformers
        return int(os.environ["KV_BYTES_PER_CHUNK"])
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_tok = c.num_hidden_layers * 2 * c.num_key_value_heads * head_dim * 2  # bf16
    return math.ceil(per_tok * BLOCK / 4096) * 4096


# ---------------------------------------------------------------- server
def server_cmd(a, out):
    cmd = ["vllm", "serve", a.model, "--port", str(a.port), "--seed", "0",
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_mem_util),
           "--num-gpu-blocks-override", str(a.gpu_blocks),
           "--max-num-seqs", str(a.max_num_seqs),
           "--enable-prompt-tokens-details", "--disable-uvicorn-access-log"]
    if a.config == "nocache":
        cmd.append("--no-enable-prefix-caching")
    elif a.config in TIER:
        pol, thr = TIER[a.config]
        extra = {"cpu_bytes_to_use": a.cpu_chunks * kv_bytes_per_chunk(a.model),
                 "block_size": BLOCK, "eviction_policy": pol,
                 "cache_policy_module_path": "kvpolicy.probe"}
        if thr:
            extra["store_threshold"] = thr
        cmd += ["--kv-transfer-config", json.dumps({
            "kv_connector": "OffloadingConnector", "kv_role": "kv_both",
            "kv_connector_extra_config": extra})]
    return cmd


def _port_in_use(port):
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def acquire_lock():
    """Only one benchmark may run per machine: two replays against one server
    would mix their traffic and metrics (and corrupt result files)."""
    import fcntl
    fh = open("/tmp/kvbench.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("ERROR: another bench/run.py is already running on this machine. "
                         "Stop it first (pkill -f bench/run.py; pkill -f 'vllm serve').")
    return fh


def start_server(a, out):
    if _port_in_use(a.port):
        raise SystemExit(f"ERROR: port {a.port} is already in use (a vLLM server from another run?). "
                         "Stop it first: pkill -f 'vllm serve'")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               KVPOLICY_LOG=os.path.join(out, "policy.log"))
    cmd = server_cmd(a, out)
    with open(os.path.join(out, "server_cmd.txt"), "w") as f:
        f.write(" ".join(cmd))
    log = open(os.path.join(out, "server.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < a.startup_timeout:
        if p.poll() is not None:
            raise RuntimeError(f"vLLM exited early (code {p.returncode}); see {out}/server.log")
        try:
            urllib.request.urlopen(f"http://localhost:{a.port}/health", timeout=2)
            return p
        except Exception:
            time.sleep(3)
    stop_server(p)
    raise TimeoutError("vLLM did not become healthy; see server.log")


def stop_server(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


def scrape(port):
    import urllib.request
    txt = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10).read().decode()
    vals = {m: 0.0 for m in METRICS}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for m in METRICS:
            if re.match(re.escape(m) + r"(_total)?(\{|\s)", line):
                vals[m] += float(line.rsplit(" ", 1)[1])
    return vals


# ---------------------------------------------------------------- replay
async def replay(a, reqs, out):
    import aiohttp
    pr = Prompter()
    url = f"http://localhost:{a.port}/v1/completions"
    t_first = reqs[0]["timestamp"]
    results = [None] * len(reqs)

    async def one(i, r, session, t_sched):
        body = {"model": a.model, "prompt": pr.prompt(r),
                "max_tokens": max(1, min(r["output_length"], a.max_out)),
                "temperature": 0.0, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
        ttft, cached, err = None, None, None
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    err = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                else:
                    async for raw in resp.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        d = json.loads(line[5:])
                        if ttft is None and d.get("choices"):
                            ttft = time.perf_counter() - t0
                        u = d.get("usage")
                        if u:
                            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
        results[i] = {"i": i, "input_len": r["input_length"], "ttft_s": ttft,
                      "e2e_s": time.perf_counter() - t0, "cached_tokens": cached,
                      "sched_lag_s": t0 - t_sched, "error": err}

    timeout = aiohttp.ClientTimeout(total=a.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        loop = asyncio.get_running_loop()
        start = loop.time()
        tasks = []
        for i, r in enumerate(reqs):
            due = start + (r["timestamp"] - t_first) / 1000 / a.speedup
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            pr.prompt(r)  # warm the token cache before timing starts
            tasks.append(asyncio.create_task(one(i, r, s, time.perf_counter())))
            if i % 500 == 0:
                print(f"  sent {i}/{len(reqs)}  elapsed {loop.time()-start:.0f}s", flush=True)
        await asyncio.gather(*tasks)
    tmp = os.path.join(out, "requests.jsonl.tmp")
    with open(tmp, "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")
    os.replace(tmp, os.path.join(out, "requests.jsonl"))  # atomic: never a half-written file
    return results


def summarize(results, m0, m1, a, wall):
    ok = [x for x in results if x["ttft_s"] is not None and not x["error"]]
    tt = np.array([x["ttft_s"] for x in ok]) * 1000
    d = {k: m1[k] - m0[k] for k in METRICS}
    q = d["vllm:prefix_cache_queries"] or 1
    gpu_hit = d["vllm:prefix_cache_hits"] / q
    ext_hit = d["vllm:external_prefix_cache_hits"] / q
    cached = [x["cached_tokens"] for x in ok if x["cached_tokens"] is not None]
    tot_in = sum(x["input_len"] for x in ok)
    s = {"config": a.config, "trace": a.trace, "model": a.model, "speedup": a.speedup,
         "cpu_chunks": a.cpu_chunks, "cpu_tier": a.config in TIER, "gpu_blocks": a.gpu_blocks,
         "n_requests": len(results), "n_ok": len(ok), "n_errors": len(results) - len(ok),
         "ttft_ms_p50": float(np.percentile(tt, 50)), "ttft_ms_p90": float(np.percentile(tt, 90)),
         "ttft_ms_p95": float(np.percentile(tt, 95)), "ttft_ms_p99": float(np.percentile(tt, 99)),
         "ttft_ms_mean": float(tt.mean()),
         "hit_rate_gpu_tokens": gpu_hit, "hit_rate_cpu_tokens": ext_hit,
         "hit_rate_total_tokens": gpu_hit + ext_hit,
         "hit_rate_from_usage": (sum(cached) / tot_in) if cached and tot_in else None,
         "max_sched_lag_s": max(x["sched_lag_s"] for x in results),
         "wall_s": wall, "metrics_delta": d}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="path to Mooncake jsonl")
    ap.add_argument("--config", required=True, choices=["nocache", "gpu", *TIER])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sample-k", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-prompt", type=int, default=30720)
    ap.add_argument("--max-out", type=int, default=64)
    ap.add_argument("--cpu-chunks", type=int, default=500)
    ap.add_argument("--gpu-blocks", type=int, default=6144)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="first N requests (smoke test)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--request-timeout", type=int, default=3600)
    a = ap.parse_args()

    name = a.run_name or f"{os.path.basename(a.trace).split('_')[0]}_{a.config}_c{a.cpu_chunks}_x{a.speedup:g}"
    out = os.path.join(a.out_root, name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print(f"[skip] {name}: summary.json exists")
        return
    os.makedirs(out, exist_ok=True)
    for f in ("policy.log",):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    _lock = acquire_lock()  # noqa: F841  (held until the process exits)
    reqs = prepare(a.trace, a.sample_k, a.max_prompt)
    if a.limit:
        reqs = reqs[: a.limit]
    print(f"[run] {name}: {stats(reqs)}  speedup={a.speedup}", flush=True)
    p = start_server(a, out)
    if p.poll() is not None:
        raise RuntimeError("vLLM server exited right after becoming healthy; see server.log")
    try:
        m0 = scrape(a.port)
        t0 = time.time()
        results = asyncio.run(replay(a, reqs, out))
        wall = time.time() - t0
        time.sleep(6)  # let the engine publish final stats
        m1 = scrape(a.port)
    finally:
        stop_server(p)
    s = summarize(results, m0, m1, a, wall)
    plog = os.path.join(out, "policy.log")
    if a.config in TIER:
        inits = [json.loads(l) for l in open(plog)] if os.path.exists(plog) else []
        inits = [x for x in inits if x.get("event") == "init"]
        if not inits:
            raise RuntimeError("policy probe never initialised: CPU tier policy did NOT load")
        s["policy_loaded"] = inits[0]["policy"]
        s["cpu_capacity_chunks_actual"] = inits[0]["capacity_chunks"]
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps({k: s[k] for k in ("config", "n_ok", "n_errors", "ttft_ms_p50", "ttft_ms_p95",
                                         "hit_rate_total_tokens", "hit_rate_from_usage")}, indent=1))


if __name__ == "__main__":
    main()
