"""Measure this GPU's prefill throughput, then pick a replay speedup per trace so the
offered load (with no caching) is a target fraction of capacity. Same speedup is
then used for every config of that trace, so configs are compared at equal load."""
import argparse, asyncio, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.prep import prepare, stats
from bench.run import start_server, stop_server


async def burst(port, model, n, length, conc):
    import aiohttp
    rng = np.random.default_rng(12345)
    sem = asyncio.Semaphore(conc)

    async def one(s):
        ids = rng.integers(1000, 150000, length).tolist()  # unique => no cache hits
        async with sem:
            async with s.post(f"http://localhost:{port}/v1/completions",
                              json={"model": model, "prompt": ids, "max_tokens": 1,
                                    "temperature": 0}) as r:
                assert r.status == 200, await r.text()
                await r.read()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        await one(s)  # warm-up
        t0 = time.perf_counter()
        await asyncio.gather(*[one(s) for _ in range(n)])
        return n * length / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--util", type=float, default=0.8, help="target no-cache utilisation")
    ap.add_argument("--sample-k", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=30720)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("[skip] calibration exists:", open(a.out).read())
        return
    run_args = argparse.Namespace(config="gpu", model=a.model, port=a.port, max_model_len=32768,
                                  gpu_mem_util=0.85, gpu_blocks=6144, max_num_seqs=32,
                                  startup_timeout=900, cpu_chunks=0)
    d = os.path.join(os.path.dirname(a.out), "calibration_server")
    os.makedirs(d, exist_ok=True)
    p = start_server(run_args, d)
    try:
        thr = {L: asyncio.run(burst(a.port, a.model, 24, L, 8)) for L in (4096, 12288)}
    finally:
        stop_server(p)
    tok_s = float(np.mean(list(thr.values())))
    res = {"gpu_prefill_tok_s": {str(k): v for k, v in thr.items()}, "tok_s_used": tok_s,
           "util_target": a.util, "speedup": {}}
    for t in a.traces:
        s = stats(prepare(t, a.sample_k, a.max_prompt))
        sp = a.util * tok_s * s["duration_s"] / s["prompt_tokens"]
        res["speedup"][os.path.basename(t).split("_")[0]] = round(sp, 3)
        res[f"replay_minutes_{os.path.basename(t).split('_')[0]}"] = round(s["duration_s"] / sp / 60, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
