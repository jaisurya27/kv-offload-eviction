"""Replay a Mooncake-format trace through the prefix cache and a prefill-latency model.

TTFT model (ANALYTICAL — to be calibrated against real vLLM runs):
  Llama-3.1-8B on H100: per new token, 2*8e9 FLOPs (dense) + 4*L*d*position FLOPs (attention),
  at 400 TFLOP/s effective, + 5 ms fixed overhead. Cached tokens cost nothing to prefill.
  M prefill workers share one global KV pool (Mooncake/LMCache-style), FIFO queue.
  M is chosen so the *no-cache* system runs at `util` utilisation at 1x trace speed.
"""
import argparse, csv, heapq, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvsim.cache import PrefixCache
from kvsim.policies import LRU, TTL, LFUAged, GDSF, SLRU, Belady, build_uses

BLOCK = 512
DENSE = 2 * 8e9
ATTN = 4 * 32 * 4096
FLOPS = 4e14
OVERHEAD = 0.005


def prefill_s(cached_tok, new_tok):
    if new_tok <= 0:
        return OVERHEAD
    return OVERHEAD + (new_tok * DENSE + ATTN * new_tok * (cached_tok + new_tok / 2)) / FLOPS


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def pick_workers(reqs, util, speed):
    work = sum(prefill_s(0, r["input_length"]) for r in reqs)
    dur = (reqs[-1]["timestamp"] - reqs[0]["timestamp"]) / 1000 / speed
    return max(1, int(np.ceil(work / dur / util)))


def simulate(reqs, policy, cap, workers, speed):
    cache = PrefixCache(cap, policy) if policy is not None else None
    free_at = [0.0] * workers
    heapq.heapify(free_at)
    ttft, hit_tok, tot_tok, req_hit = [], 0, 0, 0
    for i, r in enumerate(reqs):
        t = r["timestamp"] / 1000 / speed
        L = r["input_length"]
        if cache is not None:
            hb = cache.access(r["hash_ids"], r["timestamp"] / speed, i)
            cached = min(hb * BLOCK, L - 1)  # last token always recomputed
        else:
            cached = 0
        w = heapq.heappop(free_at)
        start = max(t, w)
        done = start + prefill_s(cached, L - cached)
        heapq.heappush(free_at, done)
        ttft.append(done - t)
        hit_tok += cached
        tot_tok += L
        req_hit += cached > 0
    a = np.array(ttft) * 1000
    return dict(token_hit=hit_tok / tot_tok, req_hit=req_hit / len(reqs),
                p50=np.percentile(a, 50), p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                mean=a.mean())


def policies(reqs, which):
    out = []
    for name in which:
        if name == "none": out.append(lambda: None)
        elif name == "lru": out.append(lambda: LRU())
        elif name.startswith("ttl"): out.append(lambda s=float(name[3:]): TTL(s * 1000))
        elif name == "lfu": out.append(lambda: LFUAged())
        elif name == "gdsf": out.append(lambda: GDSF(0.05))
        elif name == "slru": out.append(lambda: SLRU(0.8))
        elif name == "belady":
            uses = build_uses(reqs)
            out.append(lambda u=uses: Belady(u))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--caps", default="1000,4000,16000")
    ap.add_argument("--speeds", default="1")
    ap.add_argument("--policies", default="none,lru,ttl60,ttl300,lfu,gdsf,slru,belady")
    ap.add_argument("--util", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = load(a.trace)
    names = a.policies.split(",")
    facts = policies(reqs, names)
    rows = []
    for speed in map(float, a.speeds.split(",")):
        M = pick_workers(reqs, a.util, 1.0)   # fixed fleet; speed>1 = burstier/heavier load
        for cap in map(int, a.caps.split(",")):
            for n, f in zip(names, facts):
                if n == "none" and cap != int(a.caps.split(",")[0]):
                    continue
                t0 = time.time()
                p = f()
                res = simulate(reqs, p, cap, M, speed)
                row = dict(trace=os.path.basename(a.trace).split("_")[0], speed=speed, workers=M,
                           cap_blocks=cap if p else 0, policy=p.name if p else "no-cache", **res)
                rows.append(row)
                print(f"{row['trace']:12s} x{speed:<4g} M={M:<3d} cap={cap:<6d} {row['policy']:16s} "
                      f"tokhit={res['token_hit']:.3f} p50={res['p50']:8.0f} p95={res['p95']:9.0f} ms "
                      f"({time.time()-t0:.1f}s)", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
