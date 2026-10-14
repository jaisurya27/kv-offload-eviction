"""Offline replay of vLLM's CPU KV-offload tier with *real* vLLM CachePolicy classes.

Emulates CPUOffloadingManager + OffloadingConnectorScheduler call order (vLLM 0.30):
  lookup(prefix) -> touch(all keys) -> [admission] -> evict(n, protected=req keys)
  -> insert(new keys) -> touch(all keys)
Chunk = one 512-token trace block (set kv_connector_extra_config.block_size=512 on GPU).
Only full chunks are stored. The CPU tier is write-through/inclusive, so its hit
rate approximates the combined GPU+CPU prefix hit rate.
"""
import argparse, csv, heapq, json, os, sys, time
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vllm_stub"))
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from kvpolicy.lhd import LHDCachePolicy
from kvpolicy.dip import PGCachePolicy, DIPCachePolicy

BLOCK = 512


class BeladyPolicy(CachePolicy):
    """Offline optimal for a flat key cache: evict the key reused farthest in the
    future (needs the whole trace; upper bound only). Lazy max-heap on next use."""
    def __init__(self, cache_capacity, uses):
        super().__init__(cache_capacity)
        self.uses, self.ptr, self.now = uses, defaultdict(int), 0
        self.chunks, self.heap, self.ver = {}, [], defaultdict(int)

    def _next(self, k):
        u, p = self.uses[k], self.ptr[k]
        while p < len(u) and u[p] <= self.now:
            p += 1
        self.ptr[k] = p
        return u[p] if p < len(u) else 1 << 60

    def _push(self, k):
        self.ver[k] += 1
        heapq.heappush(self.heap, (-self._next(k), self.ver[k], k))

    def get(self, k): return self.chunks.get(k)
    def insert(self, k, c): self.chunks[k] = c; self._push(k)
    def remove(self, k): self.chunks.pop(k, None); self.ver[k] += 1
    def clear(self): self.chunks.clear(); self.heap.clear()

    def touch(self, keys, ctx):
        self.now = int(ctx.req_id)
        for k in keys:
            if k in self.chunks:
                self._push(k)

    def evict(self, n, protected):
        out, held = [], []
        while len(out) < n and self.heap:
            e = heapq.heappop(self.heap)
            _, v, k = e
            if self.ver[k] != v or k not in self.chunks:
                continue
            if k in protected:
                held.append(e); continue
            out.append((k, self.chunks.pop(k))); self.ver[k] += 1
        for e in held:
            heapq.heappush(self.heap, e)
        if len(out) < n:  # cannot satisfy: undo (atomic contract)
            for k, c in out:
                self.chunks[k] = c; self._push(k)
            return None
        return out


def replay(reqs, policy, cap, store_threshold=0, max_tracker=64000):
    counts = OrderedDict() if store_threshold >= 2 else None
    free = cap
    hit_tok = tot_tok = 0
    for i, r in enumerate(reqs):
        L = r["input_length"]
        keys = r["hash_ids"][: L // BLOCK]
        ctx = ReqContext(req_id=str(i))
        h = 0
        for k in keys:
            c = policy.get(k)
            if c is not None and c.is_ready:
                h += 1
            else:
                break
        policy.touch(keys, ctx)
        hit_tok += min(h * BLOCK, L - 1)
        tot_tok += L
        new = [k for k in keys if policy.get(k) is None]
        if counts is not None:  # vLLM store_threshold admission tracker
            adm = []
            for k in new:
                counts[k] = counts.get(k, 0) + 1
                counts.move_to_end(k)
                if counts[k] >= store_threshold:
                    adm.append(k)
            while len(counts) > max_tracker:
                counts.popitem(last=False)
            new = adm
        if not new:
            continue
        need = len(new) - free
        if need > 0:
            ev = policy.evict(need, set(keys))
            if ev is None:
                continue  # vLLM: store skipped (allocation failure)
            free += len(ev)
        chunks = []
        for k in new:
            c = ChunkStatus(0)          # ref_cnt = -1: store in flight
            policy.insert(k, c)
            chunks.append((k, c))
            free -= 1
        for k, c in chunks:             # complete_store: -1 -> 0, now evictable
            c.ref_cnt = 0
            policy.mark_evictable(k)
        policy.touch(keys, ctx)
    return hit_tok / tot_tok


def make(name, cap, reqs):
    if name == "lru": return LRUCachePolicy(cap), 0
    if name == "arc": return ARCCachePolicy(cap), 0
    if name == "lru+adm2": return LRUCachePolicy(cap), 2
    if name == "lhd": return LHDCachePolicy(cap), 0
    if name == "pg": return PGCachePolicy(cap), 0
    if name == "dip": return DIPCachePolicy(cap), 0
    if name == "belady":
        uses = defaultdict(list)
        for i, r in enumerate(reqs):
            for k in r["hash_ids"][: r["input_length"] // BLOCK]:
                uses[k].append(i)
        return BeladyPolicy(cap, uses), 0
    raise ValueError(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=0, help="first N requests (0=all)")
    ap.add_argument("--caps", default="1000,2000,4000,8000,16000")
    ap.add_argument("--policies", default="lru,arc,lru+adm2,lhd,pg,dip,belady")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(a.trace)]
    if a.n:
        reqs = reqs[: a.n]
    rows = []
    name = os.path.basename(a.trace).split("_")[0]
    for cap in map(int, a.caps.split(",")):
        line = []
        for p in a.policies.split(","):
            t0 = time.time()
            pol, thr = make(p, cap, reqs)
            hr = replay(reqs, pol, cap, thr)
            rows.append(dict(trace=name, n=len(reqs), cap_chunks=cap, cap_Mtok=cap * BLOCK / 1e6,
                             policy=p, token_hit=round(hr, 4)))
            line.append(f"{p}={hr:.3f}({time.time()-t0:.0f}s)")
        print(f"{name} cap={cap:<6d}", " ".join(line), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
