"""LHD-KV: prefix-aware Least-Hit-Density eviction for vLLM's CPU KV-offload tier.

Out-of-tree vLLM CachePolicy (vLLM >= 0.30), loaded with
  kv_connector_extra_config = {"eviction_policy": "LHDCachePolicy",
                               "cache_policy_module_path": "kvpolicy.lhd"}

Idea (Beckmann et al., "LHD: Improving Cache Hit Rate by Maximizing Hit Density",
NSDI'18), adapted to prefix KV chunks:
  * Prefix closure: a chunk is only useful if all its ancestors are cached, so
    only *leaf* chunks (no cached children) are eviction candidates. Parent links
    are learned from touch(), which lists a request's keys in prefix order.
  * Each chunk belongs to a class: (reused before?, request was warm?, depth bucket).
  * Online, per class and age bin, count hits and evictions. Age is measured in
    chunk insertions (cache churn), so it is invariant to request rate.
  * hit density(class, age) = E[future hits] / E[remaining cache lifetime].
    Evict the sampled leaf with the lowest density.

vLLM semantics respected:
  * touch() runs several times per request (after lookup, after each store); only
    a request's first sighting of a key counts as a hit.
  * insert() carries no request context; a new chunk's class and parent come from
    the touch() that preceded its store.
"""
import math
import random
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus

N_AGE = 40                # log-spaced age bins (2 per octave)
DEPTH_EDGES = (2, 8, 24)  # chunk-index buckets: [0,2) [2,8) [8,24) [24,inf)
N_DEPTH = len(DEPTH_EDGES) + 1
N_CLASS = 2 * 2 * N_DEPTH
SAMPLES = 64
DECAY = 0.9


def _age_bin(age: int) -> int:
    return min(N_AGE - 1, int(math.log2(age + 1) * 2))


def _depth_bucket(d: int) -> int:
    for i, e in enumerate(DEPTH_EDGES):
        if d < e:
            return i
    return N_DEPTH - 1


class LHDCachePolicy(CachePolicy):
    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self.meta: dict[OffloadKey, list] = {}      # key -> [class, last_clock, depth]
        self.parent: dict[OffloadKey, OffloadKey | None] = {}
        self.nchild: dict[OffloadKey, int] = {}
        self.evictable: set[OffloadKey] = set()
        # candidates = evictable leaves; list + index for O(1) random sampling
        self.cand: list[OffloadKey] = []
        self.cand_pos: dict[OffloadKey, int] = {}
        self.clock = 0
        self.pending: OrderedDict[OffloadKey, tuple] = OrderedDict()  # key -> (warm, depth, parent)
        self.req_seen: OrderedDict[str, set] = OrderedDict()
        self.req_warm: dict[str, int] = {}
        self.hits = [[0.0] * N_AGE for _ in range(N_CLASS)]
        self.evicts = [[0.0] * N_AGE for _ in range(N_CLASS)]
        self.density = [[1.0] * N_AGE for _ in range(N_CLASS)]
        self.next_recompute = max(256, cache_capacity // 4)
        self.rng = random.Random(0)
        self.stats = {"hits": 0, "evictions": 0, "non_leaf_evictions": 0}

    # ---------- candidate-set maintenance ----------
    @staticmethod
    def _cls(reused: int, warm: int, depth: int) -> int:
        return (reused * 2 + warm) * N_DEPTH + _depth_bucket(depth)

    def _cand_add(self, k):
        if k not in self.cand_pos:
            self.cand_pos[k] = len(self.cand)
            self.cand.append(k)

    def _cand_del(self, k):
        i = self.cand_pos.pop(k, None)
        if i is None:
            return
        last = self.cand.pop()
        if i < len(self.cand):
            self.cand[i] = last
            self.cand_pos[last] = i

    def _refresh(self, k):
        if k in self.evictable and self.nchild.get(k, 0) == 0:
            self._cand_add(k)
        else:
            self._cand_del(k)

    def _drop(self, k):
        """Remove all bookkeeping for k (eviction or explicit remove)."""
        self.chunks.pop(k, None)
        self.meta.pop(k, None)
        self.evictable.discard(k)
        self._cand_del(k)
        self.nchild.pop(k, None)
        p = self.parent.pop(k, None)
        if p is not None and p in self.chunks:
            self.nchild[p] -= 1
            self._refresh(p)

    # ---------- learning ----------
    def _recompute(self):
        """density(c,a) = E[hits | age>=a] / E[remaining lifetime | age>=a].
        Bin b covers ages [lo_b, lo_{b+1}); an event in bin x seen from bin a has
        remaining lifetime ~ mid_x - lo_a. One pseudo-event (global hit rate,
        lifetime = cache size) keeps sparse classes sane."""
        lo = [2 ** (b / 2) - 1 for b in range(N_AGE + 1)]
        mid = [(lo[b] + lo[b + 1]) / 2 for b in range(N_AGE)]
        tot_h = sum(map(sum, self.hits))
        tot_e = sum(map(sum, self.evicts))
        prior = (tot_h + 1) / (tot_h + tot_e + 2)
        for c in range(N_CLASS):
            H, E = self.hits[c], self.evicts[c]
            hs = es = mw = 0.0
            for a in range(N_AGE - 1, -1, -1):
                hs += H[a]
                es += E[a]
                mw += (H[a] + E[a]) * mid[a]
                remaining = mw - (hs + es) * lo[a]
                self.density[c][a] = (hs + prior) / (remaining + self.cache_capacity)
            for a in range(N_AGE):
                H[a] *= DECAY
                E[a] *= DECAY

    def _score(self, k) -> float:
        c, t, _ = self.meta[k]
        return self.density[c][_age_bin(self.clock - t)]

    # ---------- CachePolicy interface ----------
    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.clock += 1
        warm, depth, par = self.pending.pop(key, (0, 0, None))
        self.chunks[key] = chunk
        self.meta[key] = [self._cls(0, warm, depth), self.clock, depth]
        self.nchild.setdefault(key, 0)
        if par is not None and par in self.chunks:
            self.parent[key] = par
            self.nchild[par] = self.nchild.get(par, 0) + 1
            self._refresh(par)
        if chunk.ref_cnt == 0:
            self.evictable.add(key)
        self._refresh(key)
        if self.clock >= self.next_recompute:
            self._recompute()
            self.next_recompute = self.clock + max(256, self.cache_capacity // 4)

    @override
    def remove(self, key: OffloadKey) -> None:
        self._drop(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        keys = list(keys)
        rid = req_context.req_id
        seen = self.req_seen.get(rid)
        if seen is None:
            seen = set()
            self.req_seen[rid] = seen
            if len(self.req_seen) > 4096:
                old, _ = self.req_seen.popitem(last=False)
                self.req_warm.pop(old, None)
            n = 0
            for k in keys:
                if k in self.chunks:
                    n += 1
                else:
                    break
            # warm = request continues a cached prefix beyond its first chunk
            self.req_warm[rid] = 1 if n >= 2 else 0
        warm = self.req_warm.get(rid, 0)
        prev = None
        for depth, k in enumerate(keys):
            if k not in seen:
                seen.add(k)
                m = self.meta.get(k)
                if m is None:
                    self.pending[k] = (warm, depth, prev)
                    if len(self.pending) > 1 << 16:
                        self.pending.popitem(last=False)
                else:
                    c, t, d = m
                    self.hits[c][_age_bin(self.clock - t)] += 1
                    self.stats["hits"] += 1
                    m[0] = self._cls(1, warm, d)
                    m[1] = self.clock
            prev = k

    @override
    def clear(self) -> None:
        for d in (self.chunks, self.meta, self.parent, self.nchild, self.cand_pos, self.pending):
            d.clear()
        self.evictable.clear()
        self.cand.clear()

    @override
    def evict(self, n: int, protected: set[OffloadKey]):
        if n == 0:
            return []
        if len(self.evictable) - len(protected & self.evictable) < n:
            return None
        out = []
        for _ in range(n):
            best, best_s = None, math.inf
            if self.cand:
                for _t in range(SAMPLES * 2):
                    k = self.cand[self.rng.randrange(len(self.cand))]
                    if k in protected:
                        continue
                    s = self._score(k)
                    if s < best_s:
                        best, best_s = k, s
                    if _t >= SAMPLES and best is not None:
                        break
                if best is None:
                    best = next((k for k in self.cand if k not in protected), None)
            if best is None:
                # no evictable leaf (descendants pinned/protected): break prefix rule
                best = next(k for k in self.evictable if k not in protected)
                self.stats["non_leaf_evictions"] += 1
            c, t, _ = self.meta[best]
            self.evicts[c][_age_bin(self.clock - t)] += 1
            self.stats["evictions"] += 1
            chunk = self.chunks[best]
            self._drop(best)
            out.append((best, chunk))
        return out

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        if key in self.chunks:
            self.evictable.add(key)
            self._refresh(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable.discard(key)
        self._cand_del(key)
