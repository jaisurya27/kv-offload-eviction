"""Eviction policies. All evict only leaf blocks (see cache.py)."""
import heapq
from collections import defaultdict
from .cache import HeapPolicy


class LRU(HeapPolicy):
    """vLLM APC / SGLang radix default."""
    name = "LRU"

    def __init__(self):
        super().__init__()
        self.last = {}

    def priority(self, b):
        return self.last[b]

    def on_insert(self, b, now, i): self.last[b] = now
    def on_hit(self, b, now, i): self.last[b] = now

    def on_remove(self, b):
        super().on_remove(b)
        self.last.pop(b, None)


class TTL(LRU):
    """Expire any block idle longer than ttl (ms); LRU when full.
    TTL frees memory proactively; in a shared KV pool that memory goes to running requests."""
    def __init__(self, ttl_ms):
        super().__init__()
        self.ttl = ttl_ms
        self.name = f"TTL({ttl_ms/1000:g}s)"

    def expire(self, now):
        c = self.cache
        while self.heap:
            prio, nd, v, b = self.heap[0]
            if not self._valid(self.heap[0]):
                heapq.heappop(self.heap)
                continue
            if prio < now - self.ttl:
                heapq.heappop(self.heap)
                c.remove(b)
            else:
                break


class LFUAged(HeapPolicy):
    """LFU with dynamic aging (priority = L + freq), L = priority of last victim."""
    name = "LFU-aged"

    def __init__(self):
        super().__init__()
        self.freq = defaultdict(int)
        self.base = {}
        self.L = 0.0

    def priority(self, b):
        return self.base[b] + self.freq[b]

    def on_insert(self, b, now, i):
        self.freq[b] = 1
        self.base[b] = self.L

    def on_hit(self, b, now, i):
        self.freq[b] += 1
        self.base[b] = self.L

    def victim(self, protect):
        b = super().victim(protect)
        if b is not None:
            self.L = self.priority(b)
        return b

    def on_remove(self, b):
        super().on_remove(b)
        self.freq.pop(b, None)
        self.base.pop(b, None)


class GDSF(LFUAged):
    """Cost-aware Greedy-Dual-Size-Frequency.
    Recomputing block at depth d costs linear MLP work + attention over d*512 prior tokens,
    so deep blocks are pricier to lose: cost(d) = 1 + alpha*d. Priority = L + freq*cost."""
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.name = f"GDSF(a={alpha})"

    def priority(self, b):
        return self.base[b] + self.freq[b] * (1 + self.alpha * self.cache.depth[b])


class SLRU(HeapPolicy):
    """Segmented LRU (hybrid recency+frequency, cf. S3-FIFO / SAECache multi-queue idea):
    blocks seen once live in a probation segment and are evicted first; blocks reused
    >=2 times are promoted to a protected segment capped at `frac` of capacity."""
    def __init__(self, frac=0.8):
        super().__init__()
        self.frac = frac
        self.name = f"SLRU({frac})"
        self.last = {}
        self.hits = defaultdict(int)
        self.nprot = 0

    def _prot(self, b):
        return self.hits[b] >= 1

    def priority(self, b):
        # probation (0) always below protected (1) unless protected segment over cap
        over = self.nprot > self.frac * self.cache.cap
        tier = 0 if (not self._prot(b) or over) else 1
        return (tier, self.last[b])

    def on_insert(self, b, now, i): self.last[b] = now

    def on_hit(self, b, now, i):
        if not self._prot(b):
            self.nprot += 1
        self.hits[b] += 1
        self.last[b] = now

    def on_remove(self, b):
        super().on_remove(b)
        if self._prot(b):
            self.nprot -= 1
        self.last.pop(b, None)
        self.hits.pop(b, None)


class Belady(HeapPolicy):
    """Offline optimal (evict block whose next use is farthest). Upper bound, needs the future."""
    name = "Belady (oracle)"

    def __init__(self, uses):
        super().__init__()
        self.uses = uses          # block -> sorted list of request indices
        self.ptr = defaultdict(int)
        self.now_i = 0

    def _next(self, b):
        u = self.uses[b]
        p = self.ptr[b]
        while p < len(u) and u[p] <= self.now_i:
            p += 1
        self.ptr[b] = p
        return u[p] if p < len(u) else float("inf")

    def priority(self, b):
        return -self._next(b)     # farthest next use => smallest priority => evicted

    def on_insert(self, b, now, i): self.now_i = i
    def on_hit(self, b, now, i): self.now_i = i



def build_uses(reqs):
    uses = defaultdict(list)
    for i, r in enumerate(reqs):
        for b in r["hash_ids"]:
            uses[b].append(i)
    return uses
