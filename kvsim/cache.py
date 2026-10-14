"""Radix-tree block prefix cache with pluggable eviction (vLLM/SGLang semantics).

Each block = 512 tokens of KV, identified by a prefix-chained hash (Mooncake trace format):
block i of a request is only reusable if blocks 0..i-1 are also cached. So, like
SGLang's RadixAttention and vLLM's APC, only *leaf* blocks (no cached children) are evictable.
"""
import heapq
from collections import defaultdict


class PrefixCache:
    def __init__(self, capacity_blocks, policy):
        self.cap = capacity_blocks
        self.policy = policy
        policy.cache = self
        self.parent = {}                    # block -> parent block (None for root blocks)
        self.nchild = defaultdict(int)      # cached children count
        self.depth = {}
        self.size = 0

    def __contains__(self, b):
        return b in self.parent

    def match(self, hash_ids):
        """Longest cached prefix, in blocks."""
        n = 0
        for b in hash_ids:
            if b in self.parent:
                n += 1
            else:
                break
        return n

    def access(self, hash_ids, now, req_idx):
        """Serve a request: count hits, then insert/touch the whole path."""
        self.policy.expire(now)
        hit = self.match(hash_ids)
        prev = None
        protect = set(hash_ids)
        for d, b in enumerate(hash_ids):
            if b not in self.parent:
                if self.size >= self.cap and not self._evict_one(now, protect=protect):
                    break  # nothing evictable: stop inserting (rest of request not cached)
                self.parent[b] = prev
                self.depth[b] = d
                if prev is not None:
                    self.nchild[prev] += 1
                self.size += 1
                self.policy.on_insert(b, now, req_idx)
            else:
                self.policy.on_hit(b, now, req_idx)
            prev = b
        # leaf status may have changed along the path; let policy (re)index the tail
        self.policy.after_access(hash_ids, now)
        return hit

    def is_leaf(self, b):
        return b in self.parent and self.nchild.get(b, 0) == 0

    def remove(self, b):
        p = self.parent.pop(b)
        self.depth.pop(b, None)
        self.nchild.pop(b, None)
        self.size -= 1
        if p is not None:
            self.nchild[p] -= 1
            if self.nchild[p] == 0:
                self.policy.became_leaf(p)
        self.policy.on_remove(b)

    def _evict_one(self, now, protect):
        b = self.policy.victim(protect)
        if b is None:
            return False
        self.remove(b)
        return True


class HeapPolicy:
    """Base: lazy min-heap over evictable leaves keyed by priority(b). Lowest priority evicted."""
    name = "base"

    def __init__(self):
        self.heap = []
        self.ver = defaultdict(int)
        self.cache = None

    # --- hooks ---
    def priority(self, b):
        raise NotImplementedError

    def on_insert(self, b, now, i): pass
    def on_hit(self, b, now, i): pass
    def on_remove(self, b): self.ver[b] += 1
    def expire(self, now): pass

    def _push(self, b):
        self.ver[b] += 1
        # tie-break: deeper blocks first (matches vLLM/SGLang behaviour)
        heapq.heappush(self.heap, (self.priority(b), -self.cache.depth[b], self.ver[b], b))

    def became_leaf(self, b):
        self._push(b)

    def after_access(self, hash_ids, now):
        for b in hash_ids:
            if self.cache.is_leaf(b):
                self._push(b)

    def _valid(self, entry):
        _, _, v, b = entry
        return self.ver[b] == v and self.cache.is_leaf(b)

    def victim(self, protect):
        skipped = []
        out = None
        while self.heap:
            e = heapq.heappop(self.heap)
            if not self._valid(e):
                continue
            if e[3] in protect:
                skipped.append(e)
                continue
            if self.priority(e[3]) != e[0]:   # stale priority: re-queue fresh, retry
                self._push(e[3])
                continue
            out = e[3]
            break
        for e in skipped:
            heapq.heappush(self.heap, e)
        return out
