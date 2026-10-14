"""DIP-KV: adaptive-insertion LRU for vLLM's CPU KV-offload tier.

Out-of-tree vLLM CachePolicy (vLLM >= 0.30), loaded with
  kv_connector_extra_config = {"eviction_policy": "DIPCachePolicy",
                               "cache_policy_module_path": "kvpolicy.dip"}
(also: "PGCachePolicy" = always probation+ghost, for ablation).

Why: ~85% of KV chunks written to the offload tier are never read again
(Mooncake traces). vLLM's store_threshold=2 admission filter fixes the pollution
at small capacity but loses the first reuse of every chunk and hurts at large
capacity. Instead, keep every chunk but insert new ones on *probation*:

  * PG mode   - probation + ghost: a never-seen chunk enters at the cold
                (evict-next) end of the LRU list (BIP, Qureshi et al., ISCA'07);
                its key is remembered in a ghost list (4x capacity, keys only).
                A chunk whose key is in the ghost list, or that is reused while
                cached, is treated as hot. Unlike vLLM's store_threshold=2, the
                first sighting is still cached, so a quick first reuse can hit.
  * LRU mode  - new chunks enter at the hot end (vLLM default behaviour).
  * Adaptive  - two small shadow caches (keys only) replay a hash-sampled 1/S of
                the key stream under each mode with capacity C/S (SHARDS-style
                spatial sampling); the live cache follows whichever mode has
                more recent shadow hits.

vLLM call-pattern details handled:
  * touch() is called again after a store; only a request's first sighting of a
    key counts as reuse (otherwise every new chunk would be promoted at once -
    the same issue that collapses vLLM's ARC policy to LRU).
  * Within a request, deeper chunks are kept colder than shallower ones (keys
    are processed in reverse prefix order, as in vLLM's LRUCachePolicy), so
    prefix chains are evicted tail-first.
"""
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus

SAMPLE = 8           # shadow caches see 1/SAMPLE of keys
DECAY_EVERY = 4096   # halve shadow hit counters every N sampled accesses


def _h(key) -> int:
    if isinstance(key, (bytes, bytearray)):
        return int.from_bytes(bytes(key[:8]), "little")
    return hash(key) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF


GHOST_MULT = 4       # ghost list remembers 4x capacity worth of keys (keys only)


class _Ghost:
    """Bounded FIFO of keys recently inserted on probation."""
    def __init__(self, cap: int):
        self.cap = max(cap, 16)
        self.q: OrderedDict = OrderedDict()

    def check_and_record(self, k) -> bool:
        """True if k was seen recently (-> admit hot); else remember it."""
        if k in self.q:
            del self.q[k]
            return True
        self.q[k] = None
        if len(self.q) > self.cap:
            self.q.popitem(last=False)
        return False


class _Shadow:
    """Key-only LRU directory with probation+ghost (PG) or MRU (LRU) insertion."""
    def __init__(self, cap: int, bip: bool):
        self.cap, self.bip = max(cap, 4), bip
        self.q: OrderedDict = OrderedDict()   # last = hot
        self.ghost = _Ghost(GHOST_MULT * self.cap)
        self.hits = 0.0
        self.n_ins = 0

    def access(self, keys_rev: list):
        new = []
        for k in keys_rev:
            if k in self.q:
                self.hits += 1
                self.q.move_to_end(k)
            else:
                new.append(k)
        for k in new:
            self.n_ins += 1
            self.q[k] = None
            if self.bip and not self.ghost.check_and_record(k):
                self.q.move_to_end(k, last=False)
        while len(self.q) > self.cap:
            self.q.popitem(last=False)


class _InsertionLRU(CachePolicy):
    """LRU list with configurable insertion point; base for BIP and DIP."""
    adaptive = False
    fixed_bip = False

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self.evictable: OrderedDict[OffloadKey, None] = OrderedDict()  # first = evict next
        self.req_seen: OrderedDict[str, set] = OrderedDict()
        self.n_ins = 0
        self.unplaced: set[OffloadKey] = set()  # inserted, store still in flight
        self.ghost = _Ghost(GHOST_MULT * cache_capacity)
        self.shadow_lru = _Shadow(cache_capacity // SAMPLE, bip=False)
        self.shadow_bip = _Shadow(cache_capacity // SAMPLE, bip=True)
        self.n_sampled = 0
        self.stats = {"hits": 0, "mode_bip_inserts": 0, "mode_lru_inserts": 0}

    def _bip_mode(self) -> bool:
        if not self.adaptive:
            return self.fixed_bip
        return self.shadow_bip.hits > self.shadow_lru.hits

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        if chunk.ref_cnt == 0:
            self._place_new(key)
        else:
            self.unplaced.add(key)  # placed when the store completes

    def _place_new(self, key):
        self.n_ins += 1
        self.evictable[key] = None
        if self._bip_mode() and not self.ghost.check_and_record(key):
            # probation: cold end. Keys of one request arrive in prefix order,
            # so each later (deeper) key ends up colder than its parent.
            # A key seen recently (ghost hit) skips probation and enters hot.
            self.evictable.move_to_end(key, last=False)
            self.stats["mode_bip_inserts"] += 1
        else:
            self.stats["mode_lru_inserts"] += 1

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self.evictable.pop(key, None)
        self.unplaced.discard(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        keys = list(keys)
        rid = req_context.req_id
        first_sighting = rid not in self.req_seen
        seen = self.req_seen.setdefault(rid, set())
        if first_sighting and len(self.req_seen) > 4096:
            self.req_seen.popitem(last=False)
        if first_sighting and self.adaptive:
            sampled = [k for k in reversed(keys) if _h(k) % SAMPLE == 0]
            if sampled:
                self.shadow_lru.access(sampled)
                self.shadow_bip.access(sampled)
                self.n_sampled += len(sampled)
                if self.n_sampled >= DECAY_EVERY:
                    self.n_sampled = 0
                    self.shadow_lru.hits *= 0.5
                    self.shadow_bip.hits *= 0.5
        for k in reversed(keys):
            if k in seen:
                continue
            seen.add(k)
            if k in self.chunks:
                self.stats["hits"] += 1
                if k in self.evictable:
                    self.evictable.move_to_end(k)

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self.evictable.clear()
        self.unplaced.clear()

    @override
    def evict(self, n: int, protected: set[OffloadKey]):
        if n == 0:
            return []
        cand = []
        for k in self.evictable:
            if k in protected:
                continue
            cand.append(k)
            if len(cand) == n:
                break
        if len(cand) < n:
            return None
        out = []
        for k in cand:
            del self.evictable[k]
            out.append((k, self.chunks.pop(k)))
        return out

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # store completion (ref -1 -> 0) or load completion (1 -> 0)
        if key in self.unplaced:
            self.unplaced.discard(key)
            self._place_new(key)
        elif key in self.chunks and key not in self.evictable:
            self.evictable[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable.pop(key, None)


class PGCachePolicy(_InsertionLRU):
    """Always probation+ghost (ablation: DIP without the adaptive switch)."""
    fixed_bip = True


class DIPCachePolicy(_InsertionLRU):
    adaptive = True
