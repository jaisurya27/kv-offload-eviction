"""Logging wrappers so every GPU run proves which policy actually ran inside the
vLLM engine process (and with what capacity). Writes JSON lines to $KVPOLICY_LOG.

Use as eviction_policy = "LRUProbe" | "ARCProbe" | "DIPProbe" | "PGProbe" | "LHDProbe"
with cache_policy_module_path = "kvpolicy.probe".
"""
import json
import os
import time

from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from kvpolicy.dip import DIPCachePolicy, PGCachePolicy
from kvpolicy.lhd import LHDCachePolicy

EVERY = 2000


def _probe(base, label):
    class Probe(base):
        def __init__(self, cache_capacity):
            super().__init__(cache_capacity)
            self._n_evict_calls = 0
            self._n_evicted = 0
            self._log({"event": "init", "policy": label, "capacity_chunks": cache_capacity,
                       "pid": os.getpid()})

        def _log(self, rec):
            path = os.environ.get("KVPOLICY_LOG")
            if not path:
                return
            rec["t"] = time.time()
            extra = getattr(self, "stats", None)
            if isinstance(extra, dict):
                rec["stats"] = dict(extra)
            if hasattr(self, "t1"):
                rec["arc_t1"] = len(self.t1)
                rec["arc_t2"] = len(self.t2)
            if hasattr(self, "shadow_lru"):
                rec["shadow_hits_lru"] = round(self.shadow_lru.hits, 1)
                rec["shadow_hits_pg"] = round(self.shadow_bip.hits, 1)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        def evict(self, n, protected):
            out = super().evict(n, protected)
            self._n_evict_calls += 1
            if out:
                self._n_evicted += len(out)
            if self._n_evict_calls % EVERY == 0:
                self._log({"event": "periodic", "policy": label,
                           "evict_calls": self._n_evict_calls, "evicted": self._n_evicted})
            return out

    Probe.__name__ = label
    return Probe


LRUProbe = _probe(LRUCachePolicy, "LRUProbe")
ARCProbe = _probe(ARCCachePolicy, "ARCProbe")
DIPProbe = _probe(DIPCachePolicy, "DIPProbe")
PGProbe = _probe(PGCachePolicy, "PGProbe")
LHDProbe = _probe(LHDCachePolicy, "LHDProbe")
