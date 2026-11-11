"""vLLM's ARC policy with one change: textbook handling of ghost hits.

In ARC (Megiddo and Modha, FAST'03), a requested key that is found in a ghost
list (B1 or B2) is being referenced for the second time, so the chunk goes into
T2, the "frequent" list. vLLM 0.30's ARCCachePolicy adapts its target size on
such a hit but then inserts the chunk into T1, so T2 stays nearly empty when
reuse arrives after eviction (the common case for prefix KV caches).

Load in vLLM with
  kv_connector_extra_config = {"eviction_policy": "TextbookARCCachePolicy",
                               "cache_policy_module_path": "kvpolicy.arc_fix"}
"""
from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus


class TextbookARCCachePolicy(ARCCachePolicy):
    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        if key in self.b1 or key in self.b2:
            # ghost hit: seen recently, this is its second reference
            self.b1.pop(key, None)
            self.b2.pop(key, None)
            self.t2[key] = chunk
        else:
            super().insert(key, chunk)
