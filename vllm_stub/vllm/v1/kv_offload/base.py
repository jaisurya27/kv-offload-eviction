# Minimal stub of vllm/v1/kv_offload/base.py (v0.30.0) so vLLM's real CPU-offload
# CachePolicy classes (copied verbatim next to this file) run in the offline simulator.
from dataclasses import dataclass, field
from typing import Any, NewType
OffloadKey = NewType("OffloadKey", bytes)

@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict[str, Any] | None = None
    _state: dict = field(default_factory=dict, repr=False, init=False)
