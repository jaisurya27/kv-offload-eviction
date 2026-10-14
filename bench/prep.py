"""Deterministic trace preparation shared by the simulator and the GPU replay.

* Session sampling (SHARDS-style): keep a request iff hash(session key) % k == 0,
  where the session key is the first block that is NOT a widely shared prefix
  (system prompt / tool schema). Turns of a kept session stay together, so
  multi-turn reuse is preserved; cache capacity is then scaled by 1/k.
  Validated offline: hit-rate curves of the 1/4 sample at cap/4 track the full
  trace at cap (see README).
* Truncation: prompts are cut to max_tokens (model context), keeping the prefix,
  so prefix-sharing structure is unchanged.
"""
import json
from collections import Counter

BLOCK = 512


def _mix(x: int) -> int:
    return (x * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF


def prepare(path, sample_k=4, max_tokens=30720, shared_pop=50):
    reqs = [json.loads(l) for l in open(path)]
    pop = Counter(b for r in reqs for b in set(r["hash_ids"]))
    out = []
    for r in reqs:
        h = r["hash_ids"]
        skey = next((b for b in h if pop[b] <= shared_pop), h[-1])
        if sample_k > 1 and _mix(skey) % sample_k:
            continue
        L = min(r["input_length"], max_tokens)
        nb = -(-L // BLOCK)
        out.append({"timestamp": r["timestamp"], "input_length": L,
                    "output_length": r["output_length"], "hash_ids": h[:nb]})
    return out


def stats(reqs):
    dur = (reqs[-1]["timestamp"] - reqs[0]["timestamp"]) / 1000
    tok = sum(r["input_length"] for r in reqs)
    return {"requests": len(reqs), "duration_s": dur, "prompt_tokens": tok}
