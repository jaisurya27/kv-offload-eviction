"""Simulator predictions for exactly the GPU experiment's prepared traces and
capacities (CPU-tier hit rate; GPU tier is inclusive and much smaller)."""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:  # on Colab the real vLLM is installed; locally fall back to the vendored copy
    import vllm.v1.kv_offload.cpu.policies.lru  # noqa: F401
except Exception:  # noqa: BLE001
    sys.path.insert(0, os.path.join(ROOT, "vllm_stub"))
import kvsim.tier_sim as T
from bench.prep import prepare, stats

SIM = {"lru": "lru", "arc": "arc", "adm2": "lru+adm2", "dip": "dip", "pg": "pg", "lhd": "lhd",
       "oracle": "belady"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cpu-chunks", type=int, required=True)
    ap.add_argument("--sample-k", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=30720)
    ap.add_argument("--configs", default="lru,arc,adm2,dip,pg,oracle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reqs = prepare(a.trace, a.sample_k, a.max_prompt)
    res = {"trace": os.path.basename(a.trace), "cpu_chunks": a.cpu_chunks, **stats(reqs)}
    for c in a.configs.split(","):
        p, thr = T.make(SIM[c], a.cpu_chunks, reqs)
        res[c] = round(T.replay(reqs, p, a.cpu_chunks, thr), 4)
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
