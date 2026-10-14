"""Replay Mooncake traces through vLLM 0.30's *real* CPUOffloadingManager.

Checks that:
  1. our out-of-tree policies load through vLLM's official hook
     (eviction_policy + cache_policy_module_path),
  2. the manager's own invariants/assertions hold under our policies,
  3. hit rates match kvsim/tier_sim.py (so the simulator is faithful),
  4. vLLM's ARC policy really collapses to LRU (T1 stays empty).

Run: VLLM_SITE=<site-packages of vllm 0.30> python tests/test_real_manager.py
"""
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import real_vllm_harness as H

H.setup()
M = H.load("vllm.v1.kv_offload.cpu.manager")
B = sys.modules["vllm.v1.kv_offload.base"]
sys.path.insert(0, ROOT)  # so vLLM can import kvpolicy.* by module path

BLOCK = 512


def key(h: int):
    return B.make_offload_key(h.to_bytes(16, "little"), 0)


def replay(reqs, policy="lru", module=None, store_threshold=0, cap=1000):
    mgr = M.CPUOffloadingManager(num_chunks=cap, cache_policy=policy,
                                 cache_policy_module_path=module,
                                 store_threshold=store_threshold)
    hit_tok = tot = 0
    for i, r in enumerate(reqs):
        L = r["input_length"]
        keys = [key(h) for h in r["hash_ids"][: L // BLOCK]]
        ctx = B.ReqContext(req_id=f"r{i}")
        mgr.on_new_request(ctx)
        n = 0
        for k in keys:
            if mgr.lookup(k, ctx) == B.LookupResult.HIT:
                n += 1
            else:
                break
        mgr.touch(keys, ctx)
        if n:
            mgr.prepare_load(keys[:n], ctx)
            mgr.complete_load(keys[:n], ctx)
        out = mgr.prepare_store(keys[n:], ctx)
        if out is not None and out.keys_to_store:
            mgr.complete_store(out.keys_to_store, ctx)
        mgr.touch(keys, ctx)
        mgr.on_request_finished(ctx)
        hit_tok += min(n * BLOCK, L - 1)
        tot += L
    return hit_tok / tot, mgr


if __name__ == "__main__":
    import kvsim.tier_sim as T
    sys.path.insert(0, os.path.join(ROOT, "vllm_stub"))  # sim uses the vendored copy
    reqs = [json.loads(l) for l in open(os.path.join(ROOT, "traces/conversation_trace.jsonl"))]
    configs = [("lru", None, 0, "lru"), ("arc", None, 0, "arc"), ("lru", None, 2, "lru+adm2"),
               ("DIPCachePolicy", "kvpolicy.dip", 0, "dip"),
               ("PGCachePolicy", "kvpolicy.dip", 0, "pg"),
               ("LHDCachePolicy", "kvpolicy.lhd", 0, "lhd")]
    ok = True
    for cap in (1000, 4000):
        for pol, mod, thr, simname in configs:
            t0 = time.time()
            real, mgr = replay(reqs, pol, mod, thr, cap)
            p, th = T.make(simname, cap, reqs)
            sim = T.replay(reqs, p, cap, th)
            extra = ""
            if pol == "arc":
                pp = mgr._policy
                extra = f" | ARC |T1|={len(pp.t1)} |T2|={len(pp.t2)} target_t1={pp.target_t1_size:.0f}"
            flag = "OK " if abs(real - sim) < 0.005 else "DIFF"
            ok &= flag == "OK "
            print(f"{flag} cap={cap:<5d} {simname:9s} real-vLLM={real:.4f} sim={sim:.4f} "
                  f"({time.time()-t0:.0f}s){extra}", flush=True)
    print("ALL MATCH" if ok else "MISMATCHES FOUND")
