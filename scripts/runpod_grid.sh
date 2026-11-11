#!/usr/bin/env bash
# ARC comparison on a rented GPU (RunPod or any Linux box with an A100 80GB).
# Run from the repo root inside tmux or with nohup, so it survives a closed terminal:
#   nohup bash scripts/runpod_grid.sh > /workspace/grid.log 2>&1 &
#   tail -f /workspace/grid.log
# Finished runs are skipped, so it is safe to start again after a crash.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-/workspace/results}
mkdir -p "$OUT"

pip install -q vllm==0.30.0 aiohttp pandas matplotlib
[ -f traces/conversation_trace.jsonl ] || bash scripts_get_traces.sh
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"

# Smoke test: 80 requests with the fixed ARC policy. Stops here if the server does not load it.
if [ ! -f "$OUT/smoke_arcfix/summary.json" ]; then
  python bench/run.py --trace traces/conversation_trace.jsonl --config arcfix \
    --cpu-chunks 500 --limit 80 --speedup 20 --run-name smoke_arcfix --out-root "$OUT"
fi
python - "$OUT" <<'EOF' || exit 1
import json, sys
s = json.load(open(f"{sys.argv[1]}/smoke_arcfix/summary.json"))
print({k: s[k] for k in ["policy_loaded", "n_ok", "n_errors", "hit_rate_cpu_tokens", "ttft_ms_p50"]})
assert s["n_errors"] == 0 and s["policy_loaded"] == "ARCFixProbe", "smoke test failed, see server.log"
EOF

# Same replay speeds as the Colab A100 runs. LRU and DIP-KV are rerun here so that
# every policy in the comparison ran on the same machine.
declare -A SPEED=([conversation]=2.62 [toolagent]=3.416)
for t in conversation toolagent; do
  for c in arc arcfix lru dip; do
    name="${t}_${c}_c500_x${SPEED[$t]}_rp"
    if [ -f "$OUT/$name/summary.json" ]; then echo "skip $name"; continue; fi
    echo "=== $(date +%H:%M) $name"
    python bench/run.py --trace "traces/${t}_trace.jsonl" --config "$c" --cpu-chunks 500 \
      --speedup "${SPEED[$t]}" --run-name "$name" --out-root "$OUT" \
      || echo "!! $name failed, see $OUT/$name/server.log"
  done
done

python - "$OUT" <<'EOF'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/*_rp/summary.json")):
    s = json.load(open(f))
    print(f"{f.split('/')[-2]:42s} hit_gpu={s['hit_rate_gpu_tokens']:.3f} "
          f"hit_cpu={s['hit_rate_cpu_tokens']:.3f} p50={s['ttft_ms_p50']:.0f} p95={s['ttft_ms_p95']:.0f}")
EOF
echo "ALL DONE"
