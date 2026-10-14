"""Generate colab/kv_offload_benchmark.ipynb (kept as code so it stays reviewable)."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})


md(r"""
# KV-cache offload eviction: GPU benchmark (vLLM 0.30, Colab)

Measures **time-to-first-token (TTFT)** and **prefix-cache hit rate** for different eviction
policies in vLLM's **CPU KV-offload tier**, replaying real production traffic
(Mooncake / Kimi traces, FAST'25) with real arrival times.

| config | what runs |
|---|---|
| `nocache` | prefix caching off (recompute everything) |
| `gpu` | vLLM GPU prefix cache only (stock) |
| `lru` | + CPU offload tier, vLLM's LRU (default) |
| `adm2` | + CPU tier, vLLM's `store_threshold=2` admission filter |
| `dip` | + CPU tier, **DIP-KV (ours)**: probation + ghost list, adaptive vs LRU |
| `arc`, `pg` | extended: vLLM's ARC (checks the T1-collapse finding on GPU), DIP without the adaptive switch |

**Before you start:** Runtime → Change runtime type → **A100 GPU** (High-RAM on).
Budget: the core grid takes about **2–3 h on an A100 (~15 compute units)**; the extended grid adds ~1.5 h.
Every run saves to Google Drive and the grid **resumes** where it left off if Colab disconnects:
just re-run the cells. Keep this tab open (Colab Pro has no background execution).
""")

code(r"""
# 1. Hardware check
!nvidia-smi --query-gpu=name,memory.total --format=csv
import psutil, shutil
ram = psutil.virtual_memory().total / 2**30
print(f"System RAM: {ram:.0f} GiB   free disk: {shutil.disk_usage('/content').free/2**30:.0f} GiB")
assert ram > 40, "Need High-RAM runtime (CPU KV tier uses ~8-15 GiB pinned memory)"
""")

code(r"""
# 2. Google Drive for results (survives disconnects)
from google.colab import drive
drive.mount('/content/drive')
RESULTS = '/content/drive/MyDrive/kv-offload-bench'
import os; os.makedirs(RESULTS, exist_ok=True); print(RESULTS)
""")

code(r"""
# 3. Get the code. Push your local repo to GitHub first and set REPO_URL,
#    OR upload a zip of the repo to Drive and set REPO_ZIP.
REPO_URL = 'https://github.com/YOUR_GITHUB_USER/Prefix-KV-LLM-serving.git'
REPO_ZIP = '/content/drive/MyDrive/Prefix-KV-LLM-serving.zip'
REPO = '/content/Prefix-KV-LLM-serving'
import os, subprocess
if not os.path.exists(REPO):
    if 'YOUR_GITHUB_USER' not in REPO_URL:
        subprocess.run(['git', 'clone', REPO_URL, REPO], check=True)
    elif os.path.exists(REPO_ZIP):
        subprocess.run(['unzip', '-q', REPO_ZIP, '-d', '/content'], check=True)
    else:
        raise SystemExit('Set REPO_URL or upload REPO_ZIP')
else:
    subprocess.run(['git', '-C', REPO, 'pull'], check=False)
%cd {REPO}
!ls

def sh(*args):
    # run a command from the repo, streaming output; raise on failure
    import subprocess
    r = subprocess.run([str(a) for a in args], cwd=REPO)
    if r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {args}')
""")

code(r"""
# 4. Install vLLM 0.30.0 (pinned: the policy plugin API is from this version). ~5 min.
#    vLLM runs as a subprocess, so no runtime restart is needed.
!pip install -q vllm==0.30.0 aiohttp pandas matplotlib 2>&1 | tail -3
!python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
""")

code(r"""
# 5. Traces (public Mooncake/Kimi production traces) + sanity check that our
#    policies load through vLLM's official out-of-tree hook.
!bash scripts_get_traces.sh && ls -la traces
!python -c "from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory as F; [print(n, F.get_cache_policy_cls(n, 'kvpolicy.probe')) for n in ['LRUProbe','ARCProbe','DIPProbe','PGProbe']]"
""")

code(r"""
# 6. Simulator predictions for exactly these configs (CPU only, ~1 min).
#    Later we compare measured GPU hit rates against these.
import os
PRED = f'{RESULTS}/predictions.jsonl'
if not os.path.exists(PRED):
    for t, c in [('conversation', 500), ('toolagent', 500), ('conversation', 1000)]:
        sh('python', 'bench/predict.py', '--trace', f'traces/{t}_trace.jsonl', '--cpu-chunks', c, '--out', PRED)
print(open(PRED).read())
""")

code(r"""
# 7. Smoke test (~5 min): 80 requests, DIP policy. Must show policy_loaded=DIPProbe,
#    0 errors and non-zero hit rate before you spend compute on the full grid.
sh('python', 'bench/run.py', '--trace', 'traces/conversation_trace.jsonl', '--config', 'dip',
   '--cpu-chunks', 500, '--limit', 80, '--speedup', 20, '--run-name', 'smoke_dip', '--out-root', RESULTS)
import json
s = json.load(open(f'{RESULTS}/smoke_dip/summary.json'))
print({k: s[k] for k in ['policy_loaded', 'cpu_capacity_chunks_actual', 'n_ok', 'n_errors',
                         'hit_rate_gpu_tokens', 'hit_rate_cpu_tokens', 'ttft_ms_p50']})
assert s['n_errors'] == 0 and s['policy_loaded'] == 'DIPProbe', 'check server.log in the run folder'
""")

code(r"""
# 8. Calibrate: measure this GPU's prefill throughput and choose the replay speed per trace
#    (no-cache load = 80% of capacity). Same speed is used for every config => fair comparison.
sh('python', 'bench/calibrate.py', '--out', f'{RESULTS}/calibration.json',
   '--traces', 'traces/conversation_trace.jsonl', 'traces/toolagent_trace.jsonl')
CAL = json.load(open(f'{RESULTS}/calibration.json')); CAL
""")

code(r"""
# 9. The grid. Ordered by importance; re-run this cell after any disconnect (finished runs are skipped).
import subprocess, time, json, os
CORE = [(t, c, 500) for t in ['conversation', 'toolagent'] for c in ['gpu', 'lru', 'dip', 'adm2']] \
     + [(t, 'nocache', 500) for t in ['conversation', 'toolagent']]
EXTENDED = [('conversation', 'arc', 500), ('conversation', 'pg', 500)] \
         + [('conversation', c, 1000) for c in ['lru', 'dip', 'adm2']]
RUN_EXTENDED = False
grid = CORE + (EXTENDED if RUN_EXTENDED else [])
t_start = time.time()
for i, (t, cfg, chunks) in enumerate(grid):
    sp = CAL['speedup'][t]
    name = f'{t}_{cfg}_c{chunks}_x{sp:g}'
    done = os.path.exists(f'{RESULTS}/{name}/summary.json')
    print(f'[{i+1}/{len(grid)}] {name}' + ('  (done, skipping)' if done else ''), flush=True)
    if done:
        continue
    r = subprocess.run(['python', 'bench/run.py', '--trace', f'traces/{t}_trace.jsonl', '--config', cfg,
                        '--cpu-chunks', str(chunks), '--speedup', str(sp), '--run-name', name,
                        '--out-root', RESULTS])
    if r.returncode:
        print(f'   !! run failed, see {RESULTS}/{name}/server.log -- continuing')
    print(f'   elapsed {(time.time()-t_start)/60:.0f} min', flush=True)
""")

code(r"""
# 10. Results: table, measured vs simulated hit rate, TTFT plots (saved to Drive)
import glob, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
rows = []
for f in glob.glob(f'{RESULTS}/*/summary.json'):
    if '/smoke_' in f: continue
    s = json.load(open(f)); s['run'] = f.split('/')[-2]; rows.append(s)
df = pd.DataFrame(rows)
pred = [json.loads(l) for l in open(PRED)]
def predicted(r):
    for p in pred:
        if p['trace'].startswith(r['trace'].split('/')[-1].split('_')[0]) and p['cpu_chunks'] == r['cpu_chunks']:
            return p.get(r['config'])
df['hit_rate_sim'] = df.apply(predicted, axis=1)
order = {'nocache': 0, 'gpu': 1, 'lru': 2, 'arc': 3, 'adm2': 4, 'pg': 5, 'dip': 6}
df['trace_name'] = df['trace'].str.extract(r'(\w+)_trace')[0]
df = df.sort_values(['trace_name', 'cpu_chunks', 'config'], key=lambda c: c.map(order) if c.name == 'config' else c)
cols = ['trace_name', 'cpu_chunks', 'config', 'n_ok', 'n_errors', 'ttft_ms_p50', 'ttft_ms_p95', 'ttft_ms_p99',
        'hit_rate_gpu_tokens', 'hit_rate_cpu_tokens', 'hit_rate_total_tokens', 'hit_rate_sim']
table = df[cols].round(3); table.to_csv(f'{RESULTS}/results_table.csv', index=False); display(table)

# TTFT reduction vs the stock CPU-tier LRU and vs no-cache
for (t, c), g in df.groupby(['trace_name', 'cpu_chunks']):
    base = g[g.config == 'lru']
    if len(base):
        b = base.iloc[0]
        for _, r in g.iterrows():
            print(f"{t:12s} c{c} {r.config:8s} p50 {r.ttft_ms_p50:8.0f} ms ({100*(1-r.ttft_ms_p50/b.ttft_ms_p50):+5.1f}% vs LRU)  "
                  f"p95 {r.ttft_ms_p95:8.0f} ms ({100*(1-r.ttft_ms_p95/b.ttft_ms_p95):+5.1f}%)  hit {r.hit_rate_total_tokens:.3f}")

fig, axes = plt.subplots(1, df.trace_name.nunique(), figsize=(12, 4), squeeze=False)
for ax, (t, g) in zip(axes[0], df[df.cpu_chunks == 500].groupby('trace_name')):
    for _, r in g.iterrows():
        x = [json.loads(l)['ttft_s'] for l in open(f"{RESULTS}/{r.run}/requests.jsonl")]
        x = np.sort([v for v in x if v is not None]) * 1000
        ax.plot(x, np.linspace(0, 1, len(x)), label=r.config)
    ax.set_xscale('log'); ax.set_xlabel('TTFT (ms)'); ax.set_ylabel('CDF'); ax.set_title(t); ax.legend()
plt.tight_layout(); plt.savefig(f'{RESULTS}/ttft_cdf.png', dpi=150); plt.show()
""")

code(r"""
# 11. ARC check (extended grid): vLLM's ARC promotes every new chunk to T2 because the
#     connector re-touches keys after storing them. Look at T1 size inside the real engine.
import glob
for f in glob.glob(f'{RESULTS}/*_arc_*/policy.log'):
    recs = [json.loads(l) for l in open(f)]
    print(f.split('/')[-2], [(r.get('arc_t1'), r.get('arc_t2')) for r in recs if r['event'] == 'periodic'][-5:])
""")

md(r"""
## Reading the results
* **hit_rate_total_tokens** = (GPU + CPU-tier prefix hits) / prompt tokens, from vLLM's own Prometheus counters.
  `hit_rate_sim` is the offline simulator's prediction for the CPU tier; the two should be close
  (the GPU tier in front is small and inclusive).
* TTFT includes queueing, so a few points of hit rate can move p95 a lot under load.
* Setup caveats to state honestly: Qwen2.5-1.5B-Instruct on one A100; 1/4 session sample of each trace
  with CPU capacity scaled by 1/4; prompts truncated to 30k tokens; outputs capped at 64 tokens;
  token contents are synthetic (the traces publish only block hashes), sharing structure is real.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm",
                                                                   "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.join(ROOT, "colab"), exist_ok=True)
with open(os.path.join(ROOT, "colab", "kv_offload_benchmark.ipynb"), "w") as f:
    json.dump(nb, f, indent=1)
print("wrote colab/kv_offload_benchmark.ipynb with", len(cells), "cells")
