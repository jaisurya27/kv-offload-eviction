"""Tiny stand-in for `vllm serve` (OpenAI completions SSE + /metrics + /health) that
emulates prefix caching on 16-token blocks, so bench/run.py can be tested on CPU."""
import asyncio, json, sys, time
from aiohttp import web

BS = 16
cache = set()
M = {"q": 0, "h": 0}

async def health(_): return web.Response(text="ok")

async def metrics(_):
    t = (f'vllm:prefix_cache_queries_total{{engine="0"}} {M["q"]}\n'
         f'vllm:prefix_cache_hits_total{{engine="0"}} {M["h"]}\n'
         f'vllm:external_prefix_cache_queries_total{{engine="0"}} 0\n'
         f'vllm:external_prefix_cache_hits_total{{engine="0"}} 0\n')
    return web.Response(text=t)

async def comp(req):
    b = await req.json()
    ids = b["prompt"]; assert isinstance(ids, list) and all(isinstance(x, int) for x in ids)
    hit = 0
    keys = [tuple(ids[:i + BS]) for i in range(0, len(ids) - len(ids) % BS, BS)]
    for k in keys:
        if hash(k) in cache: hit += BS
        else: break
    for k in keys: cache.add(hash(k))
    M["q"] += len(ids); M["h"] += hit
    await asyncio.sleep(0.002 + (len(ids) - hit) * 2e-6)  # "prefill"
    r = web.StreamResponse(); r.content_type = "text/event-stream"; await r.prepare(req)
    for i in range(b["max_tokens"]):
        await r.write(f'data: {json.dumps({"choices":[{"text":"x","index":0}]})}\n\n'.encode())
    u = {"prompt_tokens": len(ids), "prompt_tokens_details": {"cached_tokens": hit}}
    await r.write(f'data: {json.dumps({"choices":[],"usage":u})}\n\n'.encode())
    await r.write(b"data: [DONE]\n\n")
    return r

app = web.Application(client_max_size=64 << 20)
app.add_routes([web.get("/health", health), web.get("/metrics", metrics), web.post("/v1/completions", comp)])
web.run_app(app, port=int(sys.argv[1]), print=None)
