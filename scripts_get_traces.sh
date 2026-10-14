#!/usr/bin/env bash
# Public Mooncake/Kimi production traces (FAST'25), 512-token blocks, prefix-chained hashes
mkdir -p traces && cd traces
for f in conversation_trace toolagent_trace synthetic_trace; do
  curl -sSLO https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/$f.jsonl
done
