#!/usr/bin/env bash
# Abliterated DeepSeek-V4-Flash-0731 (MXFP4) + DSpark.
# usage: llamacpp-abl.sh [NCMOE] [THREADS] [CTX] [NMAX] [HOST] [PORT] [DRAFT]
#   DRAFT: "abl" = abliterated Q8_0 drafter (10.9 GB)
#          "old" = original Q2_K drafter (7.0 GB, vocab-identical)
#          "none"
set -u
NCMOE=${1:-43}; THREADS=${2:-32}; CTX=${3:-16384}; NMAX=${4:-1}
HOST=${5:-172.18.0.1}; PORT=${6:-30001}; DRAFTSEL=${7:-abl}
# 8th arg: --cache-ram MiB. llama.cpp keeps completed prompt caches in host
# RAM and swaps them back into the slot, so switching between conversations
# does not force a re-prefill. Default is 8192 MiB (~6 full 16K conversations
# at this model's 86 KB/token). --cache-reuse is deliberately NOT used: this
# context cannot KV-shift (sliding-window attention), and llama.cpp disables
# the flag with "cache_reuse is not supported by this context".
CACHERAM=${8:-24576}

BIN=/home/john/llama.cpp/build/bin/llama-server
MODEL=/srv/llm/models/gguf/abliterated/DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf
DRAFT_ABL=/srv/llm/models/gguf/abliterated/dspark-abliterated/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
DRAFT_OLD=/srv/llm/models/gguf/dspark-drafter/DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf
LOGDIR=/srv/llm/logs
ts=$(date +%Y%m%d-%H%M%S)
LOG="$LOGDIR/llamacpp-abl-ncmoe${NCMOE}-t${THREADS}-nmax${NMAX}-draft${DRAFTSEL}-$ts.log"

[ -f "$MODEL" ] || { echo "FATAL: abliterated model missing: $MODEL" >&2; exit 1; }

args=(
  --model "$MODEL"
  --alias deepseek-v4-flash
  --host "$HOST" --port "$PORT"
  --ctx-size "$CTX"
  --n-gpu-layers 99
  --n-cpu-moe "$NCMOE"
  --threads "$THREADS" --threads-batch "$THREADS"
  --flash-attn on
  --cache-type-k f16 --cache-type-v f16   # K MUST stay f16 (llama.cpp #25382)
  --parallel 1
  --jinja
  --reasoning-format deepseek
  --chat-template-kwargs '{"enable_thinking":false}'
  --load-mode mmap
  --no-warmup
)
args+=( --cache-ram "$CACHERAM" )
case "$DRAFTSEL" in
  abl)  [ -f "$DRAFT_ABL" ] || { echo "FATAL: abl drafter missing" >&2; exit 1; }
        args+=( --spec-draft-model "$DRAFT_ABL" --spec-draft-ngl 99 --spec-draft-n-max "$NMAX" ) ;;
  old)  [ -f "$DRAFT_OLD" ] || { echo "FATAL: old drafter missing" >&2; exit 1; }
        args+=( --spec-draft-model "$DRAFT_OLD" --spec-draft-ngl 99 --spec-draft-n-max "$NMAX" ) ;;
  none) : ;;
  *) echo "FATAL: DRAFT must be abl|old|none" >&2; exit 1 ;;
esac

mkdir -p "$LOGDIR"
echo "=== abliterated start $(date -Is) ===" | tee "$LOG"
echo "ncmoe=$NCMOE threads=$THREADS ctx=$CTX nmax=$NMAX host=$HOST port=$PORT draft=$DRAFTSEL" | tee -a "$LOG"
exec taskset -c 0-63 "$BIN" "${args[@]}" >> "$LOG" 2>&1
