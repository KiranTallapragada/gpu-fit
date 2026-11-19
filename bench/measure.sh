#!/usr/bin/env bash
# Measure what vLLM actually allocates, for comparison with gpu-fit.
#
# Usage: measure.sh <results.tsv> <model_dir_or_id> [<model> ...]
# Env:   VLLM (path to vllm binary), CONTEXT (default 32768), KV_DTYPES (default "auto fp8"),
#        MAX_NUM_SEQS (default 32)
#
# For each model and KV dtype it starts `vllm serve`, waits for the KV cache line,
# records the three numbers vLLM prints, and stops the server. Nothing is left running.
set -uo pipefail

out=$1; shift
VLLM=${VLLM:-vllm}
CONTEXT=${CONTEXT:-32768}
KV_DTYPES=${KV_DTYPES:-"auto fp8"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ver=$("$(dirname "$VLLM")/python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo unknown)

[ -s "$out" ] || printf 'model\tgpu\tgpu_mem_mib\tvllm\tcontext\tmax_num_seqs\tkv_dtype\tweights_gib\tkv_pool_gib\tkv_tokens\tstatus\n' > "$out"

for model in "$@"; do
  for kv in $KV_DTYPES; do
    log=$(mktemp)
    "$VLLM" serve "$model" --max-model-len "$CONTEXT" --kv-cache-dtype "$kv" \
      --max-num-seqs "$MAX_NUM_SEQS" --port 8123 >"$log" 2>&1 &
    pid=$!
    status=timeout
    for _ in $(seq 1 180); do  # up to 15 minutes
      if grep -q "GPU KV cache size" "$log"; then status=ok; break; fi
      if ! kill -0 $pid 2>/dev/null; then status=failed; break; fi
      if grep -qE "Engine core initialization failed|CUDA out of memory" "$log"; then status=failed; break; fi
      sleep 5
    done
    pkill -TERM -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 5
    pkill -KILL -f "vllm serve .* --port 8123" 2>/dev/null; wait $pid 2>/dev/null
    while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 2; done

    w=$(grep -oE "Model loading took [0-9.]+ GiB" "$log" | grep -oE "[0-9.]+" | head -1)
    k=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" "$log" | grep -oE "[0-9.]+" | head -1)
    t=$(grep -oE "GPU KV cache size: [0-9,]+" "$log" | grep -oE "[0-9,]+$" | tr -d , | head -1)
    name=$(basename "$model")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$gpu" "$mem" "$ver" "$CONTEXT" "$MAX_NUM_SEQS" "$kv" \
      "${w:-}" "${k:-}" "${t:-}" "$status" >> "$out"
    echo "$(date +%T) $name kv=$kv -> $status weights=${w:-?} kv_pool=${k:-?} tokens=${t:-?}"
    [ "$status" = ok ] || grep -E "Error|error:" "$log" | grep -v Traceback | tail -2 | cut -c1-300
    rm -f "$log"
  done
done
