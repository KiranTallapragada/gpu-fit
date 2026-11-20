#!/bin/sh
# Concurrency sweep with vLLM's own `vllm bench sweep serve` (vLLM >= 0.34).
# Starts the server once per setting, runs the benchmark at concurrency 1..32,
# and writes one JSON result per step. Then: python3 summarise.py <output-dir>
# Status: flags checked against vLLM 0.34 source; not yet run on a GPU.
#
# Usage: ./run.sh MODEL [INPUT_TOKENS] [OUTPUT_TOKENS] [extra vllm serve flags...]
#   ./run.sh Qwen/Qwen3-8B 2000 400 --max-model-len 8192
set -eu
MODEL=$1; IN=${2:-2000}; OUT=${3:-400}; shift; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
DIR=results-$(echo "$MODEL" | tr '/' '_')-in$IN-out$OUT-$(date +%Y%m%d-%H%M)
HERE=$(cd "$(dirname "$0")" && pwd)
vllm bench sweep serve \
  --serve-cmd "vllm serve $MODEL $* --max-num-seqs 32" \
  --bench-cmd "vllm bench serve --model $MODEL --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99" \
  --bench-params "$HERE/concurrency.json" \
  --output-dir "$HERE/$DIR" --continue-on-error
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$HERE/$DIR/gpu.txt"
python3 -c "import vllm; print(vllm.__version__)" > "$HERE/$DIR/vllm-version.txt"
echo "done: $HERE/$DIR  ->  python3 $HERE/summarise.py $HERE/$DIR"
