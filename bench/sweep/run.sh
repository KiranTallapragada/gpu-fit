#!/bin/sh
# Concurrency sweep with vLLM's own `vllm bench sweep serve` (vLLM >= 0.34).
# Starts the server, runs the benchmark at concurrency 1..32 and writes one JSON
# result per step. Then: python3 summarise.py <output-dir>
# Status: flags checked against vLLM 0.34 source; not yet run on a GPU.
#
# Usage: ./run.sh MODEL [INPUT_TOKENS] [OUTPUT_TOKENS] [extra vllm serve flags...]
#   ./run.sh Qwen/Qwen3-8B 2000 400 --max-model-len 8192
# Extra flags keep their quoting. --port N (or --port=N, -p N) is passed to the benchmark too.
set -eu
MODEL=$1; IN=${2:-2000}; OUT=${3:-400}
shift; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=results-$(echo "$MODEL" | tr '/' '_')-in$IN-out$OUT-$(date +%Y%m%d-%H%M)
# Quote the extra flags safely and normalise --port=N / -p N to "--port N": vLLM's sweep
# readiness check only reads the two-word form, so server, check and benchmark agree.
EXTRA=$(python3 -c '
import shlex, sys
a, out = sys.argv[1:], []
i = 0
while i < len(a):
    x = a[i]
    if x.startswith("--port="):
        out += ["--port", x.split("=", 1)[1]]
    elif x in ("-p", "--port") and i + 1 < len(a):
        out += ["--port", a[i + 1]]; i += 1
    else:
        out.append(x)
    i += 1
print(shlex.join(out))' "$@")
PORT=$(python3 -c '
import shlex, sys
a = shlex.split(sys.argv[1])
print(a[a.index("--port") + 1] if "--port" in a else 8000)' "$EXTRA")
QMODEL=$(python3 -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "$MODEL")
vllm bench sweep serve \
  --serve-cmd "vllm serve $QMODEL $EXTRA --max-num-seqs 32" \
  --bench-cmd "vllm bench serve --model $QMODEL --port $PORT --dataset-name random \
    --random-input-len $IN --random-output-len $OUT --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99" \
  --bench-params "$HERE/concurrency.json" \
  --output-dir "$HERE/$DIR" --continue-on-error
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$HERE/$DIR/gpu.txt"
python3 -c "import vllm; print(vllm.__version__)" > "$HERE/$DIR/vllm-version.txt"
echo "done: $HERE/$DIR  ->  python3 $HERE/summarise.py $HERE/$DIR"
