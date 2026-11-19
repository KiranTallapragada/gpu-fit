"""Compare measured vLLM numbers (bench/results/*.tsv) with gpu-fit's predictions.

Usage: PYTHONPATH=src python3 bench/compare.py bench/results/<file>.tsv [--markdown]
"""

import csv
import sys

from gpu_fit.calc import estimate
from gpu_fit.model import load_model

# Folder names used on the benchmark host → Hugging Face ids.
MODEL_IDS = {
    "qwen3.8-27b-fp8": "Qwen/Qwen3.8-27B-FP8",
    "qwen3.6-35b-a3b-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    "dl-Qwen3-8B": "Qwen/Qwen3-8B",
    "dl-Qwen3-14B": "Qwen/Qwen3-14B",
    "dl-Qwen3-32B-FP8": "Qwen/Qwen3-32B-FP8",
}


def pct(pred: float, real: float) -> str:
    return f"{(pred - real) / real * 100:+.1f}%" if real else "n/a"


def main() -> None:
    path, markdown = sys.argv[1], "--markdown" in sys.argv
    rows = [r for r in csv.DictReader(open(path), delimiter="\t") if r["status"] == "ok"]
    specs = {}
    out = []
    for r in rows:
        mid = MODEL_IDS.get(r["model"], r["model"])
        spec = specs.setdefault(mid, load_model(mid))
        fit = estimate(spec, int(r["gpu_mem_mib"]) / 1024, int(r["context"]),
                       kv_dtype_bytes=1 if r["kv_dtype"] == "fp8" else 2)
        w, k, t = float(r["weights_gib"]), float(r["kv_pool_gib"]), int(r["kv_tokens"])
        out.append([mid, r["gpu"].replace("NVIDIA ", ""), r["kv_dtype"],
                    f"{w:.1f}", f"{fit.weights_gib:.1f}",
                    f"{k:.2f}", f"{fit.kv_pool_gib:.2f}",
                    f"{t:,}", f"{fit.kv_tokens:,}", pct(fit.kv_tokens, t),
                    f"{0.9 * int(r['gpu_mem_mib']) / 1024 - w - k:.2f}"])
    head = ["model", "gpu", "kv", "weights GiB (vLLM)", "weights GiB (gpu-fit)",
            "KV pool GiB (vLLM)", "KV pool GiB (gpu-fit)", "KV tokens (vLLM)",
            "KV tokens (gpu-fit)", "error", "implied overhead GiB"]
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for o in out:
            print("| " + " | ".join(o) + " |")
    else:
        for o in [head] + out:
            print("\t".join(o))


if __name__ == "__main__":
    main()
