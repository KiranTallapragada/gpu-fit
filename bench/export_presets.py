"""Export model specs for the browser calculator (one JSON file, no network at run time).

Usage: PYTHONPATH=src python3 bench/export_presets.py out.json
"""

import json
import sys
from dataclasses import asdict

from gpu_fit.gpus import GPUS
from gpu_fit.model import load_model

PRESETS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3.8-27B-FP8",
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "microsoft/phi-4",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def main() -> None:
    models = []
    for mid in PRESETS:
        try:
            spec = load_model(mid)
        except SystemExit as e:
            print(f"skip {mid}: {e}", file=sys.stderr)
            continue
        d = asdict(spec)
        d.pop("notes", None)
        models.append(d)
        print(f"ok   {mid}: {spec.weight_bytes / 1024**3:.1f} GiB", file=sys.stderr)
    gpus = [{"name": g.name, "memory_mib": g.memory_mib} for g in GPUS.values()]
    json.dump({"models": models, "gpus": gpus}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
