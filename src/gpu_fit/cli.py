"""gpu-fit: will this model fit on this GPU, at this context, for this many users (vLLM)?"""

import argparse
import json
import sys
from dataclasses import asdict

import math

from . import __version__
from .calc import VLLM_DEFAULT_MAX_NUM_SEQS, VLLM_DEFAULT_UTIL, estimate, estimate_speed
from .gpus import GPUS, get_gpu
from .model import load_model

TP_SIZES = (1, 2, 4, 8)
# vLLM's own CUDA context is already allocated when it checks free memory at startup.
STARTUP_MARGIN_GIB = 0.5


def resolve_util(gpu_mem: float, args) -> float:
    """gpu_memory_utilization to use. With --reserve-gib, an estimate of the highest value that
    leaves room for other processes holding that much memory on each GPU. (With estimate-based
    sizing, vLLM <= 0.33, vLLM refuses to start if free memory < total x utilization; the 0.5 GiB
    startup margin is an estimate.)"""
    util = args.util if args.util is not None else VLLM_DEFAULT_UTIL
    if args.reserve_gib:
        room = gpu_mem - args.reserve_gib - STARTUP_MARGIN_GIB
        cap = math.floor(room / gpu_mem * 100) / 100 if room > 0 else 0.0
        util = min(util, cap) if args.util is not None else cap
    return util


def parse_tokens(text: str) -> int:
    t = text.strip().lower()
    mult = 1
    if t.endswith("k"):
        mult, t = 1024, t[:-1]
    elif t.endswith("m"):
        mult, t = 1024 * 1024, t[:-1]
    return int(float(t) * mult)


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _command(model: str, fit, args, util: float) -> str:
    parts = [f"vllm serve {model}", f"--max-model-len {fit.context}"]
    if fit.tp > 1:
        parts.append(f"--tensor-parallel-size {fit.tp}")
    if args.kv_cache_dtype == "fp8":
        parts.append("--kv-cache-dtype fp8")
    parts.append(f"--max-num-seqs {fit.max_num_seqs}")
    # Always explicit: vLLM 0.34+ uses 1.0 when the flag is unset (extensible KV cache).
    parts.append(f"--gpu-memory-utilization {util}")
    return " \\\n  ".join(parts)


def _seqs(args) -> int:
    if args.max_num_seqs:
        return args.max_num_seqs
    return args.concurrency or VLLM_DEFAULT_MAX_NUM_SEQS


def report_one(spec, gpu_name, gpu_mem, args, bandwidth_gbs=0) -> int:
    kv_bytes = 1 if args.kv_cache_dtype == "fp8" else 2
    util = resolve_util(gpu_mem, args)
    if util <= 0:
        msg = (f"--reserve-gib {args.reserve_gib} leaves no room for vLLM on a {gpu_mem:.1f} GiB GPU")
        print(json.dumps({"model": spec.name, "gpu": gpu_name, "fits": False, "error": msg})
              if args.json else f"\n✗ {msg}.\n")
        return 1
    fit = estimate(spec, gpu_mem, args.context, args.tp, kv_bytes, util, args.overhead_gib,
                   _seqs(args))
    need = args.concurrency or 1
    ok = fit.kv_pool_gib > 0 and fit.concurrent >= need and fit.slots_ok

    if args.json:
        sp = estimate_speed(spec, bandwidth_gbs, args.context, args.tp, kv_bytes,
                            max(1, min(need, int(fit.concurrent))) if fit.concurrent >= 1 else 1)
        print(json.dumps({"model": spec.name, "gpu": gpu_name, "fits": ok,
                          "gpu_memory_utilization": util, "reserve_gib": args.reserve_gib or 0,
                          **asdict(fit),
                          "speed": asdict(sp) if sp else None}, indent=2))
        return 0 if ok else 1

    print(f"\n{spec.name}  on  {args.tp}× {gpu_name} ({gpu_mem:.1f} GiB each)")
    print(f"context {_fmt_tokens(args.context)} · KV cache {args.kv_cache_dtype}"
          f" · gpu_memory_utilization {util}"
          + (f" (other processes hold {args.reserve_gib:g} GiB per GPU)" if args.reserve_gib else "")
          + "\n")
    rows = [
        ("vLLM budget per GPU", f"{fit.budget_gib:6.1f} GiB"),
        ("  weights", f"{fit.weights_gib:6.1f} GiB"),
        ("  runtime overhead", f"{fit.overhead_gib:6.1f} GiB"),
        ("  left for KV cache", f"{fit.kv_pool_gib:6.1f} GiB"),
        ("KV cache capacity", f"{_fmt_tokens(fit.kv_tokens)} tokens"),
        ("Requests at full context", f"{fit.concurrent:.1f}"),
        ("Longest single request", f"{_fmt_tokens(fit.max_context_one_request)} tokens"),
        ("--max-num-seqs", f"{fit.max_num_seqs}"),
    ]
    if fit.hybrid_slots:
        rows.append(("Hybrid state slots (max seqs)", f"{fit.hybrid_slots}"))
    users = max(1, min(need, int(fit.concurrent))) if fit.concurrent >= 1 else 1
    speed = estimate_speed(spec, bandwidth_gbs, args.context, args.tp, kv_bytes, users)
    if speed and fit.kv_pool_gib > 0:
        rows.append(("Speed, one user (est.)", f"~{speed.single_tok_s:,.0f} tokens/s"))
        if speed.batch > 1:
            rows.append((f"Speed, {speed.batch} users (est.)",
                         f"~{speed.total_tok_s:,.0f} tokens/s total, ~{speed.per_user_tok_s:,.0f} each"))
    for label, value in rows:
        print(f"  {label:<30} {value}")
    for note in spec.notes:
        print(f"  note: {note}")
    if spec.max_context and args.context > spec.max_context:
        print(f"  note: {spec.name} natively supports {spec.max_context:,} tokens; longer contexts"
              " need RoPE scaling (e.g. YaRN) set in vLLM")
    if speed and fit.kv_pool_gib > 0:
        print("  note: speed is a decode estimate from memory bandwidth; prompt processing not included")

    if fit.kv_pool_gib > 0 and not fit.slots_ok:
        print(f"\n✗ vLLM will refuse to start: --max-num-seqs {fit.max_num_seqs} is more than the"
              f" {fit.hybrid_slots} recurrent-state slots this hybrid model gets."
              f" Use --max-num-seqs {fit.hybrid_slots} or lower.")
    elif fit.kv_pool_gib <= 0:
        print("\n✗ Does not fit: the weights alone exceed the budget. Try --tp, a bigger GPU or a quantised model.")
    elif not ok:
        tips = ("a shorter context or more GPUs" if args.kv_cache_dtype == "fp8"
                else "--kv-cache-dtype fp8, a shorter context or more GPUs")
        print(f"\n✗ Fits, but only {fit.concurrent:.1f} requests at {_fmt_tokens(args.context)} tokens"
              f" (you asked for {need}). Try {tips}.")
    else:
        if args.reserve_gib:
            print(f"\n  note: start the other processes first; vLLM requests {util * gpu_mem:.1f} GiB"
                  " per GPU (an estimate that keeps a 0.5 GiB margin). If they grow after vLLM"
                  " starts, raise --reserve-gib.")
        print(f"\n✓ Fits. Suggested command:\n\n  {_command(spec.name, fit, args, util)}")
    print()
    return 0 if ok else 1


def report_scan(spec, args) -> int:
    kv_bytes = 1 if args.kv_cache_dtype == "fp8" else 2
    need = args.concurrency or 1
    results = []
    utils = {}
    for gpu in GPUS.values():
        util = resolve_util(gpu.memory_gib, args)
        utils[gpu.name] = util
        for tp in TP_SIZES:
            if util <= 0:
                continue
            fit = estimate(spec, gpu.memory_gib, args.context, tp, kv_bytes, util,
                           args.overhead_gib, _seqs(args))
            if fit.kv_pool_gib > 0 and fit.concurrent >= need and fit.slots_ok:
                results.append((gpu, fit))
                break
        else:
            results.append((gpu, None))

    if args.json:
        print(json.dumps([{"gpu": g.name, "fits": f is not None,
                           "gpu_memory_utilization": utils[g.name],
                           "reserve_gib": args.reserve_gib or 0, **(asdict(f) if f else {})}
                          for g, f in results], indent=2))
        return 0

    print(f"\n{spec.name} · context {_fmt_tokens(args.context)} · {need} request(s) at once"
          f" · KV {args.kv_cache_dtype}"
          + (f" · {args.reserve_gib:g} GiB per GPU held by other processes" if args.reserve_gib else "")
          + "\n")
    print(f"  {'GPU':<13}{'smallest setup':<16}{'requests at full ctx':>22}{'KV tokens':>14}")
    for gpu, fit in results:
        if fit is None:
            print(f"  {gpu.name:<13}" + ("no room after the reserve" if utils[gpu.name] <= 0
                                            else "does not fit on 8 or fewer"))
        else:
            print(f"  {gpu.name:<13}{f'{fit.tp}× {gpu.name}':<16}{fit.concurrent:>22.1f}{_fmt_tokens(fit.kv_tokens):>14}")
    print("\nDetail for one option: gpu-fit MODEL --gpu <name> --tp <n> ...\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gpu-fit", description=__doc__)
    p.add_argument("model", help="Hugging Face id (org/name) or a local model folder")
    p.add_argument("--gpu", help="GPU name, e.g. L40S, H100, A100-80. Omit to compare all.")
    p.add_argument("--gpu-mem-gib", type=float, help="Custom GPU memory instead of --gpu")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (default 1)")
    p.add_argument("--context", type=parse_tokens, default=parse_tokens("32k"),
                   help="tokens per request, e.g. 32k, 128k (default 32k)")
    p.add_argument("--concurrency", type=int, help="requests at full context you need at once")
    p.add_argument("--max-num-seqs", type=int,
                   help="vLLM request slots (default: --concurrency, else vLLM's 256)")
    p.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    p.add_argument("--util", type=float,
                   help=f"gpu_memory_utilization (default: vLLM's {VLLM_DEFAULT_UTIL})")
    p.add_argument("--reserve-gib", type=float, default=0,
                   help="GiB per GPU already used by other processes (OCR, embeddings, a second"
                        " model); estimates the highest --gpu-memory-utilization that leaves room")
    p.add_argument("--overhead-gib", type=float, help="runtime overhead per GPU (default: measured, by model type)")
    p.add_argument("--bandwidth-gbs", type=float, help="GPU memory bandwidth in GB/s (default: from catalog)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--version", action="version", version=f"gpu-fit {__version__}")
    args = p.parse_args(argv)
    if args.reserve_gib is None or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        p.error("--reserve-gib must be a number >= 0")
    if args.util is not None and (not math.isfinite(args.util) or not 0 < args.util <= 1):
        p.error("--util must be greater than 0 and at most 1")

    spec = load_model(args.model)
    if args.gpu_mem_gib:
        return report_one(spec, "custom", args.gpu_mem_gib, args, args.bandwidth_gbs or 0)
    if args.gpu:
        gpu = get_gpu(args.gpu)
        return report_one(spec, gpu.name, gpu.memory_gib, args, args.bandwidth_gbs or gpu.bandwidth_gbs)
    return report_scan(spec, args)


if __name__ == "__main__":
    sys.exit(main())
