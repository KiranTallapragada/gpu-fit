# gpu-fit

[![PyPI](https://img.shields.io/pypi/v/gpu-fit)](https://pypi.org/project/gpu-fit/) [![Tests](https://github.com/KiranTallapragada/gpu-fit/actions/workflows/test.yml/badge.svg)](https://github.com/KiranTallapragada/gpu-fit/actions/workflows/test.yml) [![Python](https://img.shields.io/pypi/pyversions/gpu-fit)](https://pypi.org/project/gpu-fit/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**[Try it in your browser →](https://kiran-tallapragada.pages.dev/gpu-fit/)** · no install, the calculation runs in your browser

**Will this LLM fit on this GPU, at this context length, for this many users, and how fast?**

`gpu-fit` answers that before you rent the GPU. It reads the model's config and weight sizes
from Hugging Face and works out vLLM's memory budget: weights, runtime overhead, KV cache, how
many requests fit at full context, and the longest single request. Then it prints the
`vllm serve` command.

**Sharing the GPU with something else?** `--reserve-gib` estimates the `--gpu-memory-utilization`
that leaves room for the other processes ([details](#sharing-the-gpu-with-something-else)).

```
$ gpu-fit Qwen/Qwen3-32B --context 32k --concurrency 4

  GPU          smallest setup    requests at full ctx     KV tokens
  L4           8× L4                             11.0       361,200
  L40S         4× L40S                           12.0       394,683
  A100-80      2× A100-80                        10.3       336,234
  H100         2× H100                           10.2       333,577
  H200         1× H200                            8.3       270,751
  ...
```

```
$ gpu-fit deepseek-ai/DeepSeek-V3 --gpu H200 --tp 8 --context 128k

  vLLM budget per GPU             129.2 GiB
    weights                        80.2 GiB
    runtime overhead                3.0 GiB
    left for KV cache              46.0 GiB
  KV cache capacity              702,962 tokens
  Requests at full context       5.4
  Longest single request         163,840 tokens

✓ Fits. Suggested command:

  vllm serve deepseek-ai/DeepSeek-V3 \
  --max-model-len 131072 \
  --tensor-parallel-size 8 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.92
```

## Install

```
pip install gpu-fit        # no dependencies, Python 3.10+
```

## What gpu-fit models

A parameters-times-bytes estimate is a good start. Newer architectures change the answer, so gpu-fit reads them:

| Architecture | What changes | gpu-fit |
|---|---|---|
| Grouped-query attention | KV per token depends on KV heads, not attention heads | ✓ |
| Hybrid (linear attention + full attention, e.g. Qwen3-Next family) | Only full-attention layers grow with context; the rest hold fixed state per request | ✓ |
| MLA (DeepSeek) | Compact latent KV, but **copied to every** tensor-parallel GPU | ✓ |
| Sliding window (Gemma, Mistral) | Window layers stop growing past the window | ✓ |
| FP8 / AWQ / GPTQ checkpoints | Exact weight size from the safetensors metadata | ✓ |
| Tensor parallel beyond KV heads | KV heads get replicated, not split | ✓ |
| Hybrid `--max-num-seqs` limit | vLLM refuses to start above the state-slot count | ✓ warns first |
| FP8 KV cache | About twice the KV-token capacity: `--kv-cache-dtype fp8` | ✓ |

## Options

```
gpu-fit MODEL [--gpu L40S | --gpu-mem-gib 48] [--tp 2] [--context 128k]
              [--concurrency 8] [--max-num-seqs 32] [--kv-cache-dtype fp8] [--util 0.92]
              [--reserve-gib 12] [--overhead-gib 2.0] [--bandwidth-gbs 864] [--json]
```

- `MODEL`: a Hugging Face id, or a local folder with `config.json` and `*.safetensors`.
  Gated models need `HF_TOKEN` in the environment.
- Leave out `--gpu` to compare every GPU in the catalog with the smallest tensor-parallel size that works.
- `--json` for scripts and CI.

## Sharing the GPU with something else

vLLM grabs a fixed share of the GPU at startup and doesn't coordinate with anything else on
the card: an OCR model, an embedding server, a second vLLM, or the old pod during a rolling
deploy. With estimate-based sizing (vLLM 0.33 and earlier) it refuses to start if free memory
is below `total × gpu_memory_utilization`. vLLM 0.34's extensible KV cache skips that startup
check, but the budget still caps vLLM's memory. Either way, if the other process grows later,
an out-of-memory error can follow.

Tell gpu-fit how much the other processes need per GPU and it estimates the highest
`--gpu-memory-utilization` that leaves room for them, and what still fits in what's left:

```
gpu-fit Qwen/Qwen3-8B --gpu L40S --reserve-gib 12 --context 16k --concurrency 8 --kv-cache-dtype fp8
...
✓ Fits. Suggested command:

  vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.72
```

Start the other processes first, and size `--reserve-gib` for their peak, not their idle use.
gpu-fit keeps an estimated 0.5 GiB back for vLLM's own CUDA context. Suggested commands always
state `--gpu-memory-utilization`, because vLLM 0.34 uses the whole GPU when it's left out.

## How the maths works

vLLM takes `gpu_memory_utilization` × GPU memory, loads the weights, profiles a forward pass
(activations, CUDA graphs), and gives the rest to the KV cache. `gpu-fit` mirrors that:

```
KV pool   = memory × utilization − weights / tp − overhead
KV/token  = 2 × full-attention layers × KV heads per GPU × head dim × dtype bytes
requests  = KV pool / (context × KV/token + sliding-window KV + recurrent state)
```

## Speed estimates

gpu-fit also estimates decode speed (tokens per second): for one user, and per user at your
concurrency. Decoding is usually limited by memory bandwidth, so the estimate divides the GPU's memory
bandwidth by what each step reads: the weights (for mixture-of-experts models, only the experts
in use) plus every active request's KV cache, with a fixed per-layer cost that matters for small
MoE models. Prompt processing is not included. It's a planning estimate: real numbers from your
hardware are welcome (see CONTRIBUTING.md).

## Accuracy, measured

Predicted vs what vLLM 0.31 actually allocated on three AWS GPUs across two generations: **L40S 48 GB**
(g6e, Ada), **A10G 24 GB** (g5, Ampere) and **L4 24 GB** (g6, Ada). 32K context unless noted,
`--max-num-seqs 32`. Raw data and method: [`bench/`](bench/).

| Model | Type | GPU | KV cache | KV tokens (vLLM) | KV tokens (gpu-fit) | Error |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | dense | L40S (Ada) | default | 175,696 | 175,874 | +0.1% |
| Qwen/Qwen3-14B | dense | L40S (Ada) | default | 75,888 | 75,569 | −0.4% |
| Qwen/Qwen3.8-27B-FP8 | hybrid + vision | L40S (Ada) | default | 137,216 | 134,077 | −2.3% |
| Qwen/Qwen3.6-35B-A3B-FP8 | hybrid MoE + vision | L40S (Ada) | default | 179,361 | 158,477 | −11.6% (cautious) |
| Qwen/Qwen3-4B | dense | A10G (Ampere) | default | 85,712 | 85,719 | +0.0% |
| Qwen/Qwen3-8B-FP8 | dense, FP8 weights | A10G (Ampere) | default | 72,496 | 72,240 | −0.4% |
| Qwen/Qwen3-4B | dense | A10G (Ampere) | FP8 | 178,736 | 171,438 | −4.1% |
| Qwen/Qwen3-8B-FP8 | dense, FP8 weights | A10G (Ampere) | FP8 | 158,240 | 144,480 | −8.7% (cautious) |
| Qwen/Qwen3-8B | dense | A10G (Ampere), 16K | default | 26,256 | 25,142 | −4.2% |
| Qwen/Qwen3-8B | dense | A10G (Ampere), 16K | FP8 | 51,200 | 50,284 | −1.8% |
| Qwen/Qwen3-4B | dense | L4 (Ada) | default | 85,632 | 85,758 | +0.1% |
| Qwen/Qwen3-8B-FP8 | dense, FP8 weights | L4 (Ada) | default | 72,432 | 72,279 | −0.2% |
| Qwen/Qwen3-4B | dense | L4 (Ada) | FP8 | 178,448 | 171,516 | −3.9% |
| Qwen/Qwen3-8B-FP8 | dense, FP8 weights | L4 (Ada) | FP8 | 158,080 | 144,558 | −8.6% (cautious) |
| Qwen/Qwen3-8B | dense | L4 (Ada), 16K | default | 26,192 | 25,181 | −3.9% |
| Qwen/Qwen3-8B | dense | L4 (Ada), 16K | FP8 | 51,072 | 50,362 | −1.4% |

- **Dense models, default KV cache, 32K:** within 0.5% of what vLLM reported on all three GPUs
  (at most 0.15% above). At 16K and with an FP8 KV cache the estimates are more cautious (up to 9% below).
- **Same behaviour on Ampere and Ada:** A10G and L4 results agree within 0.4%, and vLLM's runtime
  overhead is the same on both. FP8 weights and the FP8 KV cache both work on the Ampere A10G.
- Qwen3-8B (16-bit) at 32K context does not start on a 24 GB GPU: gpu-fit predicts this and says so.

- Weights come from the checkpoint's file sizes and KV per token from the model config; vLLM's
  runtime allocation can differ slightly from the file sizes.
- Runtime overhead (activations, CUDA graphs, non-torch memory) is calibrated from these runs at
  vLLM's default `gpu_memory_utilization` of 0.92. For dense text models it grows with the model's
  width: about 0.5 + 0.37 GiB per 1,024 of hidden size. That was measured on Qwen3 models with hidden
  sizes from 2,560 to 5,120; outside that range, and for sliding-window models, it's an extrapolation.
  Mixture-of-experts, hybrid and vision models use fixed values measured on the L40S. A bit more is
  added per `--max-num-seqs` slot. Override it with `--overhead-gib`. With an FP8 KV cache, vLLM's
  overhead is sometimes lower, so those estimates are cautious.
- **vLLM 0.34 changed this.** On CUDA it now sizes the KV cache from memory actually free
  after warm-up ("extensible KV cache") and uses the whole GPU unless you set
  `--gpu-memory-utilization` (gpu-fit's suggested commands always set it). gpu-fit hasn't been
  measured on 0.34 yet, so check the numbers there; measurements on 0.34 are next.
- **Hybrid models have a hard limit on `--max-num-seqs`.** Each request needs one
  recurrent-state slot. Ask for more slots than fit and vLLM refuses to start; gpu-fit
  tells you the limit first (it predicted 156 where vLLM reported 164).
- Treat the output as a planning estimate and confirm on the real GPU before you commit money.

**Help improve the calibration:** share what vLLM printed on your hardware (see [CONTRIBUTING.md](CONTRIBUTING.md)).
Measurements from more GPUs and models help refine the defaults for everyone.

## Compatibility

From 1.0, the command-line flags and the fields in `--json` output stay compatible across 1.x
releases; new options and fields may be added. `--json` prints one object for a single GPU
(`--gpu` or `--gpu-mem-gib`) and a list of objects, one per GPU, when comparing all GPUs; `speed`
is `null` when no estimate is available. Memory values are GiB, contexts are tokens. Estimates can change between releases as new
measurements improve the calibration; the accuracy section above always says what has been measured.

## Roadmap

- [x] Memory model calibrated on real hardware (L40S)
- [x] Hybrid `--max-num-seqs` limit warning
- [x] Browser calculator
- [x] Decode speed estimate, MoE-aware
- [x] Shared-GPU budgeting (`--reserve-gib`)
- [ ] Re-measure on vLLM 0.34 (extensible KV cache)
- [ ] Workload mode: safe concurrency, throughput and cost per 1,000 requests
- [ ] Speed calibrated against measured vLLM throughput
- [x] Measured on three GPUs across Ampere and Ada (A10G, L4, L40S)
- [ ] More GPUs measured (A100, H100, B200) and models (Llama, Gemma, Mistral)
- [ ] Prompt-processing (prefill) time and time-to-first-token
- [ ] Cost per million tokens for cloud GPUs
- [ ] SGLang and TensorRT-LLM memory models
- [ ] LoRA adapters and speculative decoding

## License

MIT
