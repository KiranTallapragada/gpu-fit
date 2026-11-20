# gpu-fit

**[Try it in your browser →](https://kiran-tallapragada.pages.dev/gpu-fit/)** · no install, nothing leaves your browser

**Will this LLM fit on this GPU, at this context length, for this many users, and how fast?**

`gpu-fit` answers that before you rent the GPU. It reads the model's config and weight sizes
from Hugging Face and works out vLLM's memory budget: weights, runtime overhead, KV cache, how
many requests fit at full context, and the longest single request. Then it prints the
`vllm serve` command.

```
$ gpu-fit Qwen/Qwen3-32B --context 32k --concurrency 4

  GPU          smallest setup    requests at full ctx     KV tokens
  L4           8× L4                             10.6       347,888
  L40S         4× L40S                           11.6       380,656
  A100-80      2× A100-80                         9.9       323,484
  H100         2× H100                            9.8       320,885
  H200         1× H200                            7.9       259,428
  ...
```

```
$ gpu-fit deepseek-ai/DeepSeek-V3 --gpu H200 --tp 8 --context 128k

  vLLM budget per GPU             126.4 GiB
    weights                        80.1 GiB
    runtime overhead                2.0 GiB
    left for KV cache              44.2 GiB
  KV cache capacity              675,671 tokens
  Requests at full context       5.2
  Longest single request         163,840 tokens

✓ Fits. Suggested command:
  vllm serve deepseek-ai/DeepSeek-V3 --max-model-len 131072 --tensor-parallel-size 8 --max-num-seqs 5
```

## Install

```
pip install gpu-fit        # no dependencies, Python 3.10+
```

## Why another VRAM calculator?

Most calculators multiply parameters by bytes and add a guess. Modern models break that:

| Architecture | What changes | gpu-fit |
|---|---|---|
| Grouped-query attention | KV per token depends on KV heads, not attention heads | ✓ |
| Hybrid (linear attention + full attention, e.g. Qwen3-Next family) | Only full-attention layers grow with context; the rest hold fixed state per request | ✓ |
| MLA (DeepSeek) | Compact latent KV, but **copied to every** tensor-parallel GPU | ✓ |
| Sliding window (Gemma, Mistral) | Window layers stop growing past the window | ✓ |
| FP8 / AWQ / GPTQ checkpoints | Exact weight size from the safetensors metadata | ✓ |
| Tensor parallel beyond KV heads | KV heads get replicated, not split | ✓ |
| Hybrid `--max-num-seqs` limit | vLLM refuses to start above the state-slot count | ✓ warns first |
| FP8 KV cache | Doubles capacity: `--kv-cache-dtype fp8` | ✓ |

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
deploy. It refuses to start if free memory is below `total × gpu_memory_utilization`, and if
the other process grows later, someone runs out of memory.

Tell gpu-fit how much the other processes need per GPU and it works out the highest
`--gpu-memory-utilization` vLLM will accept, and what still fits in what's left:

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
gpu-fit keeps 0.5 GiB back for vLLM's own CUDA context at startup.

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
concurrency. Generating a token is memory-bound, so the estimate divides the GPU's memory
bandwidth by what each step reads: the weights (for mixture-of-experts models, only the experts
in use) plus every active request's KV cache, with a fixed per-layer cost that matters for small
MoE models. Prompt processing is not included. It's a planning estimate: real numbers from your
hardware are welcome (see CONTRIBUTING.md).

## Accuracy, measured

Predicted vs what vLLM 0.31 actually allocated on an AWS g6e.xlarge (1× L40S 48 GB),
32K context, `--max-num-seqs 32`. Raw data and method: [`bench/`](bench/).

| Model | Type | KV tokens (vLLM) | KV tokens (gpu-fit) | Error |
|---|---|---|---|---|
| Qwen/Qwen3-8B | dense | 175,696 | 174,639 | −0.6% |
| Qwen/Qwen3-14B | dense | 75,888 | 76,882 | +1.3% |
| Qwen/Qwen3.8-27B-FP8 | hybrid + vision | 137,216 | 134,122 | −2.3% |
| Qwen/Qwen3.6-35B-A3B-FP8 | hybrid MoE + vision | 179,361 | 159,023 | −11.3% (cautious) |

- Weights and KV per token come straight from the config, so they're exact.
- Runtime overhead (activations, CUDA graphs, non-torch memory) is calibrated from these
  runs at vLLM's default `gpu_memory_utilization` of **0.92**: about 2.15 GiB for text models
  and 2.9 GiB when vLLM also profiles a vision encoder, plus a little more for each
  `--max-num-seqs` slot. Override it with `--overhead-gib`. (Before 0.4.0 gpu-fit assumed a
  0.9 default and a smaller overhead; predictions at the default are unchanged.)
- **vLLM 0.34 changed this.** On CUDA it now sizes the KV cache from memory actually free
  after warm-up ("extensible KV cache") and uses the whole GPU unless you set
  `--gpu-memory-utilization`. Not measured yet: expect more KV capacity than gpu-fit predicts;
  the estimate stays on the safe side. Measurements on 0.34 are next.
- **Hybrid models have a hard limit on `--max-num-seqs`.** Each request needs one
  recurrent-state slot. Ask for more slots than fit and vLLM refuses to start; gpu-fit
  tells you the limit first (it predicted 156 where vLLM reported 164).
- Treat the output as a planning estimate and confirm on the real GPU before you commit money.

**Help make it exact:** share what vLLM printed on your hardware (see [CONTRIBUTING.md](CONTRIBUTING.md)).
Every measurement improves the defaults for everyone.

## Roadmap

- [x] Memory model calibrated on real hardware (L40S)
- [x] Hybrid `--max-num-seqs` limit warning
- [x] Browser calculator
- [x] Decode speed estimate, MoE-aware
- [x] Shared-GPU budgeting (`--reserve-gib`)
- [ ] Re-measure on vLLM 0.34 (extensible KV cache)
- [ ] Workload mode: safe concurrency, throughput and cost per 1,000 requests
- [ ] Speed calibrated against measured vLLM throughput
- [ ] More GPUs measured (L4, H100) and models (Llama, Gemma, Mistral) from community results
- [ ] Prompt-processing (prefill) time and time-to-first-token
- [ ] Cost per million tokens for cloud GPUs
- [ ] SGLang and TensorRT-LLM memory models
- [ ] LoRA adapters and speculative decoding

## License

MIT
