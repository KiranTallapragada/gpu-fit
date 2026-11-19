# gpu-fit

**Will this LLM fit on this GPU, at this context length, for this many users?**

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
    runtime overhead (assumed)      2.0 GiB
    left for KV cache              44.2 GiB
  KV cache capacity              675,671 tokens
  Requests at full context       5.2
  Longest single request         163,840 tokens

✓ Fits. Suggested command:
  vllm serve deepseek-ai/DeepSeek-V3 --max-model-len 131072 --tensor-parallel-size 8 --max-num-seqs 5
```

## Install

```
pip install gpu-fit        # no dependencies, Python 3.9+
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
| FP8 KV cache | Doubles capacity: `--kv-cache-dtype fp8` | ✓ |

## Options

```
gpu-fit MODEL [--gpu L40S | --gpu-mem-gib 48] [--tp 2] [--context 128k]
              [--concurrency 8] [--kv-cache-dtype fp8] [--util 0.9]
              [--overhead-gib 2.0] [--json]
```

- `MODEL`: a Hugging Face id, or a local folder with `config.json` and `*.safetensors`.
  Gated models need `HF_TOKEN` in the environment.
- Leave out `--gpu` to compare every GPU in the catalog with the smallest tensor-parallel size that works.
- `--json` for scripts and CI.

## How the maths works

vLLM takes `gpu_memory_utilization` × GPU memory, loads the weights, profiles a forward pass
(activations, CUDA graphs), and gives the rest to the KV cache. `gpu-fit` mirrors that:

```
KV pool   = memory × utilization − weights / tp − overhead
KV/token  = 2 × full-attention layers × KV heads per GPU × head dim × dtype bytes
requests  = KV pool / (context × KV/token + sliding-window KV + recurrent state)
```

## Accuracy, honestly

- Weights and KV per token come straight from the config, so they're exact.
- **Runtime overhead** (activations, CUDA graphs) is an assumption: 2 GiB per GPU by default.
  It grows with batch size and model width. Override it with `--overhead-gib`.
- **Hybrid models:** vLLM pads KV pages to hold the recurrent state, so real capacity can be lower than shown.
- Treat the output as a planning estimate. Confirm on the real GPU before you commit money.

**Help make it exact:** share what vLLM printed on your hardware (see [CONTRIBUTING.md](CONTRIBUTING.md)).
Every measurement improves the defaults for everyone.

## Roadmap

- [ ] Measured overhead table from community results
- [ ] Throughput estimate (tokens/s) from memory bandwidth
- [ ] SGLang and TensorRT-LLM memory models
- [ ] Web version
- [ ] Vision encoder and LoRA memory

## License

MIT
