# Contributing

Thanks for helping. The most valuable contribution is a **real measurement**.

## Share a measurement (5 minutes)

Start vLLM with your model as usual, then copy these lines from its log:

```
Model loading took XX.X GiB memory
Available KV cache memory: XX.XX GiB
GPU KV cache size: XXX,XXX tokens
```

Open an issue titled `measurement: <model> on <GPU>` with:

- those three lines;
- the model id, GPU, number of GPUs (`--tensor-parallel-size`);
- your `vllm serve` flags (`--max-model-len`, `--kv-cache-dtype`, `--gpu-memory-utilization`, `--max-num-seqs`);
- the vLLM version (`vllm --version`);
- what `gpu-fit` predicted for the same settings.

Only share public models and your own hardware details. No prompts, keys or private data.

## Code

```
git clone https://github.com/KiranTallapragada/gpu-fit && cd gpu-fit
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m gpu_fit Qwen/Qwen3-32B --gpu H100
```

- No runtime dependencies: standard library only.
- Every change to the maths needs a test that shows the arithmetic.
- New GPU? Add it to `src/gpu_fit/gpus.py` with the memory `nvidia-smi` reports.
