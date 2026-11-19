"""The vLLM memory model.

vLLM takes `gpu_memory_utilization` of the card, loads the weights, profiles a forward
pass (activations, CUDA graphs, non-torch allocations), and gives everything left to the
KV cache. We reproduce that budget.

Everything is derived from the model config and public vLLM behaviour. The one
assumption is the per-GPU runtime overhead, which you can override with --overhead-gib.
Measured results are welcome: see CONTRIBUTING.md.
"""

from dataclasses import dataclass
from typing import Optional

from .model import ModelSpec

GIB = 1024**3

# Activations + CUDA graphs + non-torch memory per GPU. An assumption, not a law:
# it grows with batch size and hidden size. Override with --overhead-gib.
DEFAULT_OVERHEAD_GIB = 2.0


@dataclass
class Fit:
    gpu_mem_gib: float
    tp: int
    budget_gib: float
    weights_gib: float
    overhead_gib: float
    kv_pool_gib: float
    kv_tokens: int
    context: int
    concurrent: float  # full-length requests that fit at once
    max_context_one_request: int

    @property
    def fits(self) -> bool:
        return self.kv_pool_gib > 0 and self.concurrent >= 1


def estimate(
    spec: ModelSpec,
    gpu_mem_gib: float,
    context: int,
    tp: int = 1,
    kv_dtype_bytes: int = 2,
    utilization: float = 0.90,
    overhead_gib: Optional[float] = None,
) -> Fit:
    overhead = DEFAULT_OVERHEAD_GIB if overhead_gib is None else overhead_gib
    budget = gpu_mem_gib * utilization
    weights = spec.weight_bytes / GIB / tp
    kv_pool = budget - weights - overhead

    full_tok, swa_tok = spec.kv_bytes_per_token(kv_dtype_bytes, tp)
    usable = max(kv_pool, 0) * GIB
    per_token = full_tok or swa_tok  # vLLM reports capacity in full-attention tokens

    window = spec.sliding_window or context
    state = spec.linear_state_bytes(tp)  # fixed per request, independent of length
    per_request = context * full_tok + min(context, window) * swa_tok + state
    concurrent = usable / per_request if per_request else float("inf")

    if full_tok:
        max_one = int((usable - min(context, window) * swa_tok - state) / full_tok)
    else:
        max_one = spec.max_context or context
    if spec.max_context:
        max_one = min(max_one, spec.max_context)

    return Fit(
        gpu_mem_gib=gpu_mem_gib, tp=tp, budget_gib=budget, weights_gib=weights,
        overhead_gib=overhead, kv_pool_gib=kv_pool,
        kv_tokens=int(usable / per_token) if per_token else 0,
        context=context, concurrent=concurrent, max_context_one_request=max(max_one, 0),
    )
