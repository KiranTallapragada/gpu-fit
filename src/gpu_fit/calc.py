"""The vLLM memory model.

vLLM takes `gpu_memory_utilization` of the card, loads the weights, profiles a forward
pass (activations, CUDA graphs, non-torch allocations), and gives everything left to the
KV cache. We reproduce that budget.

Weights and KV per token come from the model config. Runtime overhead and the hybrid
padding factor are calibrated against real vLLM runs (bench/results/). More measurements
make them better: see CONTRIBUTING.md.
"""

from dataclasses import dataclass
from typing import Optional

from .model import ModelSpec

GIB = 1024**3

# Activations + CUDA graphs + non-torch memory per GPU, measured at --max-num-seqs 32
# (bench/results/l40s-vllm0.31.0.tsv): ~1.1-1.4 GiB for text models, ~2.0 GiB when vLLM
# also profiles a vision encoder. Override with --overhead-gib.
OVERHEAD_TEXT_GIB = 1.25
OVERHEAD_MULTIMODAL_GIB = 2.0
# CUDA-graph capture grows with --max-num-seqs: 256 slots cost ~0.85 GiB more than 32.
OVERHEAD_PER_SEQ_GIB = 0.0038
BASE_SEQS = 32
# Hybrid models: vLLM pads attention pages to hold recurrent state, so it reports fewer
# KV tokens than attention-only maths. Measured 0.87 and 0.81 on two Qwen hybrids.
HYBRID_EFFICIENCY = 0.84
VLLM_DEFAULT_MAX_NUM_SEQS = 256


def default_overhead(spec: ModelSpec, max_num_seqs: int) -> float:
    base = OVERHEAD_MULTIMODAL_GIB if spec.multimodal else OVERHEAD_TEXT_GIB
    return base + OVERHEAD_PER_SEQ_GIB * (max_num_seqs - BASE_SEQS)


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
    max_num_seqs: int = VLLM_DEFAULT_MAX_NUM_SEQS
    hybrid_slots: int = 0  # hybrid models: recurrent-state slots vLLM can hold (0 = n/a)

    @property
    def slots_ok(self) -> bool:
        """vLLM refuses to start a hybrid model when --max-num-seqs exceeds its state slots."""
        return not self.hybrid_slots or self.max_num_seqs <= self.hybrid_slots

    @property
    def fits(self) -> bool:
        return self.kv_pool_gib > 0 and self.concurrent >= 1 and self.slots_ok


def estimate(
    spec: ModelSpec,
    gpu_mem_gib: float,
    context: int,
    tp: int = 1,
    kv_dtype_bytes: int = 2,
    utilization: float = 0.90,
    overhead_gib: Optional[float] = None,
    max_num_seqs: int = VLLM_DEFAULT_MAX_NUM_SEQS,
) -> Fit:
    overhead = default_overhead(spec, max_num_seqs) if overhead_gib is None else overhead_gib
    budget = gpu_mem_gib * utilization
    weights = spec.weight_bytes / GIB / tp
    kv_pool = budget - weights - overhead

    full_tok, swa_tok = spec.kv_bytes_per_token(kv_dtype_bytes, tp)
    usable = max(kv_pool, 0) * GIB * (HYBRID_EFFICIENCY if spec.hybrid else 1.0)
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

    kv_tokens = int(usable / per_token) if per_token else 0
    block = spec.hybrid_block_tokens(kv_dtype_bytes, tp)
    return Fit(
        gpu_mem_gib=gpu_mem_gib, tp=tp, budget_gib=budget, weights_gib=weights,
        overhead_gib=overhead, kv_pool_gib=kv_pool, kv_tokens=kv_tokens,
        context=context, concurrent=concurrent, max_context_one_request=max(max_one, 0),
        max_num_seqs=max_num_seqs, hybrid_slots=kv_tokens // block if block else 0,
    )


# Decode speed. Generating a token is memory-bound: every step reads the weights it needs plus
# each active request's KV cache. vLLM typically reaches ~75% of the card's bandwidth on decode;
# tensor parallel loses a little more to GPU-to-GPU communication. A planning estimate, not a
# benchmark: prefill (reading the prompt) and scheduling are not modelled.
BANDWIDTH_EFFICIENCY = 0.75
TP_COMM_EFFICIENCY = 0.85
# Fixed cost per layer per decode step (kernel launches, expert routing, all-reduce). It is what
# caps small-active MoE models well below their bandwidth limit.
LAYER_OVERHEAD_S = 15e-6
MOE_LAYER_OVERHEAD_S = 60e-6
TP_LAYER_OVERHEAD_S = 20e-6


@dataclass
class Speed:
    single_tok_s: float      # one request alone, at the given context
    batch: int               # requests decoding together
    total_tok_s: float       # all requests together
    per_user_tok_s: float    # each request when `batch` run at once


def weight_bytes_read(spec: ModelSpec, batch: int) -> float:
    """Bytes of weights one decode step reads. MoE models only touch the experts in use."""
    if not spec.moe_experts:
        return spec.weight_bytes
    p = spec.moe_topk / spec.moe_experts
    touched = 1 - (1 - p) ** max(batch, 1)  # share of experts hit by at least one request
    return spec.weight_bytes * ((1 - spec.expert_share) + spec.expert_share * touched)


def estimate_speed(spec: ModelSpec, bandwidth_gbs: float, context: int, tp: int = 1,
                   kv_dtype_bytes: int = 2, users: int = 1) -> Optional[Speed]:
    if not bandwidth_gbs:
        return None
    full, swa = spec.kv_bytes_per_token(kv_dtype_bytes, tp)
    window = spec.sliding_window or context
    kv_per_request = context * full + min(context, window) * swa + spec.linear_state_bytes(tp)
    bw = bandwidth_gbs * 1e9 * BANDWIDTH_EFFICIENCY * (TP_COMM_EFFICIENCY if tp > 1 else 1.0)

    layers = spec.full_attn_layers + spec.sliding_layers + spec.linear_attn_layers
    fixed = layers * ((MOE_LAYER_OVERHEAD_S if spec.moe_experts else LAYER_OVERHEAD_S)
                      + (TP_LAYER_OVERHEAD_S if tp > 1 else 0.0))

    def step_seconds(batch: int) -> float:
        return (weight_bytes_read(spec, batch) / tp + batch * kv_per_request) / bw + fixed

    batch = max(users, 1)
    single = 1 / step_seconds(1)
    total = batch / step_seconds(batch)
    return Speed(single_tok_s=single, batch=batch, total_tok_s=total, per_user_tok_s=total / batch)
