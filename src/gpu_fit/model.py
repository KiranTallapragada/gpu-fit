"""Read what memory maths needs from a Hugging Face model: its config.json and the
size of its weights (from the safetensors metadata, so quantised repos are exact)."""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


@dataclass
class ModelSpec:
    name: str
    weight_bytes: int
    full_attn_layers: int
    kv_heads: int
    head_dim: int
    sliding_layers: int = 0
    sliding_window: Optional[int] = None
    linear_attn_layers: int = 0  # Mamba / linear-attention layers: fixed state, no per-token KV
    linear_state_elems: int = 0  # recurrent state per request per linear layer (elements)
    linear_conv_elems: int = 0   # the convolution part of that state (kept in model dtype)
    state_dtype_bytes: int = 2
    multimodal: bool = False     # has a vision/audio encoder that vLLM also profiles
    moe_experts: int = 0         # routed experts per MoE layer (0 = dense model)
    moe_topk: int = 0            # experts each token uses
    expert_share: float = 0.0    # share of weight bytes that are routed experts
    mla_dim: Optional[int] = None  # DeepSeek-style latent KV: per-layer per-token width
    max_context: Optional[int] = None
    notes: list = field(default_factory=list)

    @property
    def hybrid(self) -> bool:
        return self.linear_attn_layers > 0

    def kv_bytes_per_token(self, kv_dtype_bytes: int, tp: int = 1) -> tuple:
        """(full-attention bytes/token, sliding-window bytes/token) on ONE GPU."""
        if self.mla_dim:  # latent KV is replicated across tensor-parallel ranks
            return self.full_attn_layers * self.mla_dim * kv_dtype_bytes, 0
        heads = max(1, self.kv_heads // tp) if self.kv_heads >= tp else 1
        per_layer = 2 * heads * self.head_dim * kv_dtype_bytes
        return self.full_attn_layers * per_layer, self.sliding_layers * per_layer


    def linear_state_bytes(self, tp: int = 1) -> int:
        """Recurrent state one request holds across all linear-attention layers, one GPU."""
        return self.linear_attn_layers * self.linear_state_layer_bytes() // tp

    def linear_state_layer_bytes(self) -> int:
        """vLLM keeps the recurrent matrix in fp32 and the conv state in the model dtype."""
        ssm = self.linear_state_elems - self.linear_conv_elems
        return ssm * 4 + self.linear_conv_elems * self.state_dtype_bytes

    def hybrid_block_tokens(self, kv_dtype_bytes: int, tp: int = 1) -> int:
        """vLLM sizes a hybrid model's KV block so one attention page holds one layer's
        recurrent state: block tokens = state bytes / attention bytes per token, rounded up to 16."""
        full, _ = self.kv_bytes_per_token(kv_dtype_bytes, tp)
        if not (self.hybrid and full and self.linear_state_elems):
            return 0
        per_layer_token = full // self.full_attn_layers
        raw = -(-self.linear_state_layer_bytes() // tp // per_layer_token)
        return -(-raw // 16) * 16


def _linear_state_elems(c: dict) -> tuple:
    """Gated-DeltaNet style (Qwen3-Next family): (total, conv) state elements per layer."""
    kh, vh = c.get("linear_num_key_heads"), c.get("linear_num_value_heads")
    kd, vd = c.get("linear_key_head_dim"), c.get("linear_value_head_dim")
    if not all((kh, vh, kd, vd)):
        return 0, 0
    conv = (2 * kh * kd + vh * vd) * (c.get("linear_conv_kernel_dim", 4) - 1)
    return conv + vh * kd * vd, conv


def _moe(c: dict) -> tuple:
    """(experts, top-k, share of parameters in routed experts), estimated from the config."""
    experts = c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts") or 0
    topk = c.get("num_experts_per_tok") or 0
    if not (experts and topk):
        return 0, 0, 0.0
    hidden, layers = c["hidden_size"], c["num_hidden_layers"]
    inter = c.get("moe_intermediate_size") or c.get("intermediate_size")
    dense = (c.get("first_k_dense_replace") or 0) + len(c.get("mlp_only_layers") or [])
    moe_layers = layers - dense
    routed = moe_layers * experts * 3 * hidden * inter
    heads = c.get("num_attention_heads") or 1
    kv = c.get("num_key_value_heads") or heads
    hd = c.get("head_dim") or hidden // heads
    attn = layers * (2 * hidden * heads * hd + 2 * hidden * kv * hd)
    shared_inter = (c.get("shared_expert_intermediate_size")
                    or (c.get("n_shared_experts") or 0) * inter)
    shared = moe_layers * 3 * hidden * shared_inter + dense * 3 * hidden * c.get("intermediate_size", inter)
    emb = c.get("vocab_size", 0) * hidden * (1 if c.get("tie_word_embeddings") else 2)
    return experts, topk, routed / (routed + attn + shared + emb)


def _fetch_json(url: str) -> dict:
    headers = {"User-Agent": "gpu-fit"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r)


def hub_weight_bytes(base: str, model_id: str):
    """Exact on-disk size of the weights, which is what lands in GPU memory.

    Parameter counts by dtype mislead for packed formats (MXFP4, AWQ, GPTQ), and some
    safetensors indexes carry a wrong total_size, so trust the weight files' own sizes first
    (skipping duplicate `consolidated` copies), then the index."""
    try:
        files = _fetch_json(f"{base}/api/models/{model_id}/tree/main")
        sizes = [f.get("size", 0) for f in files if f.get("path", "").endswith(".safetensors")
                 and not f["path"].startswith("consolidated")]
        if sum(sizes):
            return sum(sizes)
    except Exception:  # noqa: BLE001
        pass
    try:
        idx = _fetch_json(f"{base}/{model_id}/resolve/main/model.safetensors.index.json")
        return int((idx.get("metadata") or {}).get("total_size") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def weight_bytes_from_counts(counts: dict) -> int:
    return sum(n * DTYPE_BYTES.get(dt.upper(), 2) for dt, n in counts.items())


def spec_from_config(name: str, config: dict, weight_bytes: int) -> ModelSpec:
    c = config.get("text_config") or config.get("llm_config") or config
    layers = c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    if not layers:
        raise SystemExit(f"{name}: config has no num_hidden_layers; pass a text model config.")
    heads = c.get("num_attention_heads") or c.get("n_head")
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or (c["hidden_size"] // heads)
    notes = []

    full, sliding, linear = layers, 0, 0
    types = c.get("layer_types")
    if types:
        full = sum(t == "full_attention" for t in types)
        sliding = sum(t in ("sliding_attention", "sliding_window") for t in types)
        linear = len(types) - full - sliding
    elif c.get("full_attention_interval"):
        full = layers // c["full_attention_interval"]
        linear = layers - full
    elif c.get("attn_layer_indices") is not None:  # Jamba-style
        full = len(c["attn_layer_indices"])
        linear = layers - full

    window = c.get("sliding_window") if sliding or c.get("use_sliding_window") else None
    if window and not sliding and c.get("use_sliding_window"):
        sliding, full = full, 0

    mla = None
    if c.get("kv_lora_rank"):
        mla = c["kv_lora_rank"] + c.get("qk_rope_head_dim", 64)
        notes.append("MLA latent KV: compact, but replicated on every tensor-parallel GPU")
    state, conv = _linear_state_elems(c) if linear else (0, 0)
    if linear:
        notes.append(f"hybrid: {linear} linear-attention layers carry no per-token KV")
        if not state:
            notes.append("recurrent state size unknown for this architecture; not counted")

    experts, topk, share = _moe(c)
    return ModelSpec(
        moe_experts=experts, moe_topk=topk, expert_share=round(share, 4),
        name=name, weight_bytes=weight_bytes, full_attn_layers=full, kv_heads=kv_heads,
        head_dim=head_dim, sliding_layers=sliding, sliding_window=window,
        linear_attn_layers=linear, linear_state_elems=state, linear_conv_elems=conv,
        multimodal=any(k in config for k in ("vision_config", "audio_config", "vision_tower")),
        mla_dim=mla,
        max_context=c.get("max_position_embeddings"), notes=notes,
    )


def load_model(model_id: str) -> ModelSpec:
    if os.path.isdir(model_id):
        with open(os.path.join(model_id, "config.json")) as f:
            config = json.load(f)
        size = sum(
            os.path.getsize(os.path.join(model_id, p))
            for p in os.listdir(model_id)
            if p.endswith(".safetensors") and not p.startswith("consolidated")
        )
        return spec_from_config(os.path.basename(model_id.rstrip("/")), config, size)

    base = "https://huggingface.co"
    try:
        config = _fetch_json(f"{base}/{model_id}/resolve/main/config.json")
        info = _fetch_json(f"{base}/api/models/{model_id}?expand[]=safetensors")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not read {model_id} from Hugging Face ({e}). "
                         "Gated model? Set HF_TOKEN, or pass a local model folder.")
    size = hub_weight_bytes(base, model_id)
    if not size:
        counts = (info.get("safetensors") or {}).get("parameters")
        if not counts:
            raise SystemExit(f"{model_id}: no safetensors metadata on the Hub; pass a local folder.")
        size = weight_bytes_from_counts(counts)
    return spec_from_config(model_id, config, size)
