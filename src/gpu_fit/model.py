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
    state_dtype_bytes: int = 2
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
        return self.linear_attn_layers * self.linear_state_elems * self.state_dtype_bytes // tp


def _linear_state_elems(c: dict) -> int:
    """Gated-DeltaNet style (Qwen3-Next family): conv state + recurrent matrix per layer."""
    kh, vh = c.get("linear_num_key_heads"), c.get("linear_num_value_heads")
    kd, vd = c.get("linear_key_head_dim"), c.get("linear_value_head_dim")
    if not all((kh, vh, kd, vd)):
        return 0
    conv_dim = 2 * kh * kd + vh * vd
    return conv_dim * (c.get("linear_conv_kernel_dim", 4) - 1) + vh * kd * vd


def _fetch_json(url: str) -> dict:
    headers = {"User-Agent": "gpu-fit"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return json.load(r)


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
    state = _linear_state_elems(c) if linear else 0
    if linear:
        notes.append(f"hybrid: {linear} linear-attention layers carry no per-token KV")
        notes.append("hybrid: vLLM pads KV pages to fit recurrent state, so real capacity can be lower; test before you commit")
        if not state:
            notes.append("recurrent state size unknown for this architecture; not counted")

    return ModelSpec(
        name=name, weight_bytes=weight_bytes, full_attn_layers=full, kv_heads=kv_heads,
        head_dim=head_dim, sliding_layers=sliding, sliding_window=window,
        linear_attn_layers=linear, linear_state_elems=state, mla_dim=mla,
        max_context=c.get("max_position_embeddings"), notes=notes,
    )


def load_model(model_id: str) -> ModelSpec:
    if os.path.isdir(model_id):
        with open(os.path.join(model_id, "config.json")) as f:
            config = json.load(f)
        size = sum(
            os.path.getsize(os.path.join(model_id, p))
            for p in os.listdir(model_id) if p.endswith(".safetensors")
        )
        return spec_from_config(os.path.basename(model_id.rstrip("/")), config, size)

    base = "https://huggingface.co"
    try:
        config = _fetch_json(f"{base}/{model_id}/resolve/main/config.json")
        info = _fetch_json(f"{base}/api/models/{model_id}?expand[]=safetensors")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not read {model_id} from Hugging Face ({e}). "
                         "Gated model? Set HF_TOKEN, or pass a local model folder.")
    counts = (info.get("safetensors") or {}).get("parameters")
    if not counts:
        raise SystemExit(f"{model_id}: no safetensors metadata on the Hub; pass a local folder.")
    return spec_from_config(model_id, config, weight_bytes_from_counts(counts))
