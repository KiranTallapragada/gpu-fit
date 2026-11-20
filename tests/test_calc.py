import unittest

from gpu_fit.calc import GIB, estimate, estimate_speed, weight_bytes_read
from gpu_fit.cli import parse_tokens
from gpu_fit.model import spec_from_config, weight_bytes_from_counts

DENSE = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
         "hidden_size": 4096, "max_position_embeddings": 131072}
HYBRID = {"text_config": {
    "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
    "head_dim": 256, "hidden_size": 5120, "full_attention_interval": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}}
MLA = {"num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128,
       "hidden_size": 7168, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
SWA = {"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 4,
       "head_dim": 128, "hidden_size": 1024, "sliding_window": 1024,
       "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]}


class TestModelSpec(unittest.TestCase):
    def test_dense_kv_per_token(self):
        s = spec_from_config("dense", DENSE, 0)
        # 2 (K,V) × 32 layers × 8 heads × 128 dim × 2 bytes
        self.assertEqual(s.kv_bytes_per_token(2), (131072, 0))
        self.assertEqual(s.kv_bytes_per_token(1), (65536, 0))

    def test_hybrid_counts_only_full_attention_layers(self):
        s = spec_from_config("hybrid", HYBRID, 0)
        self.assertEqual((s.full_attn_layers, s.linear_attn_layers), (16, 48))
        self.assertEqual(s.kv_bytes_per_token(2)[0], 2 * 16 * 4 * 256 * 2)
        conv = (2 * 16 * 128 + 48 * 128) * 3
        self.assertEqual(s.linear_state_elems, conv + 48 * 128 * 128)

    def test_mla_is_compact_and_replicated_across_tp(self):
        s = spec_from_config("mla", MLA, 0)
        self.assertEqual(s.kv_bytes_per_token(2)[0], 61 * 576 * 2)
        self.assertEqual(s.kv_bytes_per_token(2, tp=8), s.kv_bytes_per_token(2, tp=1))

    def test_kv_heads_split_then_replicate_beyond_head_count(self):
        s = spec_from_config("dense", DENSE, 0)
        self.assertEqual(s.kv_bytes_per_token(2, tp=2)[0], 131072 // 2)
        self.assertEqual(s.kv_bytes_per_token(2, tp=16)[0], 131072 // 8)  # one head each

    def test_sliding_window_layers(self):
        s = spec_from_config("swa", SWA, 0)
        self.assertEqual((s.full_attn_layers, s.sliding_layers, s.sliding_window), (1, 3, 1024))

    def test_weight_bytes_by_dtype(self):
        self.assertEqual(weight_bytes_from_counts({"BF16": 10, "F8_E4M3": 100, "F32": 1}), 124)


class TestEstimate(unittest.TestCase):
    def test_budget_arithmetic(self):
        s = spec_from_config("dense", DENSE, 16 * GIB)
        f = estimate(s, gpu_mem_gib=80, context=8192, overhead_gib=2.0)
        self.assertAlmostEqual(f.kv_pool_gib, 80 * 0.92 - 16 - 2.0)  # vLLM default 0.92
        self.assertEqual(f.kv_tokens, int(f.kv_pool_gib * GIB / 131072))
        self.assertAlmostEqual(f.concurrent, f.kv_tokens / 8192, places=2)

    def test_fp8_kv_doubles_capacity(self):
        s = spec_from_config("dense", DENSE, 16 * GIB)
        a = estimate(s, 80, 8192, kv_dtype_bytes=2)
        b = estimate(s, 80, 8192, kv_dtype_bytes=1)
        self.assertAlmostEqual(b.kv_tokens, a.kv_tokens * 2, delta=2)  # integer rounding

    def test_does_not_fit_when_weights_exceed_budget(self):
        s = spec_from_config("dense", DENSE, 60 * GIB)
        self.assertFalse(estimate(s, 48, 8192).fits)
        self.assertTrue(estimate(s, 48, 8192, tp=2).fits)

    def test_hybrid_state_reduces_concurrency(self):
        s = spec_from_config("hybrid", HYBRID, 28 * GIB)
        f = estimate(s, 45, 65536)
        no_state = f.kv_pool_gib * GIB / (65536 * s.kv_bytes_per_token(2)[0])
        self.assertLess(f.concurrent, no_state)

    def test_single_request_capped_by_model_max(self):
        s = spec_from_config("dense", DENSE, 1 * GIB)
        self.assertEqual(estimate(s, 192, 8192).max_context_one_request, 131072)

    def test_hybrid_block_matches_vllm_state_padding(self):
        s = spec_from_config("hybrid", HYBRID, 28 * GIB)
        # fp32 recurrent matrix + bf16 conv state per layer, over 4096 attention bytes/token
        conv = (2 * 16 * 128 + 48 * 128) * 3
        state = 48 * 128 * 128 * 4 + conv * 2
        self.assertEqual(s.hybrid_block_tokens(2), -(-(-(-state // 4096)) // 16) * 16)
        self.assertEqual(s.hybrid_block_tokens(2), 784)

    def test_hybrid_slots_block_too_many_seqs(self):
        s = spec_from_config("hybrid", HYBRID, 28 * GIB)
        f = estimate(s, 45, 32768, max_num_seqs=256)
        self.assertGreater(f.hybrid_slots, 0)
        self.assertFalse(f.slots_ok)
        self.assertFalse(f.fits)
        self.assertTrue(estimate(s, 45, 32768, max_num_seqs=32).slots_ok)

    def test_dense_models_have_no_slot_limit(self):
        s = spec_from_config("dense", DENSE, 16 * GIB)
        self.assertEqual(estimate(s, 80, 8192).hybrid_slots, 0)

    def test_overhead_by_model_type_and_slots(self):
        text = spec_from_config("dense", DENSE, 16 * GIB)
        vision = spec_from_config("vl", {**DENSE, "vision_config": {}}, 16 * GIB)
        a = estimate(text, 80, 8192, max_num_seqs=32).overhead_gib
        b = estimate(vision, 80, 8192, max_num_seqs=32).overhead_gib
        c = estimate(text, 80, 8192, max_num_seqs=256).overhead_gib
        self.assertLess(a, b)
        self.assertLess(a, c)

    def test_speed_dense_is_bandwidth_bound(self):
        s = spec_from_config("dense", DENSE, 16 * 10**9)
        fast = estimate_speed(s, 3350, 4096).single_tok_s
        slow = estimate_speed(s, 864, 4096).single_tok_s
        self.assertGreater(fast, 3 * slow)            # ~3.9x the bandwidth, minus fixed overhead
        self.assertLess(fast, 3350e9 / (16 * 10**9))   # never above the raw bandwidth limit

    def test_speed_batch_raises_total_lowers_per_user(self):
        s = spec_from_config("dense", DENSE, 16 * 10**9)
        one = estimate_speed(s, 864, 8192, users=1)
        eight = estimate_speed(s, 864, 8192, users=8)
        self.assertGreater(eight.total_tok_s, one.total_tok_s)
        self.assertLess(eight.per_user_tok_s, one.per_user_tok_s)

    def test_moe_reads_only_active_experts(self):
        moe = {**DENSE, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
               "vocab_size": 151936}
        s = spec_from_config("moe", moe, 60 * GIB)
        self.assertEqual((s.moe_experts, s.moe_topk), (128, 8))
        self.assertGreater(s.expert_share, 0.5)
        self.assertLess(weight_bytes_read(s, 1), 0.4 * s.weight_bytes)
        self.assertGreater(weight_bytes_read(s, 64), weight_bytes_read(s, 1))  # bigger batch, more experts

    def test_no_speed_without_bandwidth(self):
        self.assertIsNone(estimate_speed(spec_from_config("dense", DENSE, GIB), 0, 4096))

    def test_parse_tokens(self):
        self.assertEqual(parse_tokens("128k"), 131072)
        self.assertEqual(parse_tokens("32000"), 32000)


if __name__ == "__main__":
    unittest.main()


class TestReserve(unittest.TestCase):
    def _args(self, reserve=0.0, util=None):
        import argparse
        return argparse.Namespace(reserve_gib=reserve, util=util)

    def test_default_is_vllm_default(self):
        from gpu_fit.cli import resolve_util
        self.assertEqual(resolve_util(44.99, self._args()), 0.92)

    def test_reserve_sets_highest_accepted_util(self):
        from gpu_fit.cli import resolve_util
        # 44.99 GiB GPU, 12 GiB held by others, 0.5 GiB startup margin -> floor(32.49/44.99) = 0.72
        self.assertEqual(resolve_util(44.99, self._args(12)), 0.72)

    def test_reserve_never_raises_an_explicit_util(self):
        from gpu_fit.cli import resolve_util
        self.assertEqual(resolve_util(44.99, self._args(12, 0.5)), 0.5)
        self.assertEqual(resolve_util(44.99, self._args(12, 0.9)), 0.72)

    def test_reserve_bigger_than_gpu_leaves_nothing(self):
        from gpu_fit.cli import resolve_util
        self.assertEqual(resolve_util(44.99, self._args(50)), 0.0)

    def test_budget_never_exceeds_free_memory(self):
        from gpu_fit.cli import STARTUP_MARGIN_GIB, resolve_util
        for mem in (22.0, 44.99, 79.1, 139.8):
            for reserve in (0.5, 3, 10, 20):
                u = resolve_util(mem, self._args(reserve))
                if u > 0:
                    self.assertLessEqual(u * mem, mem - reserve - STARTUP_MARGIN_GIB + 1e-9)
