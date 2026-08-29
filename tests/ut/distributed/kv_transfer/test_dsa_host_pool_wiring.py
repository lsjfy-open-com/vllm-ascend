# SPDX-License-Identifier: Apache-2.0
"""Dependency-free wiring contracts for the runner-owned DSA Host pool."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[4]
RUNNER_PATH = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
MANAGER_PATH = (
    ROOT
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "kv_offload_decode"
    / "kv_offload_decode_manager.py"
)
CONNECTOR_PATH = (
    ROOT
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "kv_p2p"
    / "mooncake_layerwise_to_dram_connector.py"
)
MULTI_CONNECTOR_PATH = (
    ROOT
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "ascend_multi_connector.py"
)
CPP_PATH = (
    ROOT
    / "vllm_ascend"
    / "distributed"
    / "kv_transfer"
    / "kv_offload_decode"
    / "kv_offload_decode.cpp"
)


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                segment = ast.get_source_segment(source, node)
                assert segment is not None
                return segment
    raise AssertionError(f"function {function_name!r} not found in {path}")


class TestDSAHostPoolWiring(unittest.TestCase):

    def test_runner_allocates_pool_before_cache_binding(self):
        source = _function_source(
            RUNNER_PATH,
            "initialize_kv_cache_tensors",
        )
        allocate_pos = source.index(
            "self._allocate_fused_overlap_host_main"
        )
        bind_pos = source.index(
            "if self.model_config.hf_text_config.model_type == "
            '"deepseek_v4"'
        )
        self.assertLess(allocate_pos, bind_pos)

    def test_runner_registers_allocated_pool_after_cache_binding(self):
        source = _function_source(RUNNER_PATH, "initialize_kv_cache")
        cache_pos = source.index("self.initialize_kv_cache_tensors")
        pool_pos = source.index("runner_host_pool = self.dsa_host_kv_pool")
        register_pos = source.index(
            "self.kv_offload_decode_manager.register_kv_caches"
        )
        self.assertLess(cache_pos, pool_pos)
        self.assertLess(pool_pos, register_pos)
        self.assertIn("connector.bind_runner_host_pool", source)

    def test_runner_binds_pool_views_into_six_tuple(self):
        source = _function_source(
            RUNNER_PATH,
            "_allocate_fused_overlap_host_main",
        )
        self.assertIn("DSAHostKVPool.allocate", source)
        self.assertIn("manager.bind_runner_host_pool(pool)", source)
        self.assertIn("pool.k_caches[layer_id]", source)
        self.assertIn("pool.v_caches[layer_id]", source)
        self.assertIn("owner_rank=0", source)

    def test_runner_skips_per_layer_host_alloc_for_mooncake(self):
        source = _function_source(
            RUNNER_PATH,
            "_allocate_kv_cache_tensors_for_kv_offload_decode",
        )
        self.assertIn(
            "not self.kv_offload_decode_manager.uses_mooncake_host",
            source,
        )

    def test_manager_selects_backend_from_connector_config(self):
        source = _function_source(MANAGER_PATH, "__init__")
        self.assertIn("resolve_sfa_kv_offload_backend", source)
        self.assertIn("kv_transfer_extra_config(vllm_config)", source)
        self.assertIn("ensure_mooncake_host_is_pd_decode_only", source)
        self.assertIn("keep_device_kv_cache=", source)
        mooncake_pos = source.index("if self.uses_mooncake_host")
        initialize_pos = source.index("offload.initialize(config)")
        self.assertLess(mooncake_pos, initialize_pos)

    def test_manager_uses_each_rank_local_shared_segment_address(self):
        source = _function_source(MANAGER_PATH, "register_kv_caches")
        mooncake_branch = source.split(
            "if self.uses_mooncake_host:",
            maxsplit=2,
        )[2].split("else:", maxsplit=1)[0]
        self.assertIn(
            "self.gvas_k_bases.append(k_cpu.data_ptr())",
            mooncake_branch,
        )
        self.assertIn(
            "self.gvas_v_bases.append(v_cpu.data_ptr())",
            mooncake_branch,
        )
        self.assertNotIn("broadcast(", mooncake_branch)

    def test_mooncake_current_kv_index_copy_is_captured_on_main_stream(self):
        init_source = _function_source(MANAGER_PATH, "__init__")
        offload_source = _function_source(MANAGER_PATH, "offload_new_kv")
        copy_source = _function_source(
            MANAGER_PATH,
            "_offload_new_kv_via_index_copy",
        )
        cpp_source = CPP_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "self.d2h_index_copy_bypass = self.uses_mooncake_host",
            init_source,
        )
        self.assertIn("and not self.d2h_index_copy_bypass", offload_source)
        self.assertIn(
            "enqueue_current_kv_index_copy_descriptors",
            copy_source,
        )
        self.assertIn("self.d2h_src_idx_npu.copy_", copy_source)
        self.assertIn("self.d2h_dst_idx_npu.copy_", copy_source)
        self.assertIn("flat_host_k.index_copy_", copy_source)
        self.assertIn("flat_host_v.index_copy_", copy_source)
        self.assertIn("aclrtLaunchHostFunc", cpp_source)
        self.assertIn(
            "current_kv_index_copy_descriptor_callback",
            cpp_source,
        )

    def test_manager_requires_matching_pool_topology(self):
        source = _function_source(
            MANAGER_PATH,
            "bind_runner_host_pool",
        )
        self.assertIn("pool.topology.tp_rank != self.tp_rank", source)
        self.assertIn("pool.topology.tp_size != self.tp_size", source)
        self.assertIn("pool.layout.block_size != self.block_size", source)

    def test_mooncake_membership_has_operator_and_planner_storage(self):
        source = _function_source(
            MANAGER_PATH,
            "allocate_fused_overlap_membership_map",
        )
        self.assertIn("allocate_mooncake_host_region", source)
        self.assertIn(
            "self.fused_overlap_planner_membership_map = planner_map",
            source,
        )
        self.assertIn("device=\"cpu\"", source)
        self.assertIn("pin_memory=True", source)
        sync_pos = source.index(
            "torch_npu.npu.current_stream().synchronize()"
        )
        barrier_pos = source.index("self.tp_group.barrier()", sync_pos)
        self.assertLess(sync_pos, barrier_pos)

    def test_external_plan_publishes_planner_output(self):
        source = _function_source(
            MANAGER_PATH,
            "prepare_fused_overlap_external_plan",
        )
        self.assertIn("planner_storage.data_ptr()", source)
        self.assertIn("plan_storage.copy_", source)
        self.assertIn("publish_plan(non_blocking=True)", source)
        self.assertIn(
            "current_stream().wait_stream",
            source,
        )

    def test_d2rh_connector_registers_shared_pool_on_owner(self):
        source = _function_source(CONNECTOR_PATH, "register_kv_caches")
        self.assertIn("is_main_owner = pool.is_owner", source)
        self.assertIn("pool.register(self.engine)", source)
        self.assertIn("k_caches_cpu = pool.k_caches", source)
        self.assertIn("v_caches_cpu = pool.v_caches", source)
        self.assertNotIn("_reg(k_cpu", source)
        self.assertNotIn("_reg(v_cpu", source)

    def test_d2rh_main_transfer_has_single_writer(self):
        source = _function_source(
            CONNECTOR_PATH,
            "get_transfer_meta_asymmetric",
        )
        self.assertIn("is_main_sender = self.tp_rank == 0", source)
        self.assertIn("skip_main = not is_main_sender", source)
        connector = CONNECTOR_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "Prefill TP0 → Decode TP0 registered shared Host pool",
            connector,
        )
        self.assertNotIn("each Decode TP local Main HOST", connector)

    def test_multi_connector_forwards_host_pool_binding(self):
        source = _function_source(
            MULTI_CONNECTOR_PATH,
            "bind_runner_host_pool",
        )
        self.assertIn("for connector in self._connectors", source)
        self.assertIn("bind_pool(pool)", source)


if __name__ == "__main__":
    unittest.main()
