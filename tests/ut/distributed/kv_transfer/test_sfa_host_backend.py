# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_backend_module():
    path = (
        Path(__file__).parents[4]
        / "vllm_ascend"
        / "distributed"
        / "kv_transfer"
        / "kv_offload_decode"
        / "host_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sfa_host_backend_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BACKEND = _load_backend_module()


class TestSFAHostBackend(unittest.TestCase):

    def test_backend_defaults_to_memfabric(self):
        self.assertEqual(
            BACKEND.resolve_sfa_kv_offload_backend(
                None,
                use_fused_overlap_offload=True,
            ),
            BACKEND.SFA_KV_OFFLOAD_BACKEND_MEMFABRIC,
        )

    def test_fused_overlap_can_select_mooncake(self):
        self.assertEqual(
            BACKEND.resolve_sfa_kv_offload_backend(
                {"sfa_kv_offload_backend": " MoonCake "},
                use_fused_overlap_offload=True,
            ),
            BACKEND.SFA_KV_OFFLOAD_BACKEND_MOONCAKE,
        )

    def test_mooncake_requires_fused_overlap(self):
        with self.assertLogs(BACKEND.logger, level="WARNING") as captured:
            backend = BACKEND.resolve_sfa_kv_offload_backend(
                {"sfa_kv_offload_backend": "mooncake"},
                use_fused_overlap_offload=False,
            )
        self.assertEqual(
            backend,
            BACKEND.SFA_KV_OFFLOAD_BACKEND_MEMFABRIC,
        )
        self.assertTrue(
            any(
                "requires fused overlap" in line
                for line in captured.output
            )
        )

    def test_invalid_backend_raises(self):
        for value in ("rdma", "swapped"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "sfa_kv_offload_backend",
                ):
                    BACKEND.resolve_sfa_kv_offload_backend(
                        {"sfa_kv_offload_backend": value},
                        use_fused_overlap_offload=True,
                    )

    def test_extracts_mapping_from_vllm_config(self):
        extra = {"sfa_kv_offload_backend": "mooncake"}
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config=extra,
            )
        )
        self.assertIs(
            BACKEND.kv_transfer_extra_config(config),
            extra,
        )
        self.assertIsNone(
            BACKEND.kv_transfer_extra_config(
                SimpleNamespace(kv_transfer_config=None)
            )
        )


if __name__ == "__main__":
    unittest.main()
