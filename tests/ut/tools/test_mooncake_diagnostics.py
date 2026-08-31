# SPDX-License-Identifier: Apache-2.0
"""Run with unittest directly; no torch, vLLM, NPU, pytest or network required."""

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestMooncakeDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[3] / ".agents/skills/ascend-mooncake-diagnostics/scripts/collect.py"
        )
        spec = importlib.util.spec_from_file_location("mooncake_diagnostics_under_test", cls.script)
        cls.diag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.diag)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.key = b"a" * 32

    def parse(self, text):
        path = self.root / "source.log"
        path.write_text(text)
        collector = self.diag.LogCollector(self.key, "P")
        coverage = collector.read_log(path, 0, self.diag.MAX_LOG_BYTES)
        return collector, coverage

    @staticmethod
    def trace(rank=0, message="tuple index out of range"):
        return (
            f"(Worker_TP{rank} pid=123) Traceback (most recent call last):\n"
            f'(Worker_TP{rank} pid=123)   File "/private/secret/mooncake_layerwise_connector.py", '
            "line 1264, in create_kv_buffer\n"
            f"(Worker_TP{rank} pid=123)     first_v_cache = first_kv_cache_tuple[1]\n"
            f"(Worker_TP{rank} pid=123) IndexError: {message}\n"
        )

    def test_single_error_preserves_cause_without_message_or_path(self):
        collector, coverage = self.parse(self.trace(message="tuple index out of range secret-prompt 10.9.8.7"))
        error = next(iter(collector.errors.values()))
        self.assertEqual(error["chain"][0]["reason"], "tuple_index_out_of_range")
        self.assertEqual(error["chain"][0]["frames"][0]["function"], "create_kv_buffer")
        self.assertFalse(error["incomplete"])
        text = json.dumps(error)
        for forbidden in ("secret", "10.9.8.7", "first_v_cache", "123"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(coverage["byte_limit_reached"])

    def test_repeated_rank_errors_are_deduplicated(self):
        collector, _ = self.parse(self.trace(0) + self.trace(3))
        self.assertEqual(len(collector.errors), 1)
        error = next(iter(collector.errors.values()))
        self.assertEqual(error["occurrences"], 2)
        self.assertEqual(error["ranks"], [{"tp": 0}, {"tp": 3}])

    def test_chained_exception_keeps_inner_cause(self):
        text = (
            self.trace() + "(Worker_TP0 pid=123) The above exception was the direct cause of the following exception:\n"
        )
        text += (
            "(Worker_TP0 pid=123) Traceback (most recent call last):\n"
            '(Worker_TP0 pid=123)   File "/root/core.py", line 42, in run\n'
            "(Worker_TP0 pid=123) RuntimeError: Engine core initialization failed secret-data\n"
        )
        collector, _ = self.parse(text)
        chain = next(iter(collector.errors.values()))["chain"]
        self.assertEqual([part["type"] for part in chain], ["IndexError", "RuntimeError"])
        self.assertNotIn("secret-data", json.dumps(chain))

    def test_other_error_message_does_not_overwrite_closed_trace(self):
        collector, _ = self.parse(self.trace() + "RuntimeError: unrelated worker exited\n")
        self.assertEqual(next(iter(collector.errors.values()))["chain"][0]["type"], "IndexError")

    def test_interleaved_rank_traces_are_not_mixed(self):
        first = self.trace(0).splitlines(True)
        second = self.trace(1).replace("create_kv_buffer", "initialize_kv_cache").splitlines(True)
        collector, _ = self.parse("".join(a + b for a, b in zip(first, second)))
        self.assertEqual(len(collector.errors), 2)
        for error in collector.errors.values():
            self.assertEqual(len(error["chain"][0]["frames"]), 1)

    def test_unprefixed_frames_follow_only_unambiguous_header(self):
        text = (
            self.trace().splitlines(True)[0]
            + '  File "/root/core.py", line 1, in run\nIndexError: list index out of range\n'
        )
        collector, _ = self.parse(text)
        error = next(iter(collector.errors.values()))
        self.assertEqual(error["chain"][0]["type"], "IndexError")
        self.assertEqual(error["ranks"], [{"tp": 0}])

    def test_ambiguous_unprefixed_frames_mark_incomplete(self):
        collector, _ = self.parse(
            self.trace(0).splitlines(True)[0]
            + self.trace(1).splitlines(True)[0]
            + '  File "/private/core.py", line 9, in run\n'
        )
        self.assertEqual(collector.counts["ambiguous_unprefixed_trace_lines"], 1)
        self.assertTrue(all(error["incomplete"] for error in collector.errors.values()))

    def test_rank_parser_includes_dp_tp_from_combined_prefix(self):
        self.assertEqual(self.diag.rank_of("(Worker_DP2_TP3 pid=99)"), {"dp": 2, "tp": 3})

    def test_unknown_frame_identifiers_never_export(self):
        text = (
            self.trace()
            .replace("mooncake_layerwise_connector.py", "API_SECRET_CANARY.py")
            .replace("create_kv_buffer", "CREDENTIAL_CANARY")
        )
        collector, _ = self.parse(text)
        output = json.dumps(list(collector.errors.values()))
        self.assertNotIn("CANARY", output)
        self.assertIn("other_file", output)
        self.assertIn("other_function", output)

    def test_frame_budget_keeps_final_failing_frame(self):
        text = "Traceback (most recent call last):\n"
        text += "".join(f'  File "/root/core.py", line {number}, in run\n' for number in range(80))
        text += (
            '  File "/root/mooncake_layerwise_connector.py", line 1264, in create_kv_buffer\n'
            "IndexError: tuple index out of range\n"
        )
        collector, _ = self.parse(text)
        error = next(iter(collector.errors.values()))
        self.assertTrue(error["incomplete"])
        self.assertEqual(len(error["chain"][0]["frames"]), self.diag.MAX_FRAMES)
        self.assertEqual(error["chain"][0]["frames"][-1]["function"], "create_kv_buffer")

    def test_unfinished_trace_is_not_claimed_complete(self):
        collector, _ = self.parse("Traceback (most recent call last):\n")
        self.assertTrue(next(iter(collector.errors.values()))["incomplete"])

    def test_event_budget_keeps_late_ready_and_counts(self):
        collector, _ = self.parse("Starting to load model SECRET\n" * 110 + "Application startup complete.\n")
        self.assertEqual(collector.counts["model_load_start"], 110)
        self.assertEqual(collector.events[-1]["event"], "api_ready")
        self.assertEqual(len(collector.events), self.diag.MAX_EVENTS)
        self.assertNotIn("SECRET", json.dumps(collector.events))

    def test_request_correlation_across_engine_suffix_and_keys(self):
        request = "cmpl-12345678-1234-1234-1234-123456789abc-0"
        external = self.diag.request_alias(f"req={request}", self.key)
        internal = self.diag.request_alias(f"req={request}-abcD1234", self.key)
        self.assertEqual(external, internal)
        self.assertNotEqual(external, self.diag.request_alias(f"req={request}", b"b" * 32))
        self.assertNotIn(request, external)

    def test_generic_ids_are_not_blindly_trimmed(self):
        self.assertNotEqual(
            self.diag.request_alias("req=custom-request", self.key),
            self.diag.request_alias("req=custom-request-12345678", self.key),
        )

    def test_payload_and_block_values_never_export(self):
        collector, _ = self.parse(
            'Using prefill prefiller.url="http://user:password@10.2.3.4:8000" '
            'req_data={"prompt":"SECRET_PROMPT","token_ids":[99001,99002]}\n'
            "MooncakeLayerwiseToDram D advertised req req-secret: "
            "indexer=[1234567, 1234568] main_host=[555555] host=10.2.3.4 prompt_len=128\n"
        )
        serialized = json.dumps(collector.events)
        for value in ("SECRET_PROMPT", "password", "10.2.3.4", "99001", "1234567", "555555", "req-secret"):
            self.assertNotIn(value, serialized)
        self.assertEqual(collector.events[-1]["details"]["indexer_count"], 2)

    def test_w8a8_does_not_enable_cache_quantization(self):
        summary = self.diag.model_summary(
            {"model_type": "glm_moe_dsa", "indexer_types": ["full", "shared"]}, {"model.layers.0.weight": "W8A8"}
        )
        self.assertFalse(any(summary["derived_metadata_flags"].values()))
        self.assertEqual(summary["indexer_types_counts"], {"full": 1, "shared": 1})
        self.assertEqual(summary["weight_quant_counts"], {"W8A8": 1})

    def test_indexer_quant_is_distinct_from_fa_and_kv_quant(self):
        summary = self.diag.model_summary({}, {"indexer_quant_type": "C8"})
        self.assertEqual(
            summary["derived_metadata_flags"],
            {"enable_fa_quant": False, "enable_c8_quant": False, "enable_indexer_quant": True},
        )

    def test_command_parser_does_not_execute_or_export_unknowns(self):
        command = "vllm serve /private/SECRET_MODEL --api-key API_CANARY --tensor-parallel-size 8 "
        command += (
            '--additional-config \'{"multistream_overlap_shared_expert":false,"enable_fused_mc2":1,"secret":"CANARY"}\''
        )
        summary = self.diag.command_summary(command)
        self.assertEqual(summary["values"]["tensor_parallel_size"], 8)
        self.assertFalse(summary["values"]["additional_config"]["multistream_overlap_shared_expert"])
        self.assertNotIn("CANARY", json.dumps(summary))
        self.assertNotIn("SECRET_MODEL", json.dumps(summary))
        self.assertNotIn("enable_sparse_li_c8", summary["values"]["additional_config"])

    def test_malformed_additional_config_is_not_treated_as_false(self):
        summary = self.diag.command_summary("vllm serve model --additional-config '$LOCAL_JSON'")
        self.assertNotIn("additional_config", summary["values"])

    def test_runtime_cache_allowlist(self):
        summary = self.diag.runtime_summary(
            {
                "flags": {"enable_fa_quant": False, "secret": "CANARY"},
                "layers": [
                    {
                        "layer_index": 1,
                        "kind": "indexer",
                        "tuple_len": 1,
                        "shapes": [[2, 128, 1, 128]],
                        "dtypes": ["bfloat16"],
                        "pointer": "0xdeadbeef",
                        "contents": "CANARY",
                    }
                ],
            }
        )
        self.assertEqual(summary["layers"][0]["tuple_len"], 1)
        self.assertNotIn("CANARY", json.dumps(summary))
        self.assertNotIn("deadbeef", json.dumps(summary))

    def test_long_line_is_dropped_and_next_error_still_parses(self):
        collector, coverage = self.parse("SECRET" * self.diag.MAX_LINE_BYTES + "\n" + self.trace())
        self.assertEqual(coverage["oversized_lines_omitted"], 1)
        self.assertEqual(len(collector.errors), 1)
        self.assertEqual(next(iter(collector.errors.values()))["line"], 2)

    def test_byte_limit_reports_missing_evidence(self):
        path = self.root / "limited.log"
        path.write_text("uninteresting\n" * 100 + self.trace())
        collector = self.diag.LogCollector(self.key, "P")
        coverage = collector.read_log(path, 0, 20)
        self.assertTrue(coverage["byte_limit_reached"])
        self.assertEqual(len(collector.errors), 0)

    def test_gzip_log_is_processed_locally(self):
        path = self.root / "compressed.log.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(self.trace())
        collector = self.diag.LogCollector(self.key, "P")
        collector.read_log(path, 0, self.diag.MAX_LOG_BYTES)
        self.assertEqual(len(collector.errors), 1)

    def test_output_cannot_overwrite_or_enter_git(self):
        repo = self.root / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        with self.assertRaises(self.diag.CollectionError):
            self.diag.private_directory(repo / "export")
        with self.assertRaises(self.diag.CollectionError):
            self.diag.private_directory(repo)

    def test_output_symlink_is_rejected(self):
        linked = self.root / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(self.diag.CollectionError):
            self.diag.private_directory(linked / "export")

    def test_cli_failure_never_echoes_input_or_traceback(self):
        manifest = self.root / "SECRET_PATH.json"
        manifest.write_text('{"secret": "API_CANARY", malformed')
        completed = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "collect",
                "--manifest",
                str(manifest),
                "--output",
                str(self.root / "export"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        for value in ("SECRET_PATH", "API_CANARY", "Traceback", str(self.root)):
            self.assertNotIn(value, completed.stdout + completed.stderr)
        self.assertFalse((self.root / "export").exists())

    def test_proxy_checks_actual_script_not_copied_header(self):
        path = self.root / "load_balance_proxy_layerwise_server_example.py"
        path.write_text(
            '# python load_balance_proxy_server_example.py\n"/v1/metaserver"\n'
            '"do_remote_prefill": True\nasync def dispatch_prefill_batch(): pass\n'
        )
        summary = self.diag.proxy_summary(["python", path.name], self.root)
        self.assertTrue(summary["layerwise_markers_present"])
        self.assertEqual(summary["entry"], path.name)
        self.assertNotIn(str(self.root), json.dumps(summary))

    def test_versions_with_local_suffix_do_not_leak(self):
        self.assertEqual(self.diag.safe_version("0.3.13+PRIVATE_CANARY"), "0.3.13")
        self.assertEqual(self.diag.safe_version("2.10.0.post4"), "2.10.0.post4")
        self.assertEqual(self.diag.safe_version("0.25rc1+PRIVATE_CANARY"), "0.25rc1")
        self.assertEqual(self.diag.safe_version("0.25.0rc1+PRIVATE_CANARY"), "0.25.0rc1")

    def test_cli_end_to_end_exports_exactly_three_private_files(self):
        repo = self.root / "checkout"
        repo.mkdir()
        for args in (
            ["init", "-q"],
            [
                "-c",
                "user.name=Diagnostics Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--allow-empty",
                "-qm",
                "test fixture",
            ],
        ):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        work = self.root / "private"
        self.diag.initialize(work)
        log = self.root / "raw.log"
        log.write_text(self.trace(message="tuple index out of range SECRET_CONTENT"))
        manifest_path = work / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(ascend_repo=str(repo), logs=[str(log)])
        manifest_path.write_text(json.dumps(manifest))
        output = self.root / "export"
        completed = subprocess.run(
            [sys.executable, str(self.script), "collect", "--manifest", str(manifest_path), "--output", str(output)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        facts = json.loads((output / "facts.json").read_text())
        self.assertEqual(facts["declared_test_baseline"]["label"], "0.25rc1")
        self.assertEqual(facts["declared_test_baseline"]["source"], "user_not_runtime_verification")
        self.assertEqual(set(path.name for path in output.iterdir()), set(self.diag.EXPORT_FILES))
        for path in output.iterdir():
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertNotIn("SECRET_CONTENT", path.read_text())
            self.assertNotIn(str(self.root), path.read_text())
            self.assertNotIn((work / "correlation.key").read_text().strip(), path.read_text())
        self.assertEqual(log.read_text(), self.trace(message="tuple index out of range SECRET_CONTENT"))


if __name__ == "__main__":
    unittest.main()
