#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline, allowlist-only diagnostics. Never copy raw log text to an export."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import hmac
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shlex
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_SHA = "baf3cbcf22851bdb97102ce477329bd9f621240e"
DECLARED_TEST_BASELINE = "0.25rc1"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_LOG_BYTES = 512 * 1024 * 1024
MAX_FILES = 16
MAX_ERRORS = 12
MAX_FRAMES = 48
MAX_CHAIN = 6
MAX_EVENTS = 96
MAX_CACHE_SAMPLES = 24
EXPORT_FILES = ("facts.json", "evidence.txt", "analysis.md")
PACKAGES = (
    "vllm",
    "vllm-ascend",
    "torch",
    "torch-npu",
    "transformers",
    "mooncake-transfer-engine-npu",
    "mooncake-transfer-engine",
    "memfabric-hybrid",
)
SOURCE_FILES = (
    "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py",
    "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_to_dram_connector.py",
    "vllm_ascend/worker/model_runner_v1.py",
    "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py",
)
FRAME_FILES = frozenset(Path(name).name for name in SOURCE_FILES) | frozenset(
    {
        "ascend_multi_connector.py",
        "sfa_v1.py",
        "sfa_cp.py",
        "sfa_kv_offload.py",
        "host_pool.py",
        "host_backend.py",
        "kv_offload_decode_manager.py",
        "pool_worker.py",
        "fused_moe.py",
        "moe_mlp.py",
        "token_dispatcher.py",
        "prepare_finalize.py",
        "ascend_config.py",
        "ascend_forward_context.py",
        "modelslim_config.py",
        "deepseek_v2.py",
        "patch_deepseek_v2.py",
        "deepseek_mtp.py",
        "gpu_model_runner.py",
        "kv_cache_utils.py",
        "kv_cache_interface.py",
        "worker.py",
        "multiproc_executor.py",
        "core.py",
        "shared_segment.py",
        "indexer.py",
        "mla_v1.py",
        "block_table.py",
        "load_balance_proxy_server_example.py",
    }
)
FRAME_FUNCTIONS = frozenset(
    {
        "<module>",
        "<listcomp>",
        "<dictcomp>",
        "<genexpr>",
        "__init__",
        "forward",
        "forward_impl",
        "run",
        "main",
        "initialize",
        "init_device",
        "load_model",
        "process_weights_after_loading",
        "wrapped_process_weights",
        "profile_run",
        "_dummy_run",
        "_validate_shared_expert_consistency",
        "_shared_experts_part1",
        "_shared_experts_part2",
        "initialize_kv_cache",
        "initialize_kv_cache_tensors",
        "initialize_attn_backend",
        "register_kv_caches",
        "create_kv_buffer",
        "_compose_sfa_kv_cache",
        "_build_attention_metadata",
        "_allocate_kv_cache_tensors",
        "_reshape_kv_cache_tensors",
        "may_reinitialize_input_batch",
        "_allocate_fused_overlap_host_main",
        "_validate_layerwise_reuse_layouts",
        "get_kv_cache_spec",
        "save_kv_layer",
        "start_load_kv",
        "_get_kernel_block_ids",
        "_align_remote_block_ids",
        "_get_kv_split_metadata",
        "get_transfer_meta_asymmetric",
        "build_asymmetric_transfer_lists",
        "_transfer_one_leg",
        "_transfer_kv_cache",
        "_transfer_kv_cache_split_types",
        "_handle_request",
        "wait_for_layer_reuse",
        "update_state_after_alloc",
        "update_decoder_info",
        "_preprocess",
        "get_attn_backends_for_group",
        "create_attn_groups",
        "get_transfer_meta",
        "get_finished",
        "_handle_completions",
        "metaserver",
        "dispatch_prefill_batch",
        "create_shared_segment",
        "allocate_mooncake_host_region",
        "_select_shared_segment_mode",
        "quant_apply_mlp",
    }
)
ERROR_TYPES = frozenset(
    {
        "IndexError",
        "KeyError",
        "AttributeError",
        "TypeError",
        "ValueError",
        "RuntimeError",
        "AssertionError",
        "ImportError",
        "ModuleNotFoundError",
        "TimeoutError",
        "ConnectionError",
        "OSError",
        "MemoryError",
        "HTTPException",
        "Exception",
        "EngineDeadError",
    }
)
REASONS = (
    ("tuple index out of range", "tuple_index_out_of_range"),
    ("list index out of range", "list_index_out_of_range"),
    ("out of bounds", "index_out_of_bounds"),
    ("Layerwise slot-release layout mismatch", "layer_count_mismatch"),
    ("Main block_size mismatch", "main_block_size_mismatch"),
    ("block_len mismatch", "block_byte_length_mismatch"),
    ("shared_segment", "shared_segment_error"),
    ("shared experts split computation", "shared_expert_self_check"),
    ("out of memory", "out_of_memory"),
    ("Unknown metaserver request_id", "unknown_metaserver_request"),
)
EVENT_PATTERNS = (
    ("Starting to load model", "model_load_start"),
    ("Loading model weights took", "weights_loaded"),
    ("Model loading took", "weights_loaded"),
    ("Available KV cache memory", "memory_profile_complete"),
    ("Initializing Mooncake work", "mooncake_worker_init"),
    ("MooncakeLayerwiseToDram P: asymmetric Push", "p_connector_registered"),
    ("MooncakeLayerwiseToDram D register:", "d_cache_registered"),
    ("MooncakeLayerwiseToDram D recv ready", "d_recv_ready"),
    ("Graph capturing finished", "graph_capture_complete"),
    ("Application startup complete", "api_ready"),
    ("MooncakeLayerwiseToDram D advertised req", "d_blocks_advertised"),
    ("Using prefill prefiller.url=", "proxy_prefill_dispatch"),
    ("MooncakeToDram P start_load_kv", "p_blocks_mapped"),
    ("MooncakeToDram P transfer meta", "p_transfer_planned"),
    ("Sending transmitting signal b'done_sending_msg'", "p_done_signal_sent"),
    ("Sending transmitting signal b'failed_sending_msg'", "p_failed_signal_sent"),
    ("MooncakeToDram D get_finished done=", "d_completion_observed"),
    ("Mooncake transfer failed", "transfer_failure"),
    ("Failed to transfer KV cache", "transfer_failure"),
    ("/v1/metaserver HTTP/", "metaserver_http_response"),
)
BOOL_KEYS = frozenset(
    {
        "multistream_overlap_shared_expert",
        "enable_sparse_sfa_c8",
        "enable_sparse_li_c8",
        "enable_prefill_mc2",
        "enable_mc2_hierarchy_comm",
        "enable_dsa_cp",
        "enable_flashcomm1",
        "enable_fa_quant",
        "enable_c8_quant",
        "enable_indexer_quant",
        "use_layerwise",
        "use_fused_overlap",
        "enabled",
        "keep_device_kv_cache",
        "enforce_eager",
    }
)
INT_KEYS = frozenset(
    {
        "tensor_parallel_size",
        "data_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "block_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_fused_mc2",
        "num_speculative_tokens",
        "cp_kv_cache_interleave_size",
        "layerwise_num_shared_buffers",
        "layerwise_prefetch_layers",
        "total_layers",
        "event_count",
        "num_blocks",
    }
)
ENUM_KEYS = {
    "quantization": frozenset({"ascend", "modelslim", "none"}),
    "dtype": frozenset({"auto", "bfloat16", "float16", "float32"}),
    "kv_cache_dtype": frozenset({"auto", "bfloat16", "float16", "int8", "fp8"}),
    "kv_role": frozenset({"kv_producer", "kv_consumer", "kv_both"}),
    "sfa_kv_offload_backend": frozenset({"mooncake", "memfabric"}),
    "backend": frozenset({"mooncake", "memcache", "memfabric"}),
    "kv_connector": frozenset(
        {
            "MultiConnector",
            "AscendMultiConnector",
            "AscendStoreConnector",
            "MooncakeConnectorStoreV1",
            "MooncakeConnector",
            "MooncakeLayerwiseConnector",
            "MooncakeLayerwiseToDramConnector",
            "MooncakeLayerwiseD2RHConnector",
        }
    ),
    "cudagraph_mode": frozenset({"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}),
}
NESTED_KEYS = frozenset(
    {
        "additional_config",
        "kv_transfer_config",
        "kv_connector_extra_config",
        "kv_offload_decode_config",
        "speculative_config",
        "compilation_config",
        "ascend_compilation_config",
    }
)
CLI_ALIASES = {"tp": "tensor_parallel_size", "dp": "data_parallel_size", "pp": "pipeline_parallel_size"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([\w<>]+)')
EXCEPTION = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z_0-9.]*(?:Error|Exception)):\s*(.*)")
VERSION = re.compile(r"\d{1,4}(?:\.\d{1,8}){1,3}(?:(?:\.?post|\.?dev|rc|a|b)\d{1,4})?")


class CollectionError(Exception):
    """Only fixed, non-sensitive codes may be passed to this exception."""


def safe_int(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**48 else None


def safe_version(value: str) -> str | None:
    match = VERSION.match(value)
    return match.group(0) if match else None


def read_local(path: Path, limit: int = MAX_JSON_BYTES) -> bytes:
    if not stat.S_ISREG(path.stat().st_mode):
        raise CollectionError("input_not_regular_file")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise CollectionError("input_file_too_large")
    return data


def load_json(path: Path) -> dict:
    value = json.loads(read_local(path))
    if not isinstance(value, dict):
        raise CollectionError("json_object_required")
    return value


def filtered_config(value: Any, depth: int = 0) -> dict:
    if not isinstance(value, dict) or depth > 6:
        return {}
    result = {}
    for key, item in value.items():
        if key in BOOL_KEYS and type(item) is bool or key in INT_KEYS and safe_int(item) is not None:
            result[key] = item
        elif key in ENUM_KEYS:
            result[key] = item if isinstance(item, str) and item in ENUM_KEYS[key] else "unrecognized"
        elif key in NESTED_KEYS:
            result[key] = filtered_config(item, depth + 1)
        elif key == "connectors" and isinstance(item, list):
            result[key] = [filtered_config(entry, depth + 1) for entry in item[:8]]
    return result


def runtime_summary(value: Any) -> dict:
    """Validate optional structured runtime facts, including actual cache tuple shapes."""
    if not isinstance(value, dict):
        return {}
    result = filtered_config(value)
    result["flags"] = filtered_config(value.get("flags"))
    result["layers"] = []
    layers = value.get("layers", [])
    if not isinstance(layers, list):
        return result
    for layer in layers[:MAX_CACHE_SAMPLES]:
        if not isinstance(layer, dict):
            continue
        clean = {
            key: layer[key]
            for key in ("layer_index", "group_id", "tuple_len", "block_size")
            if safe_int(layer.get(key)) is not None
        }
        clean["kind"] = layer.get("kind") if layer.get("kind") in {"main", "indexer", "shared_indexer"} else "other"
        for key in ("has_indexer", "skip_topk"):
            if type(layer.get(key)) is bool:
                clean[key] = layer[key]
        shapes = layer.get("shapes", [])
        if isinstance(shapes, list):
            clean["shapes"] = [
                shape
                for shape in shapes[:8]
                if isinstance(shape, list) and 0 < len(shape) <= 8 and all(safe_int(dim) is not None for dim in shape)
            ]
        dtypes = layer.get("dtypes", [])
        if isinstance(dtypes, list):
            clean["dtypes"] = [
                dtype if dtype in ("bfloat16", "float16", "float32", "int8", "int32", "int64") else "other"
                for dtype in dtypes[:8]
            ]
        result["layers"].append(clean)
    return result


def command_summary(text: str) -> dict:
    """Parse text, never execute shell or expand variables; unknown values are omitted."""
    tokens = shlex.split(text.replace("\\\n", " "), comments=True)
    values: dict[str, Any] = {}
    occurrences: Counter = Counter()
    for index, token in enumerate(tokens):
        if not token.startswith("--") and token not in {"-tp", "-dp", "-pp"}:
            continue
        flag, equal, inline = token.lstrip("-").partition("=")
        key = CLI_ALIASES.get(flag, flag.replace("-", "_"))
        if key not in BOOL_KEYS | INT_KEYS | NESTED_KEYS | ENUM_KEYS.keys():
            continue
        occurrences[key] += 1
        raw = inline if equal else tokens[index + 1] if index + 1 < len(tokens) else ""
        if key == "enforce_eager" and not equal:
            values[key] = True
        elif key in INT_KEYS and raw.isdecimal():
            values[key] = int(raw)
        elif key in BOOL_KEYS and raw.lower() in {"true", "false"}:
            values[key] = raw.lower() == "true"
        elif key in ENUM_KEYS:
            values[key] = raw
        elif key in NESTED_KEYS:
            try:
                values[key] = json.loads(raw)
            except (ValueError, TypeError):
                continue
    return {
        "values": filtered_config(values),
        "repeated_flags": sorted(key for key, count in occurrences.items() if count > 1),
        "source": "argv_or_script_not_effective_runtime",
    }


def model_summary(config: dict, quant: dict) -> dict:
    markers = {}
    for key in ("fa_quant_type", "indexer_quant_type", "kv_cache_type"):
        raw = quant.get(key)
        markers[key] = raw if raw in (None, "", "C8", "INT8", "FP8", "W8A8") else "other"
    types = config.get("indexer_types")
    indexer_counts = None
    if isinstance(types, list):
        indexer_counts = dict(Counter(item if item in ("shared", "full") else "other" for item in types))
    return {
        "source": "checkpoint_metadata_not_effective_runtime",
        "model_type": config.get("model_type")
        if config.get("model_type") in {"glm_moe_dsa", "deepseek_v3", "deepseek_v32"}
        else "other_or_missing",
        "dimensions": {
            key: config[key]
            for key in (
                "num_hidden_layers",
                "num_nextn_predict_layers",
                "index_topk",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "index_head_dim",
            )
            if safe_int(config.get(key)) is not None
        },
        "indexer_types_counts": indexer_counts,
        "quant_markers": markers,
        "derived_metadata_flags": {
            "enable_fa_quant": quant.get("fa_quant_type", "") != "",
            "enable_indexer_quant": quant.get("indexer_quant_type", "") != "",
            "enable_c8_quant": quant.get("kv_cache_type", "") == "C8",
        }
        if quant
        else {},
        "weight_quant_counts": dict(
            Counter(
                item if item in ("W8A8", "W8A8_DYNAMIC", "W4A8", "FLOAT") else "other"
                for key, item in quant.items()
                if key.endswith(".weight")
            )
        ),
    }


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise CollectionError("git_probe_failed")
    return result.stdout.strip()


def repository_summary(repo: Path) -> dict:
    head = git_output(repo, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise CollectionError("invalid_git_head")
    files = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if path.is_file():
            files[relative] = hashlib.sha256(read_local(path)).hexdigest()
    return {
        "head": head,
        "matches_reviewed_experiment": head == TARGET_SHA,
        "dirty": bool(git_output(repo, "status", "--porcelain")),
        "source_sha256": files,
    }


def environment_summary(repo: Path, pid: int | None) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = safe_version(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    spec = importlib.machinery.PathFinder.find_spec("vllm_ascend")
    installed_matches = None
    if spec is not None and spec.origin:
        installed_matches = Path(spec.origin).resolve().is_relative_to(repo.resolve())
    executable_matches = None
    if pid is not None and Path(f"/proc/{pid}/exe").exists():
        executable_matches = Path(f"/proc/{pid}/exe").resolve() == Path(sys.executable).resolve()
    return {
        "python": platform.python_version(),
        "machine": platform.machine() if platform.machine() in {"aarch64", "arm64", "x86_64"} else "other",
        "glibc": safe_version(platform.libc_ver()[1]),
        "packages": versions,
        "collector_import_matches_ascend_checkout": installed_matches,
        "service_executable_matches_collector": executable_matches,
        "scope": "collector_interpreter_only_not_proof_of_running_worker_imports",
    }


def proxy_summary(tokens: list[str], cwd: Path) -> dict:
    """Inspect the actual argv/script token, not a copied command in a comment."""
    for token in tokens:
        name = Path(token).name
        if name not in {"load_balance_proxy_layerwise_server_example.py", "load_balance_proxy_server_example.py"}:
            continue
        path = Path(token) if Path(token).is_absolute() else cwd / token
        result = {"entry": name, "source_sha256": None, "layerwise_markers_present": None}
        if path.is_file():
            data = read_local(path)
            result["source_sha256"] = hashlib.sha256(data).hexdigest()
            result["layerwise_markers_present"] = all(
                marker in data
                for marker in (b"/v1/metaserver", b'"do_remote_prefill": True', b"dispatch_prefill_batch")
            )
        return result
    return {"entry": "not_observed", "source_sha256": None, "layerwise_markers_present": None}


def rank_of(line: str) -> dict:
    return {
        name.lower(): int(number)
        for name, number in re.findall(
            r"(?:^|[^A-Za-z0-9])(?:Worker_)?(TP|DP|PP|PCP|DCP)(\d{1,5})(?=[^A-Za-z0-9]|$)", line
        )
    }


def clock_of(line: str) -> str | None:
    match = re.search(r"\b(?:\d{4}-)?\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?", line)
    return match.group(0) if match else None


def request_alias(line: str, key: bytes) -> str | None:
    match = re.search(r"\b(?:req_id|request_id|req|request)\s*[=:]?\s*['\"]?([A-Za-z0-9_.:-]{1,200})", line)
    if not match:
        return None
    identifier = match.group(1)
    # Only strip the known vLLM EngineCore suffix from known proxy-generated IDs.
    generated = re.fullmatch(r"((?:chatcmpl-[0-9a-f-]{36}|cmpl-[0-9a-f-]{36}-\d+))(?:-[A-Za-z0-9]{8})?", identifier)
    if generated:
        identifier = generated.group(1)
    return "req-" + hmac.new(key, identifier.encode(), hashlib.sha256).hexdigest()[:16]


def safe_shape(text: str) -> list[int] | None:
    try:
        shape = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError):
        return None
    if isinstance(shape, (list, tuple)) and 0 < len(shape) <= 8 and all(safe_int(dim) is not None for dim in shape):
        return list(shape)
    return None


def event_details(line: str) -> dict:
    result = {}
    for key in (
        "tp_ratio",
        "num_blocks",
        "num_computed",
        "num_external",
        "remote_tp",
        "prompt_len",
        "local_computed_tokens",
        "local_transed_tokens",
        "layer",
        "ret",
        "xfer",
    ):
        match = re.search(rf"\b{key}[=:]\s*(-?\d{{1,15}})\b", line)
        if match:
            result[key] = int(match.group(1))
    layer = re.search(r"\blayers\.(\d{1,5})\.", line)
    if layer:
        result["layer_index"] = int(layer.group(1))
    for key in ("indexer", "main_host", "indexer_mapped", "local_mapped", "main_mapped"):
        match = re.search(rf"\b{key}=\[([\d,\s]*)\]", line)
        if match:
            result[key + "_count"] = len(re.findall(r"\d+", match.group(1)))
    for key in ("main_owner", "skip_indexer", "skip_main", "indexer_sender", "main_sender"):
        match = re.search(rf"\b{key}=(True|False)\b", line)
        if match:
            result[key] = match.group(1) == "True"
    response = re.search(r'/v1/metaserver HTTP/[^"\s]+"\s+(\d{3})', line)
    if response:
        result["http_status"] = int(response.group(1))
    return result


class LogCollector:
    def __init__(self, key: bytes, role: str):
        self.key, self.role = key, role
        self.errors: dict[str, dict] = {}
        self.events: list[dict] = []
        self.caches: list[dict] = []
        self.runtime: list[dict] = []
        self.counts: Counter = Counter()
        self.pending: dict[tuple, dict] = {}

    def finish_error(self, stream: tuple) -> None:
        error = self.pending.pop(stream, None)
        if error is None:
            return
        if not error["chain"][-1].get("type"):
            error["incomplete"] = True
        fingerprint = hashlib.sha256(json.dumps(error["chain"], sort_keys=True).encode()).hexdigest()[:16]
        if fingerprint in self.errors:
            existing = self.errors[fingerprint]
            existing["occurrences"] += 1
            existing["incomplete"] |= error["incomplete"]
            if error["rank"] not in existing["ranks"] and len(existing["ranks"]) < 64:
                existing["ranks"].append(error["rank"])
        elif len(self.errors) < MAX_ERRORS:
            error.pop("linked", None)
            error["ranks"] = [error.pop("rank")]
            error["occurrences"] = 1
            error["fingerprint"] = fingerprint
            self.errors[fingerprint] = error
        else:
            self.counts["errors_over_budget"] += 1

    def trace_line(self, line: str, location: dict, stream: tuple) -> None:
        error = self.pending.get(stream)
        if "Traceback (most recent call last):" in line:
            if error is not None and error["linked"]:
                error["linked"] = False
                if len(error["chain"]) < MAX_CHAIN:
                    error["chain"].append({"frames": []})
                else:
                    error["incomplete"] = True
                    error["chain_overflow"] = True
            else:
                self.finish_error(stream)
                self.pending[stream] = dict(location, chain=[{"frames": []}], linked=False, incomplete=False)
            return
        if error is None:
            return
        if "direct cause of the following exception" in line or "During handling of the above exception" in line:
            error["linked"] = True
            return
        if error.get("chain_overflow") or error["chain"][-1].get("type"):
            return
        frame = FRAME.search(line)
        if frame:
            filename, number, function = frame.groups()
            frames = error["chain"][-1]["frames"]
            if len(frames) < MAX_FRAMES:
                frames.append(
                    {
                        "file": Path(filename).name if Path(filename).name in FRAME_FILES else "other_file",
                        "line": int(number),
                        "function": function if function in FRAME_FUNCTIONS else "other_function",
                    }
                )
            else:
                error["incomplete"] = True
                # Retain the deepest failing frames as well as the entry frames.
                frames.pop(MAX_FRAMES // 2)
                frames.append(
                    {
                        "file": Path(filename).name if Path(filename).name in FRAME_FILES else "other_file",
                        "line": int(number),
                        "function": function if function in FRAME_FUNCTIONS else "other_function",
                    }
                )
            return
        exception = EXCEPTION.search(line)
        if exception:
            kind, message = exception.groups()
            kind = kind.rsplit(".", 1)[-1]
            error["chain"][-1]["type"] = kind if kind in ERROR_TYPES else "other_exception"
            error["chain"][-1]["reason"] = next((code for text, code in REASONS if text in message), "message_withheld")

    def accept(self, line: str, source: int, number: int) -> None:
        line = ANSI.sub("", line)
        rank = rank_of(line)
        location = {"source": source, "line": number, "rank": rank, "clock": clock_of(line)}
        stream = (source, tuple(sorted(rank.items())))
        if not rank and "Traceback (most recent call last):" not in line and stream not in self.pending:
            candidates = [item for item in self.pending if item[0] == source]
            if len(candidates) == 1:
                stream = candidates[0]
            elif len(candidates) > 1 and (FRAME.search(line) or EXCEPTION.search(line)):
                self.counts["ambiguous_unprefixed_trace_lines"] += 1
                for candidate in candidates:
                    self.pending[candidate]["incomplete"] = True
        self.trace_line(line, location, stream)
        for pattern, event in EVENT_PATTERNS:
            if pattern not in line:
                continue
            self.counts[event] += 1
            if len(self.events) < MAX_EVENTS:
                self.events.append(
                    dict(location, event=event, request=request_alias(line, self.key), details=event_details(line))
                )
            else:
                self.counts["events_over_budget"] += 1
                self.events.pop(MAX_EVENTS // 2)
                self.events.append(
                    dict(location, event=event, request=request_alias(line, self.key), details=event_details(line))
                )
            break
        if "layer:" in line and "num_blocks:" in line and "block_shape:" in line:
            match = re.search(r"block_shape:\s*(\([^)]*\)|\[[^]]*\])", line)
            if match and len(self.caches) < MAX_CACHE_SAMPLES:
                self.caches.append(
                    dict(
                        location,
                        kind="indexer" if "indexer" in line else "main_or_other",
                        block_shape=safe_shape(match.group(1)),
                        details=event_details(line),
                    )
                )
        # Optional output of an already-authorized runtime probe. Do not insert one automatically.
        if "ASCEND_DIAG_FACTS=" in line and len(self.runtime) < 8:
            try:
                data = json.loads(line.split("ASCEND_DIAG_FACTS=", 1)[1])
                self.runtime.append(dict(location, values=runtime_summary(data)))
            except (ValueError, TypeError):
                self.counts["invalid_runtime_record"] += 1

    def read_log(self, path: Path, source: int, byte_limit: int) -> dict:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise CollectionError("log_not_regular_file")
        opener = gzip.open if path.suffix == ".gz" else open
        consumed, number, oversized = 0, 0, 0
        truncated = False
        with opener(path, "rb") as handle:
            while consumed < byte_limit:
                data = handle.readline(min(MAX_LINE_BYTES + 1, byte_limit - consumed + 1))
                if not data:
                    break
                consumed += len(data)
                number += 1
                if consumed > byte_limit:
                    truncated = True
                    break
                if len(data) > MAX_LINE_BYTES:
                    oversized += 1
                    while data and not data.endswith(b"\n") and consumed <= byte_limit:
                        data = handle.readline(min(MAX_LINE_BYTES, byte_limit - consumed + 1))
                        consumed += len(data)
                    if consumed > byte_limit:
                        truncated = True
                    continue
                self.accept(data.decode("utf-8", errors="replace"), source, number)
            else:
                truncated = bool(handle.read(1))
        for stream in list(self.pending):
            if stream[0] == source:
                self.finish_error(stream)
        after = path.stat()
        return {
            "source": source,
            "lines_scanned": number,
            "oversized_lines_omitted": oversized,
            "byte_limit_reached": truncated,
            "changed_while_reading": (before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns),
        }


def private_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise CollectionError("output_must_not_exist")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise CollectionError("output_symlink_forbidden")
        if (parent / ".git").exists():
            raise CollectionError("output_inside_git_repository_forbidden")
    path.mkdir(mode=0o700, parents=True)


def write_private(path: Path, data: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        handle.write(data)


def initialize(path: Path) -> None:
    private_directory(path)
    write_private(path / "correlation.key", secrets.token_hex(32) + "\n")
    manifest = {
        "schema_version": 1,
        "role": "P",
        "instance": 0,
        "ascend_repo": "/REPLACE/experiment/vllm-ascend",
        "vllm_repo": None,
        "correlation_key_file": str((path / "correlation.key").resolve()),
        "logs": ["/REPLACE/current-attempt/p.log"],
        "pid": None,
        "command_file": None,
        "command_cwd": None,
        "model_config_file": None,
        "quant_config_file": None,
        "version_files": {},
        "max_bytes_per_log": 128 * 1024 * 1024,
    }
    write_private(path / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def collect_facts(manifest: dict) -> tuple[dict, LogCollector]:
    if manifest.get("schema_version") != 1 or manifest.get("role") not in {"P", "D", "PROXY"}:
        raise CollectionError("invalid_manifest_schema_or_role")
    instance = manifest.get("instance")
    if type(instance) is not int or not 0 <= instance <= 999:
        raise CollectionError("invalid_instance")
    logs = manifest.get("logs")
    if not isinstance(logs, list) or not 1 <= len(logs) <= MAX_FILES or len(set(logs)) != len(logs):
        raise CollectionError("invalid_log_list")
    limit = manifest.get("max_bytes_per_log", 128 * 1024 * 1024)
    if type(limit) is not int or not 1 <= limit <= MAX_LOG_BYTES:
        raise CollectionError("invalid_byte_limit")
    pid = manifest.get("pid")
    if pid is not None and (type(pid) is not int or not 1 <= pid <= 2**31):
        raise CollectionError("invalid_pid")
    key_text = read_local(Path(manifest["correlation_key_file"]), 128).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", key_text):
        raise CollectionError("invalid_correlation_key")
    key = bytes.fromhex(key_text)
    repo = Path(manifest["ascend_repo"])
    facts = {
        "schema_version": 1,
        "role": manifest["role"],
        "declared_test_baseline": {"label": DECLARED_TEST_BASELINE, "source": "user_not_runtime_verification"},
        "instance": instance,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "correlation_scope": hashlib.sha256(key).hexdigest()[:16],
        "clock_order_across_hosts": "unverified",
        "ascend_repository": repository_summary(repo),
        "environment": environment_summary(repo, pid),
        "vllm_repository": repository_summary(Path(manifest["vllm_repo"])) if manifest.get("vllm_repo") else None,
        "command": None,
        "proxy": None,
        "checkpoint": None,
        "system_versions": {},
    }
    tokens = None
    cwd = Path(manifest.get("command_cwd") or repo)
    if pid is not None:
        tokens = read_local(Path(f"/proc/{pid}/cmdline"), MAX_LINE_BYTES).decode().strip("\0").split("\0")
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        facts["command"] = command_summary(shlex.join(tokens))
        facts["command"]["source"] = "live_process_argv_not_effective_runtime"
    elif manifest.get("command_file"):
        command = read_local(Path(manifest["command_file"])).decode()
        facts["command"] = command_summary(command)
        tokens = shlex.split(command.replace("\\\n", " "), comments=True)
    if manifest["role"] == "PROXY" and tokens is not None:
        facts["proxy"] = proxy_summary(tokens, cwd)
    if manifest.get("model_config_file") or manifest.get("quant_config_file"):
        facts["checkpoint"] = model_summary(
            load_json(Path(manifest["model_config_file"])) if manifest.get("model_config_file") else {},
            load_json(Path(manifest["quant_config_file"])) if manifest.get("quant_config_file") else {},
        )
    for name in ("cann", "hixl", "driver"):
        version_file = manifest.get("version_files", {}).get(name)
        if version_file:
            raw = read_local(Path(version_file), MAX_LINE_BYTES).decode(errors="replace")
            version = re.search(r"(?:Version|version|version_dir)\s*=\s*([0-9][^\s]*)", raw)
            facts["system_versions"][name] = safe_version(version.group(1)) if version else None
    collector = LogCollector(key, manifest["role"])
    facts["coverage"] = [collector.read_log(Path(path), index, limit) for index, path in enumerate(logs)]
    facts["event_counts"] = dict(collector.counts)
    facts["runtime_observations"] = collector.runtime
    facts["cache_samples"] = collector.caches
    return facts, collector


def make_analysis(facts: dict, errors: list[dict], events: list[dict]) -> str:
    lines = [
        "# 诊断摘要",
        "",
        "仅依据本次允许字段；没有读取请求正文进行分析，也没有执行网络请求。",
        "",
        "## 事实",
        "",
        f"- 角色：{facts['role']}，实例：{facts['instance']}。",
        f"- 用户声明的测试基线：{DECLARED_TEST_BASELINE}；实测包版本和 commit 需分别核对，不能混入 0.23 的结果。",
        f"- 唯一异常样本：{len(errors)}；保留事件：{len(events)}。",
    ]
    if facts["proxy"]:
        lines.append(f"- proxy 入口识别：`{facts['proxy']['entry']}`；只能结合事件确认实际路由。")
    for index, error in enumerate(errors):
        lines.append(
            f"- E{index + 1:03d}：出现 {error['occurrences']} 次；异常链完整性缺失标志为 {error['incomplete']}。"
        )
    lines.extend(["", "## 待验证假设", ""])
    for index, error in enumerate(errors):
        segments = error["chain"]
        if any(
            segment.get("reason") == "tuple_index_out_of_range"
            and any(frame["function"] == "create_kv_buffer" for frame in segment["frames"])
            for segment in segments
        ):
            lines.append(
                f"- E{index + 1:03d} 与已审查的单 tensor 注册越界相符；"
                "仍需核对同一运行的量化标志、Indexer tuple 长度。W8A8 不能替代这些证据。"
            )
        if any(segment.get("reason") == "layer_count_mismatch" for segment in segments):
            lines.append(
                f"- E{index + 1:03d} 表明层数校验失败；比较真实 MTP 层、total_layers 和事件数组长度，"
                "不直接判定为 Mooncake 版本问题。"
            )
    lines.extend(
        [
            "- 日志中没有某事件，不代表该事件没有发生。此前的 mock 测试不是本次 NPU 验证。",
            "",
            "## 缺失证据与停止条件",
            "",
            "- 核对 coverage 的截断、超长行和读取期间变化；核对 event_counts 中 over_budget 字段。",
            "- 未采到的运行时字段保持未知；argv 和 checkpoint 推导不等于运行时有效配置。",
            "- 先确认收到请求前是否已报错，再归因 proxy。跨主机时钟未校准时不宣称全局最早异常。",
            "- 本采集不证明 P/D ready、传输完成或生成正确；需要同一次运行、同一匿名请求的正向证据。",
            "",
            "## 下一步",
            "",
            "按 references/analysis-guide.md 核对事实，并按模板追加引用证据编号的分析。不得粘贴原始日志。",
            "未授权时，不重启服务、不修改源码、不切换 MTP/MC2/量化、不上传文件。",
            "",
        ]
    )
    return "\n".join(lines)


def export(manifest: dict, output: Path) -> None:
    # Parse everything before creating the export; failures must not leave partial raw artifacts.
    facts, collector = collect_facts(manifest)
    errors = list(collector.errors.values())
    events = collector.events
    evidence = []
    for prefix, records in (("E", errors), ("T", events)):
        for index, record in enumerate(records):
            evidence.append(
                json.dumps({"id": f"{prefix}{index + 1:03d}", **record}, ensure_ascii=False, sort_keys=True)
            )
    private_directory(output)
    write_private(output / "facts.json", json.dumps(facts, ensure_ascii=False, indent=2) + "\n")
    write_private(output / "evidence.txt", "\n".join(evidence) + "\n")
    write_private(output / "analysis.md", make_analysis(facts, errors, events))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser("init", help="create a local-only manifest and correlation key")
    init_parser.add_argument("--work-dir", type=Path, required=True)
    collect_parser = commands.add_parser("collect", help="write only facts.json, evidence.txt and analysis.md")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.work_dir)
            print("OK: local manifest and private correlation key created. Do not export them.")
        else:
            export(load_json(args.manifest), args.output)
            print("OK: facts.json, evidence.txt, analysis.md created. No upload performed.")
        return 0
    except CollectionError as error:
        print(f"Collection stopped: {error}", file=sys.stderr)
    except Exception:
        # OSError/JSON/shlex messages can contain paths, credentials or raw input. Never print them.
        print(
            "Collection stopped: input_or_environment_error. "
            "Check the local manifest and permissions; no raw detail exported.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
