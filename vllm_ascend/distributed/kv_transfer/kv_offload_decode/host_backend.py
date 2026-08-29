# SPDX-License-Identifier: Apache-2.0
"""Decode Host-memory backend selection for SFA KV offload."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

SFA_KV_OFFLOAD_BACKEND_MEMFABRIC = "memfabric"
SFA_KV_OFFLOAD_BACKEND_MOONCAKE = "mooncake"
SFA_KV_OFFLOAD_BACKENDS = frozenset(
    {
        SFA_KV_OFFLOAD_BACKEND_MEMFABRIC,
        SFA_KV_OFFLOAD_BACKEND_MOONCAKE,
    }
)


def normalize_sfa_kv_offload_backend(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value not in SFA_KV_OFFLOAD_BACKENDS:
        raise ValueError(
            f"sfa_kv_offload_backend={raw!r} is invalid; expected "
            f"{SFA_KV_OFFLOAD_BACKEND_MEMFABRIC!r} or "
            f"{SFA_KV_OFFLOAD_BACKEND_MOONCAKE!r}"
        )
    return value


def resolve_sfa_kv_offload_backend(
    extra_config: Mapping[str, Any] | None,
    *,
    use_fused_overlap_offload: bool,
) -> str:
    """Resolve the Decode Host-memory backend from connector configuration."""
    extra = extra_config or {}
    requested = normalize_sfa_kv_offload_backend(
        extra.get("sfa_kv_offload_backend")
    )
    if not use_fused_overlap_offload:
        if requested == SFA_KV_OFFLOAD_BACKEND_MOONCAKE:
            logger.warning(
                "sfa_kv_offload_backend=mooncake requires fused overlap; "
                "using memfabric"
            )
        return SFA_KV_OFFLOAD_BACKEND_MEMFABRIC
    return requested or SFA_KV_OFFLOAD_BACKEND_MEMFABRIC


def ensure_mooncake_host_is_pd_decode_only(
    backend: str,
    *,
    keep_device_kv_cache: bool,
) -> None:
    """Mooncake Host is PD-disaggregated decode only.

    ``keep_device_kv_cache`` is the memfabric colocate-debug escape hatch.
    PD decode keeps that flag false, so this returns immediately.
    """
    if not keep_device_kv_cache:
        return
    if backend == SFA_KV_OFFLOAD_BACKEND_MOONCAKE:
        raise RuntimeError(
            "Mooncake Host backend is PD-disaggregated decode only; "
            "keep_device_kv_cache (PD-colocate debug) requires memfabric"
        )


def kv_transfer_extra_config(vllm_config: Any) -> Mapping[str, Any] | None:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None
    extra_config = getattr(
        kv_transfer_config,
        "kv_connector_extra_config",
        None,
    )
    return extra_config if isinstance(extra_config, Mapping) else None
