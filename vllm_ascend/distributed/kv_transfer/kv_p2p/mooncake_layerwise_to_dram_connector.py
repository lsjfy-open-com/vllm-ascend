# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Mooncake layerwise Push to Decode Indexer HBM and shared Main DRAM."""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.host_backend import (
    SFA_KV_OFFLOAD_BACKEND_MOONCAKE,
    kv_transfer_extra_config,
    resolve_sfa_kv_offload_backend,
)
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.host_pool import (
    DSAHostKVPool,
)
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.kv_offload_decode_manager import (
    get_kv_offload_decode_manager,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    KVCacheRecvingLayerThread,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorWorker,
    LayerMetadata as MooncakeLayerMetadata,
    TransferMeta,
    group_concurrent_contiguous,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.protocol import (
    get_external_request_id,
    infer_sfa_component_group_ids,
)
from vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler import (
    SFAPDCpuOffloadScheduler,
    SFAPDProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import get_transfer_timeout_value

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")

# Advertised ``remote_block_ids`` layout for Mooncake Push. This is the PD
# wire order (Indexer, Main), not vLLM's local KV-group ids.
_REMOTE_INDEXER_IDX = 0
_REMOTE_MAIN_IDX = 1


def _register_te_memory(
    engine: Any,
    ptr: int,
    nbytes: int,
    *,
    location: str | None,
    name: str,
) -> None:
    """Register one non-pool buffer with Mooncake TE."""
    fn = engine.register_memory
    used = "2-arg"
    if location is None:
        ret = fn(ptr, nbytes)
    else:
        try:
            ret = fn(ptr, nbytes, location=location)
            used = f"kw:{location}"
        except TypeError:
            try:
                ret = fn(ptr, nbytes, location)
                used = f"pos:{location}"
            except TypeError as e:
                raise RuntimeError(
                    "Mooncake register_memory rejects location="
                    f"{location!r} for {name}; need TE with location support "
                    "(Mooncake PR #2191)."
                ) from e
    if ret != 0:
        raise RuntimeError(
            f"Mooncake register_memory failed for {name}: ret={ret} "
            f"ptr=0x{ptr:x} nbytes={nbytes} location={used}"
        )
    logger.debug(
        "MooncakeToDram TE register %s ptr=0x%x nbytes=%s location=%s",
        name,
        ptr,
        nbytes,
        used,
    )


def _layer_idx(layer_name: str) -> int:
    match = _LAYER_IDX_RE.search(layer_name)
    assert match is not None, f"no transformer layer index in layer name {layer_name!r}"
    return int(match.group(1))


def _indexer_layer_name(main_layer_name: str) -> str:
    """Decode Indexer GET_META key for a Prefill co-located MLA layer."""
    return f"{main_layer_name}.indexer.k_cache"


def _find_remote_meta_by_layer_idx(
    remote_layers: dict[str, Any],
    layer_idx: int,
    *,
    want_indexer: bool,
) -> tuple[str | None, Any | None]:
    """Resolve Decode GET_META entry by transformer layer index (SFAPD-style).

    Prefill layer names need not equal Decode keys: Indexer is published under
    ``...indexer.k_cache``, Main HOST under the MLA attn name.
    """
    return _find_layer_meta_by_idx(remote_layers, layer_idx, want_indexer=want_indexer)


def _find_layer_meta_by_idx(
    layer_metadata: dict[str, Any],
    layer_idx: int,
    *,
    want_indexer: bool,
) -> tuple[str | None, Any | None]:
    """Pick Indexer vs Main meta for one transformer index from a name→meta map."""
    for name, meta in layer_metadata.items():
        try:
            if _layer_idx(name) != layer_idx:
                continue
        except AssertionError:
            continue
        is_indexer = "indexer" in name.lower()
        if want_indexer and is_indexer:
            return name, meta
        if not want_indexer and not is_indexer:
            return name, meta
    return None, None


def _split_local_addrs_for_layer(
    layer_metadata: dict[str, Any],
    layer_name: str,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Prefill src ptrs: 0723 split groups or 0713 colocated (k,v,indexer).

    Returns ``(main_addrs, main_lens, indexer_addrs, indexer_lens)``.
    """
    layer_idx = _layer_idx(layer_name)
    _main_name, main_meta = _find_layer_meta_by_idx(
        layer_metadata, layer_idx, want_indexer=False
    )
    _idx_name, idx_meta = _find_layer_meta_by_idx(
        layer_metadata, layer_idx, want_indexer=True
    )
    if main_meta is None:
        main_meta = layer_metadata.get(layer_name)
    main_addrs = list(getattr(main_meta, "kv_caches_base_addr", None) or [])
    main_lens = list(getattr(main_meta, "block_len", None) or [])
    if idx_meta is not None and idx_meta is not main_meta:
        return main_addrs[:2], main_lens[:2], list(idx_meta.kv_caches_base_addr), list(
            idx_meta.block_len
        )
    # Colocated: Indexer tensors follow Main k/v on the same layer_name.
    if len(main_addrs) >= 3:
        return main_addrs[:2], main_lens[:2], main_addrs[2:], main_lens[2:]
    return main_addrs[:2], main_lens[:2], [], []


def _map_remote_ids_by_indexer(
    mapped_indexer_ids: list[int],
    full_indexer_ids: list[int],
    full_main_ids: list[int],
) -> list[int]:
    """Map this rank's Indexer remote-id subset onto Main HOST remote ids.

    Prefill ``start_load_kv`` splits local/remote pairs using Indexer ids
    (``remote_block_ids[0]``). Main HOST ids live in ``[1]`` and must follow
    the same positional selection; Indexer blocks past ``len(full_main_ids)``
    have no HOST counterpart and are skipped.
    """
    if not mapped_indexer_ids or not full_main_ids:
        return []
    used = [False] * len(full_indexer_ids)
    out: list[int] = []
    for rid in mapped_indexer_ids:
        pos = None
        for i, fid in enumerate(full_indexer_ids):
            if not used[i] and fid == rid:
                pos = i
                break
        if pos is None:
            continue
        used[pos] = True
        if pos < len(full_main_ids):
            out.append(full_main_ids[pos])
    return out


def _expand_block_ids_for_pd(
    block_ids: list[int],
    remote_block_size: int,
    local_block_size: int,
    kernel_scale: int,
) -> list[int]:
    """Mirror Mooncake ``_align_remote_block_ids`` + ``_get_kernel_block_ids``."""
    ids = list(block_ids)
    if (
        remote_block_size != local_block_size
        and ids
        and remote_block_size > local_block_size
        and remote_block_size % local_block_size == 0
    ):
        ratio = remote_block_size // local_block_size
        ids = [b * ratio + j for b in ids for j in range(ratio)]
    if kernel_scale > 1 and ids:
        ids = [b * kernel_scale + j for b in ids for j in range(kernel_scale)]
    return ids


def _append_block_transfers(
    src_list: list[int],
    dst_list: list[int],
    length_list: list[int],
    src_base: int,
    dst_base: int,
    src_block_len: int,
    dst_block_len: int,
    local_block_ids: list[int],
    remote_block_ids: list[int],
    *,
    leg: str = "kv",
    allow_indexer_page_prefix: bool = False,
    page_slots: list[int] | None = None,
) -> None:
    """Append Mooncake SG entries for one tensor leg.

    Equal lens: contiguous groups.

    Indexer offload: Decode lightning uses ``PA_BSND`` with ``key.shape[1]`` as
    page tokens (512). Manager block id selects that tensor row:
    ``dst = base + id * dst_len + page_slot * src_len``.
    Prefill pages are shorter (128 tokens / ``src_len`` bytes); ``page_slot`` in
    ``[0, scale)`` packs multiple Prefill pages into one Decode row.
    """
    n = min(len(local_block_ids), len(remote_block_ids))
    if n == 0:
        return
    if src_block_len <= 0 or dst_block_len <= 0:
        raise RuntimeError(
            f"MooncakeToDram {leg} invalid block_len: src={src_block_len} dst={dst_block_len}"
        )

    if src_block_len == dst_block_len:
        grouped_remote, grouped_local = group_concurrent_contiguous(
            remote_block_ids[:n], local_block_ids[:n]
        )
        for group_remote, group_local in zip(grouped_remote, grouped_local):
            src_list.append(src_base + group_local[0] * src_block_len)
            dst_list.append(dst_base + group_remote[0] * dst_block_len)
            length_list.append(len(group_local) * src_block_len)
        return

    if not allow_indexer_page_prefix or dst_block_len % src_block_len != 0:
        raise RuntimeError(
            f"MooncakeToDram {leg} block_len mismatch: src={src_block_len} dst={dst_block_len}. "
            "Indexer allows dst_len multiple of src_len (copy Prefill page into Decode row). "
            "Main HOST requires equal lenses."
        )
    scale = dst_block_len // src_block_len
    for i, (local_id, remote_id) in enumerate(
        zip(local_block_ids[:n], remote_block_ids[:n])
    ):
        if page_slots is not None and i < len(page_slots):
            slot = int(page_slots[i])
        else:
            slot = 0
        if slot < 0 or slot >= scale:
            raise RuntimeError(
                f"MooncakeToDram {leg} page_slot={slot} out of range for scale={scale} "
                f"(src_len={src_block_len} dst_len={dst_block_len})"
            )
        src_list.append(src_base + local_id * src_block_len)
        dst_list.append(dst_base + remote_id * dst_block_len + slot * src_block_len)
        length_list.append(src_block_len)


def mooncake_to_dram_tp_ratio(*, tp_size: int, remote_tp_size: int) -> int:
    """Prefill-TP / Decode-TP ratio for side-channel port mapping."""
    remote = max(int(remote_tp_size), 1)
    local = int(tp_size)
    if local % remote != 0:
        raise RuntimeError(
            "MooncakeToDram requires Prefill TP divisible by Decode TP: "
            f"prefill_tp={local} decode_tp={remote}"
        )
    return local // remote


def mooncake_to_dram_remote_port(
    *,
    base_port: int,
    tp_rank: int,
    tp_size: int,
    remote_tp_size: int,
) -> int:
    """Map Prefill TP rank → Decode side-channel port (base + D-TP)."""
    ratio = mooncake_to_dram_tp_ratio(tp_size=tp_size, remote_tp_size=remote_tp_size)
    return int(base_port) + int(tp_rank) // ratio


def indexer_token_scale(*, src_block_len: int, dst_block_len: int) -> int:
    """Decode/Prefill Indexer ``block_len`` ratio (tensor row vs manager page)."""
    if src_block_len <= 0 or dst_block_len <= 0 or dst_block_len % src_block_len != 0:
        return 1
    return dst_block_len // src_block_len


def map_locals_to_indexer_pages(
    send_local: list[int],
    full_local: list[int],
    full_indexer: list[int],
    *,
    scale: int,
) -> tuple[list[int], list[int]]:
    """Map Prefill local pages → (Decode indexer manager id, page_slot).

    Physical packing: ``scale = dst_block_len / src_block_len`` (typically 4).
    Local position ``pos`` maps to ``indexer[pos // scale]`` at ``page_slot = pos % scale``.
    """
    if not send_local or not full_indexer:
        return [], []
    scale = max(1, int(scale))
    pos_by_id: dict[int, int] = {}
    for i, lid in enumerate(full_local):
        pos_by_id.setdefault(lid, i)
    n_idx = len(full_indexer)
    remote_ids: list[int] = []
    page_slots: list[int] = []
    for lid in send_local:
        pos = pos_by_id.get(lid, 0)
        remote_ids.append(full_indexer[min(pos // scale, n_idx - 1)])
        page_slots.append(pos % scale)
    return remote_ids, page_slots


def align_indexer_ids_to_local_window(
    send_local: list[int],
    full_local: list[int],
    full_indexer: list[int],
    *,
    scale: int | None = None,
) -> list[int]:
    """Map Prefill local block window → Decode Indexer ids (same length as window)."""
    if scale is None:
        n_local = max(len(full_local), 1)
        n_idx = max(len(full_indexer), 1)
        scale = max(1, (n_local + n_idx - 1) // n_idx)
    remote_ids, _slots = map_locals_to_indexer_pages(
        send_local, full_local, full_indexer, scale=scale
    )
    return remote_ids


def mooncake_to_dram_chunk_send_window(
    *,
    local_ids: list[int],
    remote_indexer_ids: list[int],
    local_computed_tokens: int,
    local_transed_tokens: int,
    local_bs: int,
    chunk_finish: bool,
    full_local_ids: list[int] | None = None,
    indexer_scale: int | None = None,
) -> tuple[list[int], list[int]] | None:
    """Compute this chunk's Prefill→Decode send window.

    Returns:
      - ``None``: drop the request from connector metadata (nothing to send /
        not finished).
      - ``([], [])``: keep the request (for last-layer DONE) with no payload.
      - ``(send_local, send_indexer)``: blocks to push this step.

    Window length is driven by Prefill **local** (Main-aligned) token progress,
    not ``len(remote_indexer_ids)``. Capping by Indexer length previously made
    ``chunk_finish=True`` chunks return empty after the first Indexer-sized
    window, so DONE never fired and trailing Main blocks were skipped.
    """
    if local_bs <= 0:
        return ([], []) if chunk_finish else None
    if chunk_finish:
        to_trans_idx = math.ceil(local_computed_tokens / local_bs)
    else:
        to_trans_idx = math.floor(local_computed_tokens / local_bs)
    transed_idx = math.floor(local_transed_tokens / local_bs)
    n = min(len(local_ids), to_trans_idx)
    start = min(max(transed_idx, 0), n)
    send_local = list(local_ids[start:n])
    if not send_local:
        return ([], []) if chunk_finish else None
    full_local = list(full_local_ids) if full_local_ids is not None else list(local_ids)
    send_indexer = align_indexer_ids_to_local_window(
        send_local,
        full_local,
        list(remote_indexer_ids),
        scale=indexer_scale,
    )
    return send_local, send_indexer


def _block_abs_stats(tensor: torch.Tensor, block_id: int) -> tuple[float, float, int]:
    if block_id < 0 or block_id >= tensor.shape[0]:
        return -1.0, -1.0, 0
    block = tensor[block_id].detach()
    vals = block.float() if block.is_floating_point() else block.to(torch.float32)
    return float(vals.abs().mean().item()), float(vals.abs().sum().item()), int(vals.numel())


def build_asymmetric_transfer_lists(
    *,
    layer_name: str,
    local_addrs: list[int],
    local_lens: list[int],
    local_block_ids: list[int],
    remote_layers: dict[str, Any],
    remote_indexer_ids: list[int],
    remote_main_ids: list[int],
    main_local_ids: list[int],
    remote_port: int | None,
    main_owner_port: int | None,
    skip_main: bool | None = None,
    skip_indexer: bool = False,
    full_local_ids: list[int] | None = None,
    full_indexer_ids: list[int] | None = None,
    indexer_page_slots: list[int] | None = None,
    indexer_addrs: list[int] | None = None,
    indexer_lens: list[int] | None = None,
    main_addrs: list[int] | None = None,
    main_lens: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Build Mooncake SG lists: Prefill Indexer/Main → D Indexer HBM / Main HOST.

    ``local_addrs`` may be 0713 colocated ``(k, v, indexer, ...)``. 0723 split
    groups should pass ``main_addrs`` / ``indexer_addrs`` instead.

    Destination meta is resolved by layer index (not same-name zip with Prefill).

    Indexer: ``dst = base + id * dst_len + page_slot * src_len`` (PA row + page
    packing). One Prefill TP per Decode TP must send Indexer (leader).

    ``skip_indexer`` / ``skip_main`` split D2D vs D2RH so ADXL TransferSync is
    not mixed (buffer mode requires one transfer type per batch).
    """
    src_list: list[int] = []
    dst_list: list[int] = []
    length_list: list[int] = []

    if main_addrs is None or main_lens is None:
        if len(local_addrs) >= 2:
            main_addrs = list(local_addrs[:2])
            main_lens = list(local_lens[:2])
        else:
            main_addrs, main_lens = [], []
    if indexer_addrs is None or indexer_lens is None:
        if len(local_addrs) >= 3:
            indexer_addrs = list(local_addrs[2:])
            indexer_lens = list(local_lens[2:])
        else:
            indexer_addrs, indexer_lens = [], []

    if not indexer_addrs and not main_addrs:
        return src_list, dst_list, length_list

    layer_idx = _layer_idx(layer_name)

    # --- Indexer D2D: Prefill indexer group → D Indexer HBM ---
    _indexer_name, indexer_remote = _find_remote_meta_by_layer_idx(
        remote_layers, layer_idx, want_indexer=True
    )
    if indexer_remote is None:
        indexer_name = _indexer_layer_name(layer_name)
        indexer_remote = remote_layers.get(indexer_name)
    if not skip_indexer and indexer_remote is not None and indexer_addrs:
        for src_base, src_len, dst_base, dst_len in zip(
            indexer_addrs,
            indexer_lens,
            indexer_remote.kv_caches_base_addr,
            indexer_remote.block_len,
        ):
            scale = indexer_token_scale(src_block_len=src_len, dst_block_len=dst_len)
            idx_ids = list(remote_indexer_ids)
            slots = list(indexer_page_slots) if indexer_page_slots is not None else None
            if full_local_ids is not None and full_indexer_ids is not None:
                idx_ids, slots = map_locals_to_indexer_pages(
                    list(local_block_ids),
                    list(full_local_ids),
                    list(full_indexer_ids),
                    scale=scale,
                )
            elif slots is None:
                slots = [0] * len(idx_ids)
            _append_block_transfers(
                src_list,
                dst_list,
                length_list,
                src_base,
                dst_base,
                src_len,
                dst_len,
                local_block_ids,
                idx_ids,
                leg="indexer",
                allow_indexer_page_prefix=True,
                page_slots=slots,
            )

    if skip_main is None:
        skip_main = False
    if skip_main:
        return src_list, dst_list, length_list
    if (
        remote_port is not None
        and main_owner_port is not None
        and int(remote_port) != int(main_owner_port)
    ):
        raise RuntimeError(
            "Main D2RH must target Decode TP0: "
            f"remote_port={remote_port}, owner_port={main_owner_port}"
        )

    # --- Main D2RH: Prefill TP0 attn K/V → Decode TP0 shared Host pool ---
    _main_name, main_remote = _find_remote_meta_by_layer_idx(
        remote_layers, layer_idx, want_indexer=False
    )
    if main_remote is None:
        main_remote = remote_layers.get(layer_name)
    if main_remote is None or len(main_remote.kv_caches_base_addr) < 2:
        raise RuntimeError(
            f"Decode TP0 did not advertise Main destination for {layer_name}"
        )
    if len(main_addrs) < 2:
        raise RuntimeError(
            f"Prefill TP0 did not provide Main K/V source for {layer_name}"
        )

    use_lens = main_lens if len(main_lens) >= 2 else local_lens
    for k in (0, 1):
        _append_block_transfers(
            src_list,
            dst_list,
            length_list,
            main_addrs[k],
            main_remote.kv_caches_base_addr[k],
            use_lens[k],
            main_remote.block_len[k],
            main_local_ids,
            remote_main_ids,
            leg="main_k" if k == 0 else "main_v",
        )
    return src_list, dst_list, length_list


def ensure_last_layer_done_signals(
    *,
    send_request: dict[str, Any],
    layer_idx: int,
    total_layers: int,
    already_signaled: set[str],
    failed_reqs: set[str],
    callback_func,
) -> set[str]:
    """Emit DONE/FAILED for chunk_finish reqs missing a payload-gated signal.

    Mooncake base only calls ``callback_func`` when ``len(src) > 0``. Prefill
    ranks mapped to non-TP0 Decode may skip Main and (if Indexer also empty)
    would otherwise leave Decode waiting forever.
    """
    if layer_idx != total_layers - 1:
        return already_signaled
    signaled = set(already_signaled)
    for req_id, req_meta in send_request.items():
        if not getattr(req_meta, "chunk_finish", False) or req_id in signaled:
            continue
        if req_id in failed_reqs:
            callback_func(req_id, req_meta, 0, trans_flag=False)
            failed_reqs.discard(req_id)
        else:
            callback_func(req_id, req_meta, 0, trans_flag=True)
        signaled.add(req_id)
        logger.info(
            "MooncakeToDram P empty-payload DONE layer=%s req=%s port=%s",
            layer_idx,
            req_id,
            getattr(req_meta, "remote_port", None),
        )
    return signaled


def transfer_layerwise_d2rh(send_thread, send_task, transfer_one_leg) -> None:
    """Finish both cache legs before signaling; always release the source."""
    layer_idx = send_task.layer_idx
    done_events = send_thread.layer_transfer_finished_events
    pending_events = send_thread.layer_transfer_pending_events
    if done_events is not None:
        done_events[layer_idx].clear()
    if pending_events is not None and send_task.send_request:
        pending_events[layer_idx].set()
    try:
        if send_task.wait_event is not None:
            send_task.wait_event.synchronize()
        # ADXL cannot mix Indexer NPU->NPU and Main NPU->Host in one batch.
        transfer_one_leg(send_task, skip_indexer=False, skip_main=True)
        transfer_one_leg(send_task, skip_indexer=True, skip_main=False)
    except Exception:
        # Remember failure until the last physical layer, including MTP.
        # Keeping the worker alive lets subsequent layers drain normally.
        send_thread.failed_reqs.update(send_task.send_request)
        logger.exception("MooncakeToDram layer %d transfer failed", layer_idx)
    finally:
        try:
            ensure_last_layer_done_signals(
                send_request=send_task.send_request,
                layer_idx=layer_idx,
                total_layers=send_thread.total_layers,
                already_signaled=set(),
                failed_reqs=send_thread.failed_reqs,
                callback_func=send_thread.callback_func,
            )
        finally:
            # A reshape/transfer/callback exception must not strand the
            # source-slot waiter. No transfer reads this source after exit.
            if pending_events is not None:
                pending_events[layer_idx].clear()
            if done_events is not None:
                done_events[layer_idx].set()


class MooncakeToDramDecodeScheduler(SFAPDCpuOffloadScheduler):
    """D scheduler: vLLM Main HOST + Indexer HBM ids; advertise for Mooncake Push.

    Main ids index the Decode shared Host pool (TP0 registered). Indexer ids
    index per-rank HBM. Mooncake Push puts both on the wire as
    ``remote_block_ids``.
    """

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return

        block_ids_by_group = SFAPDProducerScheduler._normalize_block_ids(blocks.get_block_ids())
        required_group = max(self.main_group_idx, self.indexer_group_idx)
        if len(block_ids_by_group) <= required_group:
            raise RuntimeError(
                "MooncakeToDram D allocation did not provide all SFA KV cache groups: "
                f"required={required_group + 1}, got={len(block_ids_by_group)}"
            )
        main_block_ids = list(block_ids_by_group[self.main_group_idx])
        indexer_block_ids = list(block_ids_by_group[self.indexer_group_idx])
        self._request_trackers[request.request_id] = (main_block_ids, indexer_block_ids)
        self._reqs_need_recv.add(request.request_id)

        # remote_port is D side-channel *base*. Prefill maps Indexer to
        # base + d_tp_rank; Main D2RH always targets Decode TP0 owner port.
        kv_transfer_params = dict(
            request_id=get_external_request_id(request.request_id),
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_block_ids=[
                list(indexer_block_ids),
                list(main_block_ids),
            ],
            remote_block_size=[
                self.block_size[self.indexer_group_idx],
                self.block_size[self.main_group_idx],
            ],
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
            remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            remote_cached_tokens=request.num_computed_tokens,
        )
        params["do_remote_prefill"] = False
        metaserver = params.get("metaserver")
        if metaserver is not None and not params.get("do_virtual", False):
            with self._metaserver_lock:
                self._cancelled_metaserver_requests.discard(request.request_id)
            self._submit_metaserver_request(
                request_id=request.request_id,
                url=metaserver,
                message=kv_transfer_params,
            )
        logger.info(
            "MooncakeLayerwiseToDram D advertised req %s: indexer=%s main_host=%s "
            "host=%s port_base=%s remote_tp=%s prompt_len=%s "
            "num_computed=%s num_external=%s remote_cached_tokens=%s",
            request.request_id,
            indexer_block_ids,
            main_block_ids,
            self.side_channel_host,
            self.side_channel_port,
            self.vllm_config.parallel_config.tensor_parallel_size,
            len(request.prompt_token_ids or []),
            request.num_computed_tokens,
            num_external_tokens,
            kv_transfer_params.get("remote_cached_tokens"),
        )


class MooncakeToDramDecodeWorker:
    """D worker: manager Host Main + Indexer HBM TE + Mooncake recv thread."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None,
    ):
        import os

        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.use_layerwise = use_layerwise
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * self.tp_size
        )
        self.engine_id = str(vllm_config.kv_transfer_config.engine_id)
        self.decode_manager = None
        self.layer_metadata: dict[str, MooncakeLayerMetadata] = {}
        self.request_map: dict[str, str] = {}
        self._cpu_blocks_by_req: dict[str, int] = {}
        self._main_ids_by_req: dict[str, list[int]] = {}
        self._indexer_ids_by_req: dict[str, list[int]] = {}
        self._invalid_block_ids: set[int] = set()
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.engine = None
        self.te_rpc_port: int | None = None
        self._num_offload_layers: int = 0
        self._pending_runner_host_pool: DSAHostKVPool | None = None

    def bind_runner_host_pool(self, pool: DSAHostKVPool) -> None:
        self._pending_runner_host_pool = pool

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        host_backend = resolve_sfa_kv_offload_backend(
            kv_transfer_extra_config(self.vllm_config),
            use_fused_overlap_offload=True,
        )
        if host_backend != SFA_KV_OFFLOAD_BACKEND_MOONCAKE:
            raise RuntimeError(
                "MooncakeLayerwiseToDramConnector requires "
                "sfa_kv_offload_backend=mooncake"
            )
        pool = self._pending_runner_host_pool
        if pool is None:
            raise RuntimeError(
                "ModelRunner must bind the DSA Host pool before KV registration"
            )
        if pool.topology.owner_rank != 0:
            raise RuntimeError("Decode TP0 must own the DSA Host pool")
        if pool.topology.tp_rank != self.tp_rank:
            raise RuntimeError(
                "DSA Host pool TP rank mismatch: "
                f"pool={pool.topology.tp_rank}, worker={self.tp_rank}"
            )

        self.decode_manager = get_kv_offload_decode_manager()
        if not hasattr(self.decode_manager, "offload_layer_names"):
            raise RuntimeError(
                "KVOffloadDecodeManager.register_kv_caches must run before the PD connector is registered"
            )
        if self.decode_manager.runner_host_pool is not pool:
            raise RuntimeError(
                "Decode manager and PD connector must share one DSA Host pool"
            )
        self._num_offload_layers = len(self.decode_manager.offload_layer_names)
        k_caches_cpu = pool.k_caches
        v_caches_cpu = pool.v_caches
        is_main_owner = pool.is_owner
        if is_main_owner:
            # TP0 registers the shared Host pool with TE; peers already map it.
            pool.register(self.engine)

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        _main_group_idx, indexer_group_idx = infer_sfa_component_group_ids(self.kv_cache_config)
        indexer_names: list[str] = []
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            for name in list(getattr(group, "layer_names", ()) or ()):
                if "indexer" in name.lower() and (
                    group_id == indexer_group_idx or indexer_group_idx == _main_group_idx
                ):
                    indexer_names.append(name)
        if not indexer_names:
            indexer_names = [n for n in kv_caches if "indexer" in n.lower()]
        indexer_by_idx = {_layer_idx(n): n for n in indexer_names}
        main_names = list(self.decode_manager.offload_layer_names)

        ptrs: list[int] = []
        registered: set[int] = set()

        def _reg(tensor: torch.Tensor, *, location: str | None, name: str) -> None:
            ptr = tensor.data_ptr()
            if ptr in registered:
                return
            nbytes = tensor.numel() * tensor.element_size()
            registered.add(ptr)
            ptrs.append(ptr)
            _register_te_memory(self.engine, ptr, nbytes, location=location, name=name)

        for mname in main_names:
            iname = indexer_by_idx.get(_layer_idx(mname))
            if iname is not None:
                indexer_tuple = kv_caches[iname]
                if not isinstance(indexer_tuple, (list, tuple)):
                    indexer_tuple = (indexer_tuple,)
                indexer_t = indexer_tuple[0]
                indexer_addrs = [indexer_t.data_ptr()]
                indexer_block_lens = [indexer_t.element_size() * math.prod(indexer_t.shape[1:])]
                indexer_scales = [indexer_t.shape[0] // num_blocks if num_blocks else 1]
                if iname == indexer_names[0]:
                    logger.info(
                        "MooncakeLayerwiseToDram D indexer shape=%s dtype=%s "
                        "block_len=%s num_blocks=%s tp=%s/%s",
                        tuple(indexer_t.shape),
                        indexer_t.dtype,
                        indexer_block_lens[0],
                        num_blocks,
                        self.tp_rank,
                        self.tp_size,
                    )
                _reg(indexer_t, location=None, name=f"indexer:{iname}")
                if len(indexer_tuple) > 1:
                    scale_t = indexer_tuple[1]
                    indexer_addrs.append(scale_t.data_ptr())
                    indexer_block_lens.append(scale_t.element_size() * math.prod(scale_t.shape[1:]))
                    indexer_scales.append(scale_t.shape[0] // num_blocks if num_blocks else 1)
                    _reg(scale_t, location=None, name=f"indexer_scale:{iname}")
                self.layer_metadata[iname] = MooncakeLayerMetadata(
                    tensor_group_idx=[_REMOTE_INDEXER_IDX],
                    kv_caches_base_addr=indexer_addrs,
                    block_len=indexer_block_lens,
                    block_size_scale=indexer_scales,
                )

            if is_main_owner:
                offload_id = self.decode_manager.layer_name_to_offload_id[mname]
                k_cpu = k_caches_cpu[offload_id]
                v_cpu = v_caches_cpu[offload_id]
                self.layer_metadata[mname] = MooncakeLayerMetadata(
                    tensor_group_idx=[_REMOTE_MAIN_IDX, _REMOTE_MAIN_IDX],
                    kv_caches_base_addr=[k_cpu.data_ptr(), v_cpu.data_ptr()],
                    block_len=[
                        k_cpu.element_size() * math.prod(k_cpu.shape[1:]),
                        v_cpu.element_size() * math.prod(v_cpu.shape[1:]),
                    ],
                    block_size_scale=[
                        k_cpu.shape[0] // num_blocks if num_blocks else 1,
                        v_cpu.shape[0] // num_blocks if num_blocks else 1,
                    ],
                )

        if ptrs:
            # Mark GlobalTE so later accidental register_buffer is a no-op.
            global_te.is_register_buffer = True
        logger.info(
            "MooncakeLayerwiseToDram D register: tp=%s/%s main_owner=%s "
            "indexer_buffers=%s host_pool_ptr=0x%x host_pool_bytes=%s "
            "layout=%s te_rpc_port=%s meta_layers=%s",
            self.tp_rank,
            self.tp_size,
            is_main_owner,
            len(ptrs),
            pool.data_ptr,
            pool.nbytes,
            pool.layout.fingerprint,
            self.te_rpc_port,
            list(self.layer_metadata.keys())[:6],
        )

        metadata = MooncakeAgentMetadata(
            te_rpc_port=self.te_rpc_port,
            layer_metadata=self.layer_metadata,
        )
        ready_event = threading.Event()
        self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
            self.tp_rank,
            self.side_channel_port,
            self.tp_size,
            get_ascend_config().pd_head_ratio,
            self.engine_id,
            metadata,
            ready_event,
        )
        self.kv_recv_layer_thread.start()
        ready_event.wait()
        logger.info(
            "MooncakeLayerwiseToDram D recv ready %s:%s tp_rank=%s",
            self.side_channel_host,
            self.side_channel_port + self.tp_rank,
            self.tp_rank,
        )

    def start_load_kv(self, metadata: KVConnectorMetadata) -> None:
        requests = getattr(metadata, "requests", None) or []
        logger.info(
            "MooncakeToDram D start_load_kv enter tp=%s meta_reqs=%s",
            self.tp_rank,
            [getattr(r, "req_id", None) for r in requests][:8],
        )
        for req in requests:
            req_id = getattr(req, "req_id", None)
            if req_id is None:
                continue
            ext_id = get_external_request_id(req_id)
            self.request_map[ext_id] = req_id
            main_ids = list(getattr(req, "main_block_ids", None) or getattr(req, "block_ids_cpu", None) or [])
            indexer_ids = list(
                getattr(req, "indexer_block_ids", None) or getattr(req, "block_ids_indexer", None) or []
            )
            self._cpu_blocks_by_req[req_id] = len(main_ids)
            self._main_ids_by_req[req_id] = main_ids
            self._indexer_ids_by_req[req_id] = indexer_ids

    def _debug_checksum_after_recv(self, req_id: str) -> None:
        if self.decode_manager is None or not getattr(self.decode_manager, "k_caches_cpu", None):
            return
        main_ids = self._main_ids_by_req.get(req_id) or []
        if not main_ids:
            return
        bid = int(main_ids[0])
        k_t = self.decode_manager.k_caches_cpu[0]
        v_t = self.decode_manager.v_caches_cpu[0]
        k_mean, k_sum, k_n = _block_abs_stats(k_t, bid)
        v_mean, v_sum, v_n = _block_abs_stats(v_t, bid)
        logger.info(
            "MooncakeToDram D main dst checksum req=%s tp=%s local_block=%s "
            "k_abs_mean=%.6g k_abs_sum=%.6g k_numel=%s "
            "v_abs_mean=%.6g v_abs_sum=%.6g v_numel=%s",
            req_id,
            self.tp_rank,
            bid,
            k_mean,
            k_sum,
            k_n,
            v_mean,
            v_sum,
            v_n,
        )

    def get_finished(self, finished_req_ids: set[str] | None = None) -> tuple[set[str], set[str]]:
        done_ext = (
            self.kv_recv_layer_thread.get_and_clear_done_requests()
            if self.kv_recv_layer_thread is not None
            else set()
        )
        failed_ext = (
            self.kv_recv_layer_thread.get_and_clear_failed_requests()
            if self.kv_recv_layer_thread is not None
            else set()
        )
        done_recving = {self.request_map[s] for s in done_ext if s in self.request_map}
        failed = {self.request_map[s] for s in failed_ext if s in self.request_map}
        for req_id in done_recving:
            try:
                self._debug_checksum_after_recv(req_id)
            except Exception as e:
                logger.warning(
                    "MooncakeToDram D main checksum failed req=%s err=%s",
                    req_id,
                    e,
                )
        for req_id in done_recving | failed:
            ext = get_external_request_id(req_id)
            self.request_map.pop(ext, None)
            self._cpu_blocks_by_req.pop(req_id, None)
            self._main_ids_by_req.pop(req_id, None)
            self._indexer_ids_by_req.pop(req_id, None)
        if finished_req_ids:
            for req_id in finished_req_ids:
                ext = get_external_request_id(req_id)
                self.request_map.pop(ext, None)
                self._cpu_blocks_by_req.pop(req_id, None)
                self._main_ids_by_req.pop(req_id, None)
                self._indexer_ids_by_req.pop(req_id, None)
        if envs.VLLM_ASCEND_SFA_DEBUG and (done_recving or failed):
            logger.info(
                "MooncakeLayerwiseToDram D get_finished done=%s failed=%s",
                done_recving,
                failed,
            )
        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def save_kv_layer(self, layer_name: str, *args, **kwargs) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def set_req_ids(self, req_ids: list) -> None:
        del req_ids

    def save_current_kv_tokens(self, *args, **kwargs) -> None:
        return

    def get_fused_overlap_cpu_kv_inputs(self, layer_name: str):
        assert self.decode_manager is not None
        return self.decode_manager.get_fused_overlap_cpu_kv_inputs(layer_name)

    def prepare_lru_resident_and_load(self, *args, **kwargs) -> bool:
        return False

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        result = {rid: self._cpu_blocks_by_req[rid] for rid in req_ids if rid in self._cpu_blocks_by_req}
        if not result:
            logger.warning(
                "MooncakeToDram D get_num_cpu_blocks miss req_ids=%s known=%s",
                req_ids,
                list(self._cpu_blocks_by_req.keys()),
            )
            return None
        return result


class MooncakeToDramProducerWorker(MooncakeLayerwiseConnectorWorker):
    """P worker: Indexer leaders D2D and TP0 Main D2RH.

    Prefill may be 0713 colocated ``(k, v, indexer)`` on one layer_name, or 0723
    split groups (``indexer.k_cache`` + ``self_attn.attn``). Decode
    ``enabled=true`` advertises two ``remote_block_ids`` lists. Do not index D
    destinations by Prefill ``num_kv_cache_groups``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_group_idx, self.indexer_group_idx = infer_sfa_component_group_ids(
            self.kv_cache_config
        )
        # attn layer_name -> (k, v, indexer) for puncture source checksums.
        self._p_kv_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._p_checksum_logged: set[tuple[str, int, str]] = set()

    def _maybe_log_p_src_checksum(
        self,
        *,
        layer_name: str,
        req_id: str,
        main_local_ids: list[int],
        indexer_local_ids: list[int],
        skip_main: bool | None,
        skip_indexer: bool,
        is_leader: bool,
    ) -> None:
        if not is_leader:
            return
        try:
            layer_idx = _layer_idx(layer_name)
        except AssertionError:
            return
        # Only first/last transformer layer to keep logs small.
        total = int(getattr(self, "total_layers", 0) or 0)
        if layer_idx not in (0, max(total - 1, 0)):
            return
        tensors = self._p_kv_tensors.get(layer_name)
        if tensors is None:
            return
        k_t, v_t, idx_t = tensors
        if skip_main is False and main_local_ids:
            key = (req_id, layer_idx, "main")
            if key not in self._p_checksum_logged:
                self._p_checksum_logged.add(key)
                bid = int(main_local_ids[0])
                k_mean, k_sum, k_n = _block_abs_stats(k_t, bid)
                v_mean, v_sum, v_n = _block_abs_stats(v_t, bid)
                logger.info(
                    "MooncakeToDram P main src checksum req=%s tp=%s layer=%s "
                    "local_block=%s k_abs_mean=%.6g k_abs_sum=%.6g k_numel=%s "
                    "v_abs_mean=%.6g v_abs_sum=%.6g v_numel=%s "
                    "k_ptr=0x%x v_ptr=0x%x",
                    req_id,
                    self.tp_rank,
                    layer_name,
                    bid,
                    k_mean,
                    k_sum,
                    k_n,
                    v_mean,
                    v_sum,
                    v_n,
                    k_t.data_ptr(),
                    v_t.data_ptr(),
                )
        if not skip_indexer and indexer_local_ids:
            key = (req_id, layer_idx, "indexer")
            if key not in self._p_checksum_logged:
                self._p_checksum_logged.add(key)
                bid = int(indexer_local_ids[0])
                mean, s, n = _block_abs_stats(idx_t, bid)
                logger.info(
                    "MooncakeToDram P indexer src checksum req=%s tp=%s layer=%s "
                    "local_block=%s abs_mean=%.6g abs_sum=%.6g numel=%s "
                    "shape=%s ptr=0x%x",
                    req_id,
                    self.tp_rank,
                    layer_name,
                    bid,
                    mean,
                    s,
                    n,
                    tuple(idx_t.shape),
                    idx_t.data_ptr(),
                )

    def _get_kv_split_metadata(self, req_meta, req_idx: int, req_id: str, group_idx: int):
        """All Prefill TP ranks push to the mapped Decode TP.

        Base Mooncake MLA CP selection only keeps ``Decode_TP`` Prefill ranks per
        request (e.g. TP0/TP1 of TP8), dropping Indexer heads. This connector maps:

          P TP[i] → D side-channel port base + i // (P_tp / D_tp)
          e.g. TP0-3 → D-TP0, TP4-7 → D-TP1.
        """
        remote_pcp = int(getattr(req_meta, "remote_pcp_size", 1) or 1)
        remote_dcp = int(getattr(req_meta, "remote_dcp_size", 1) or 1)
        if self.pcp_size * self.dcp_size != 1 or remote_pcp * remote_dcp != 1:
            # Context-parallel layouts still use the base splitter.
            return super()._get_kv_split_metadata(req_meta, req_idx, req_id, group_idx)

        remote_tp_size = int(getattr(req_meta, "remote_tp_size", 1) or 1)
        tp_ratio = mooncake_to_dram_tp_ratio(
            tp_size=self.tp_size, remote_tp_size=remote_tp_size
        )
        remote_host = req_meta.remote_host
        base_port = int(req_meta.remote_port)
        remote_port = mooncake_to_dram_remote_port(
            base_port=base_port,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            remote_tp_size=remote_tp_size,
        )

        local_bs = int(self.block_size[group_idx])
        local_transed_tokens = max(
            int(req_meta.remote_cache_tokens), int(req_meta.local_transed_tokens)
        )
        local_computed_tokens = int(req_meta.local_computed_tokens)
        local_all = list(req_meta.local_block_ids or [])
        remote_all = list(req_meta.remote_block_ids or [])
        local_ids = list(local_all[group_idx]) if group_idx < len(local_all) else []
        remote_ids = list(remote_all[group_idx]) if group_idx < len(remote_all) else []
        idx_g = self.indexer_group_idx
        main_g = self.main_group_idx
        split_groups = idx_g != main_g
        if split_groups and group_idx == idx_g:
            remote_ids = list(getattr(req_meta, "d_indexer_block_ids_full", None) or remote_ids)
            full_local = list(getattr(req_meta, "p_local_indexer_ids_full", None) or local_ids)
            full_indexer = list(getattr(req_meta, "d_indexer_block_ids_full", None) or remote_ids)
        elif split_groups and group_idx == main_g:
            remote_ids = list(getattr(req_meta, "d_main_block_ids_full", None) or remote_ids)
            full_local = list(getattr(req_meta, "p_local_main_ids_full", None) or local_ids)
            full_indexer = list(getattr(req_meta, "d_main_block_ids_full", None) or remote_ids)
        else:
            full_local = list(getattr(req_meta, "p_local_block_ids_full", None) or local_ids)
            full_indexer = list(getattr(req_meta, "d_indexer_block_ids_full", None) or remote_ids)

        # Prefer physical Indexer token scale (Decode PA row / Prefill page).
        indexer_scale = None
        if not split_groups or group_idx == idx_g:
            d_idx_bs = getattr(req_meta, "d_indexer_remote_block_size", None)
            if (
                d_idx_bs is not None
                and local_bs > 0
                and int(d_idx_bs) > local_bs
                and int(d_idx_bs) % local_bs == 0
            ):
                indexer_scale = int(d_idx_bs) // local_bs
        elif split_groups and group_idx == main_g:
            indexer_scale = 1
        window = mooncake_to_dram_chunk_send_window(
            local_ids=local_ids,
            remote_indexer_ids=full_indexer,
            local_computed_tokens=local_computed_tokens,
            local_transed_tokens=local_transed_tokens,
            local_bs=local_bs,
            chunk_finish=bool(req_meta.chunk_finish),
            full_local_ids=full_local,
            indexer_scale=indexer_scale,
        )
        if window is None:
            return {}
        send_local, send_remote = window

        if envs.VLLM_ASCEND_SFA_DEBUG:
            logger.info(
                "MooncakeToDram P kv_split req=%s tp=%s/%s → %s:%s "
                "tp_ratio=%s blocks=%s indexer_ids=%s indexer_scale=%s "
                "chunk_finish=%s computed=%s transed=%s",
                req_id,
                self.tp_rank,
                self.tp_size,
                remote_host,
                remote_port,
                tp_ratio,
                len(send_local),
                send_remote,
                indexer_scale,
                bool(req_meta.chunk_finish),
                local_computed_tokens,
                local_transed_tokens,
            )
        return {
            (remote_host, remote_port): {
                "local_block_ids": send_local,
                "remote_block_ids": send_remote,
                "trans_count": tp_ratio,
            }
        }

    def start_load_kv(self, metadata):
        # Preserve D-advertised Indexer/Main ids + side-channel base port before
        # the symmetric Mooncake rewrite.
        idx_g = self.indexer_group_idx
        main_g = self.main_group_idx
        split_groups = idx_g != main_g
        n_groups = self.num_kv_cache_groups
        for req_meta in metadata.requests.values():
            req_meta.main_owner_port = req_meta.remote_port
            remote_ids = list(req_meta.remote_block_ids or [])
            remote_bs = list(req_meta.remote_block_size or [])
            local_ids = list(req_meta.local_block_ids or [])
            req_meta.d_indexer_block_ids_full = (
                list(remote_ids[_REMOTE_INDEXER_IDX]) if len(remote_ids) > _REMOTE_INDEXER_IDX else []
            )
            req_meta.d_main_block_ids_full = (
                list(remote_ids[_REMOTE_MAIN_IDX]) if len(remote_ids) > _REMOTE_MAIN_IDX else []
            )
            req_meta.p_local_indexer_ids_full = (
                list(local_ids[idx_g]) if idx_g < len(local_ids) else []
            )
            req_meta.p_local_main_ids_full = (
                list(local_ids[main_g]) if main_g < len(local_ids) else []
            )
            req_meta.p_local_block_ids_full = (
                req_meta.p_local_indexer_ids_full or req_meta.p_local_main_ids_full
            )
            req_meta.d_indexer_remote_block_size = (
                remote_bs[_REMOTE_INDEXER_IDX] if len(remote_bs) > _REMOTE_INDEXER_IDX else None
            )
            req_meta.d_main_remote_block_size = (
                remote_bs[_REMOTE_MAIN_IDX] if len(remote_bs) > _REMOTE_MAIN_IDX else None
            )
            if split_groups:
                padded_remote = [[] for _ in range(n_groups)]
                padded_bs = [int(self.block_size[i]) for i in range(n_groups)]
                padded_remote[idx_g] = list(req_meta.d_indexer_block_ids_full)
                padded_remote[main_g] = list(req_meta.d_main_block_ids_full)
                req_meta.remote_block_ids = padded_remote
                req_meta.remote_block_size = padded_bs
            else:
                # Colocated Prefill: keep Indexer ids at [0] so group-0 split
                # still drives Indexer D2D; Main remotes stay on req_meta.
                if len(remote_ids) > _REMOTE_INDEXER_IDX:
                    req_meta.remote_block_ids = [list(remote_ids[_REMOTE_INDEXER_IDX])]
                local_bs0 = int(self.block_size[0]) if self.block_size else 0
                if local_bs0:
                    req_meta.remote_block_size = [local_bs0]
                elif len(remote_bs) > _REMOTE_INDEXER_IDX:
                    req_meta.remote_block_size = [remote_bs[_REMOTE_INDEXER_IDX]]
        super().start_load_kv(metadata)

        main_bs = int(self.block_size[main_g]) if main_g < len(self.block_size) else 0
        kernel_scale = (
            self.kernel_block_size_scale[main_g]
            if main_g < len(self.kernel_block_size_scale)
            else 1
        )
        for req_meta in metadata.requests.values():
            d_main_bs = getattr(req_meta, "d_main_remote_block_size", None)
            if d_main_bs is not None and main_bs and int(d_main_bs) != int(main_bs):
                raise RuntimeError(
                    "MooncakeToDram Main block_size mismatch: "
                    f"P local_bs={main_bs} D main_bs={d_main_bs}. "
                    "Main HOST id mapping assumes equal token/block sizes; "
                    "refusing Push to avoid silent misalignment."
                )
            if split_groups:
                rb = list(req_meta.remote_block_ids or [])
                lb = list(req_meta.local_block_ids or [])
                mapped_indexer = list(rb[idx_g]) if idx_g < len(rb) else []
                mapped_local = list(lb[idx_g]) if idx_g < len(lb) else []
                req_meta.mapped_main_block_ids = list(rb[main_g]) if main_g < len(rb) else []
                req_meta.mapped_main_local_block_ids = (
                    list(lb[main_g]) if main_g < len(lb) else []
                )
            else:
                mapped_indexer = (
                    list(req_meta.remote_block_ids[0]) if req_meta.remote_block_ids else []
                )
                mapped_local = (
                    list(req_meta.local_block_ids[0]) if req_meta.local_block_ids else []
                )
                full_local = list(getattr(req_meta, "p_local_block_ids_full", []) or [])
                full_main = list(getattr(req_meta, "d_main_block_ids_full", []) or [])
                expanded_full_local = _expand_block_ids_for_pd(
                    full_local, int(main_bs), int(main_bs), int(kernel_scale)
                )
                mapped_main: list[int] = []
                mapped_main_local: list[int] = []
                if mapped_local and full_main:
                    used = [False] * len(expanded_full_local)
                    for lid in mapped_local:
                        pos = None
                        for i, fid in enumerate(expanded_full_local):
                            if not used[i] and fid == lid:
                                pos = i
                                break
                        if pos is None:
                            continue
                        used[pos] = True
                        if pos < len(full_main):
                            mapped_main.append(full_main[pos])
                            mapped_main_local.append(lid)
                req_meta.mapped_main_block_ids = mapped_main
                req_meta.mapped_main_local_block_ids = mapped_main_local
            if envs.VLLM_ASCEND_SFA_DEBUG:
                logger.info(
                    "MooncakeToDram P start_load_kv port=%s owner=%s "
                    "indexer_mapped=%s local_mapped=%s main_mapped=%s split=%s",
                    req_meta.remote_port,
                    getattr(req_meta, "main_owner_port", None),
                    mapped_indexer,
                    mapped_local,
                    req_meta.mapped_main_block_ids,
                    split_groups,
                )

    def save_kv_layer(self, layer_name: str, *args, **kwargs) -> None:
        # 0723 split layout calls this for indexer.k_cache then self_attn.attn.
        # Indexer HBM is already filled; enqueue one send per transformer layer
        # on the Main attn callback so current_layer stays transformer-indexed.
        if "indexer" in layer_name.lower():
            return
        return super().save_kv_layer(layer_name, *args, **kwargs)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._p_kv_tensors = {}
        by_idx: dict[int, dict[str, Any]] = {}
        for layer_name, kv_tuple in kv_caches.items():
            if not isinstance(kv_tuple, (list, tuple)) or not kv_tuple:
                continue
            try:
                layer_idx = _layer_idx(layer_name)
            except AssertionError:
                continue
            slot = by_idx.setdefault(layer_idx, {})
            if "indexer" in layer_name.lower():
                slot["indexer"] = kv_tuple[0]
            else:
                slot["attn_name"] = layer_name
                slot["k"] = kv_tuple[0]
                if len(kv_tuple) > 1:
                    slot["v"] = kv_tuple[1]
                if len(kv_tuple) > 2:
                    slot.setdefault("indexer", kv_tuple[2])
        for slot in by_idx.values():
            attn_name = slot.get("attn_name")
            if attn_name and "k" in slot and "v" in slot and "indexer" in slot:
                self._p_kv_tensors[attn_name] = (slot["k"], slot["v"], slot["indexer"])
        send_thread = self.kv_send_layer_thread
        if send_thread is None:
            return

        def get_transfer_meta_asymmetric(
            send_task,
            req_id,
            req_meta,
            layer_group_idx,
            *,
            skip_indexer: bool = False,
            skip_main: bool | None = None,
        ):
            del layer_group_idx  # Prefill group count must not select D destinations.
            layer_name = send_task.layer_name
            main_addrs, main_lens, indexer_addrs, indexer_lens = _split_local_addrs_for_layer(
                send_thread.layer_metadata, layer_name
            )
            local_addrs = list(main_addrs) + list(indexer_addrs)
            local_lens = list(main_lens) + list(indexer_lens)
            idx_g = self.indexer_group_idx
            main_g = self.main_group_idx
            lb = list(req_meta.local_block_ids or [])
            rb = list(req_meta.remote_block_ids or [])
            indexer_local_ids = list(lb[idx_g]) if idx_g < len(lb) else (list(lb[0]) if lb else [])
            mapped_main_local = getattr(req_meta, "mapped_main_local_block_ids", None)
            if mapped_main_local is not None:
                main_local_ids = list(mapped_main_local)
            elif main_g < len(lb):
                main_local_ids = list(lb[main_g])
            else:
                main_local_ids = list(indexer_local_ids)
            remote_indexer_ids = (
                list(rb[idx_g]) if idx_g < len(rb) else (list(rb[0]) if rb else [])
            )
            mapped_main_remote = getattr(req_meta, "mapped_main_block_ids", None)
            if mapped_main_remote is not None:
                remote_main_ids = list(mapped_main_remote)
            elif main_g < len(rb):
                remote_main_ids = list(rb[main_g])
            else:
                remote_main_ids = []
            remote_layers = req_meta.remote_layer_metadata or {}
            remote_tp_size = int(getattr(req_meta, "remote_tp_size", 1) or 1)
            try:
                tp_ratio = mooncake_to_dram_tp_ratio(
                    tp_size=self.tp_size, remote_tp_size=remote_tp_size
                )
            except RuntimeError:
                tp_ratio = 1
            is_indexer_sender = (
                tp_ratio <= 1 or (self.tp_rank % tp_ratio) == 0
            )
            # Main has one writer: Prefill TP0 → Decode TP0 shared Host pool.
            is_main_sender = self.tp_rank == 0
            if not skip_indexer:
                skip_indexer = not is_indexer_sender
            if skip_main is not True:
                skip_main = not is_main_sender

            if not indexer_addrs and not skip_indexer:
                logger.warning(
                    "MooncakeToDram P layer %s has no Indexer src addrs. req=%s",
                    layer_name,
                    req_id,
                )

            self._maybe_log_p_src_checksum(
                layer_name=layer_name,
                req_id=req_id,
                main_local_ids=main_local_ids,
                indexer_local_ids=indexer_local_ids,
                skip_main=skip_main,
                skip_indexer=skip_indexer,
                is_leader=is_indexer_sender,
            )

            full_local_ids = list(
                getattr(req_meta, "p_local_indexer_ids_full", None)
                or getattr(req_meta, "p_local_block_ids_full", None)
                or indexer_local_ids
            )
            full_indexer_ids = list(
                getattr(req_meta, "d_indexer_block_ids_full", None) or remote_indexer_ids
            )

            src_list, dst_list, length_list = build_asymmetric_transfer_lists(
                layer_name=layer_name,
                local_addrs=local_addrs,
                local_lens=local_lens,
                local_block_ids=indexer_local_ids,
                remote_layers=remote_layers,
                remote_indexer_ids=remote_indexer_ids,
                remote_main_ids=remote_main_ids,
                main_local_ids=main_local_ids,
                remote_port=req_meta.remote_port,
                main_owner_port=getattr(req_meta, "main_owner_port", None),
                skip_main=skip_main,
                skip_indexer=skip_indexer,
                full_local_ids=full_local_ids,
                full_indexer_ids=full_indexer_ids,
                indexer_addrs=indexer_addrs,
                indexer_lens=indexer_lens,
                main_addrs=main_addrs,
                main_lens=main_lens,
            )
            # Always log Indexer SG (short-request gate); cheap and unblocks e2e debug.
            if not skip_indexer and remote_indexer_ids and indexer_lens:
                idx_src = int(indexer_lens[0])
                layer_idx = _layer_idx(layer_name)
                _n, idx_remote = _find_remote_meta_by_layer_idx(
                    remote_layers, layer_idx, want_indexer=True
                )
                if idx_remote is None:
                    idx_remote = remote_layers.get(_indexer_layer_name(layer_name))
                idx_row = (
                    int(idx_remote.block_len[0])
                    if idx_remote is not None and idx_remote.block_len
                    else None
                )
                packed_ids: list[int] = []
                packed_slots: list[int] = []
                if idx_row is not None and idx_src > 0:
                    scale = indexer_token_scale(
                        src_block_len=idx_src, dst_block_len=idx_row
                    )
                    packed_ids, packed_slots = map_locals_to_indexer_pages(
                        list(indexer_local_ids),
                        full_local_ids,
                        full_indexer_ids,
                        scale=scale,
                    )
                first_rid = int(packed_ids[0]) if packed_ids else int(remote_indexer_ids[0])
                first_slot = int(packed_slots[0]) if packed_slots else 0
                dst_offset = (
                    first_rid * idx_row + first_slot * idx_src
                    if idx_row is not None
                    else None
                )
                logger.info(
                    "MooncakeToDram P indexer SG layer=%s req=%s "
                    "mgr_id=%s page_slot=%s stride=dst_len=%s dst_offset=%s "
                    "copy_len=%s (PA row = id*dst_len + slot*src_len)",
                    layer_name,
                    req_id,
                    first_rid,
                    first_slot,
                    idx_row,
                    dst_offset,
                    idx_src,
                )
            if envs.VLLM_ASCEND_SFA_DEBUG:
                idx_src_len = int(indexer_lens[0]) if indexer_lens else None
                idx_dst_len = None
                token_scale = None
                packed_ids = None
                packed_slots = None
                if not skip_indexer:
                    layer_idx = _layer_idx(layer_name)
                    _n, idx_remote = _find_remote_meta_by_layer_idx(
                        remote_layers, layer_idx, want_indexer=True
                    )
                    if idx_remote is None:
                        idx_remote = remote_layers.get(_indexer_layer_name(layer_name))
                    if idx_remote is not None and idx_remote.block_len:
                        idx_dst_len = idx_remote.block_len[0]
                        if idx_src_len is not None:
                            token_scale = indexer_token_scale(
                                src_block_len=idx_src_len,
                                dst_block_len=idx_dst_len,
                            )
                            packed_ids, packed_slots = map_locals_to_indexer_pages(
                                list(indexer_local_ids),
                                full_local_ids,
                                full_indexer_ids,
                                scale=token_scale,
                            )
                logger.info(
                    "MooncakeToDram P transfer meta layer=%s req=%s port=%s "
                    "owner=%s xfer=%s skip_indexer=%s skip_main=%s main_ids=%s "
                    "tp_ratio=%s indexer_sender=%s main_sender=%s "
                    "idx_src_len=%s idx_dst_len=%s "
                    "token_scale=%s indexer_ids=%s page_slots=%s",
                    layer_name,
                    req_id,
                    req_meta.remote_port,
                    getattr(req_meta, "main_owner_port", None),
                    len(src_list),
                    skip_indexer,
                    skip_main,
                    remote_main_ids,
                    tp_ratio,
                    is_indexer_sender,
                    is_main_sender,
                    idx_src_len,
                    idx_dst_len,
                    token_scale,
                    packed_ids if not skip_indexer else None,
                    packed_slots if not skip_indexer else None,
                )
            return src_list, dst_list, length_list

        def _transfer_one_leg(
            send_task, *, skip_indexer: bool, skip_main: bool | None
        ) -> None:
            """One TransferSync batch: all legs must share the same ADXL type."""
            layer_name = send_task.layer_name
            layer_group_idx = send_thread.layer_metadata[layer_name].tensor_group_idx[0]
            session_meta: dict[str, TransferMeta] = {}
            for req_id, req_meta in send_task.send_request.items():
                session_id = f"{req_meta.remote_host}:{req_meta.remote_te_rpc_port}"
                if session_id not in session_meta:
                    session_meta[session_id] = TransferMeta(
                        src=[], dst=[], length=[], req_ids=[]
                    )
                src_list, dst_list, length_list = get_transfer_meta_asymmetric(
                    send_task,
                    req_id,
                    req_meta,
                    layer_group_idx,
                    skip_indexer=skip_indexer,
                    skip_main=skip_main,
                )
                session_meta[session_id].src.extend(src_list)
                session_meta[session_id].dst.extend(dst_list)
                session_meta[session_id].length.extend(length_list)
                session_meta[session_id].req_ids.append(req_id)

            for session_id, transfer_meta in session_meta.items():
                if len(transfer_meta.src) == 0:
                    continue
                ret = send_thread.engine.batch_transfer_sync_write(
                    session_id,
                    transfer_meta.src,
                    transfer_meta.dst,
                    transfer_meta.length,
                )
                if ret < 0:
                    logger.error(
                        "MooncakeToDram transfer failed (skip_indexer=%s "
                        "skip_main=%s). req_ids=%s, destination=%s, ret=%d.",
                        skip_indexer,
                        skip_main,
                        transfer_meta.req_ids,
                        session_id,
                        ret,
                    )
                    for rid in transfer_meta.req_ids:
                        send_thread.failed_reqs.add(rid)
                else:
                    logger.debug(
                        "MooncakeToDram Layer%d leg skip_indexer=%s "
                        "skip_main=%s write %dKB → [%s]",
                        send_task.layer_idx,
                        skip_indexer,
                        skip_main,
                        sum(transfer_meta.length) / 1024,
                        session_id,
                    )

        def _transfer_kv_cache_split_types(send_task):
            transfer_layerwise_d2rh(send_thread, send_task, _transfer_one_leg)

        send_thread.get_transfer_meta = get_transfer_meta_asymmetric  # type: ignore[method-assign]
        send_thread._transfer_kv_cache = _transfer_kv_cache_split_types  # type: ignore[method-assign]
        logger.info(
            "MooncakeLayerwiseToDram P: asymmetric Push "
            "(Indexer: all Prefill TP → mapped Decode TP HBM; "
            "Main: Prefill TP0 → Decode TP0 shared Host pool; "
            "separate TransferSync; DONE decoupled from payload)"
        )


class MooncakeLayerwiseD2RHConnector(KVConnectorBase_V1, SupportsHMA):
    """Layerwise Indexer D2D and Main D2RH connector.

    Indexer: Prefill ranks push to the mapped Decode TP HBM.
    Main: Prefill TP0 pushes to Decode TP0's registered shared Host pool;
    other Decode TP ranks read that pool through the share-segment local VA.
    """

    # Prefill/Decode PD senders still need real local block ids even when another
    # MultiConnector sibling owns prefix loading.
    requires_full_blocks_on_update_after_alloc = True
    # Upstream PR 14046: AscendStore waits on this before reusing Prefill HBM slots.
    supports_layerwise_buffer_reuse = True

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_consumer = vllm_config.kv_transfer_config.is_kv_consumer
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "use_layerwise", True
        )
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        init_ascend_config(vllm_config)
        decode_offload_enabled = get_ascend_config().kv_offload_decode_config.enabled
        if self.is_producer:
            assert not decode_offload_enabled, (
                "MooncakeLayerwiseToDramConnector producer (P) must run with "
                "kv_offload_decode_config.enabled=false."
            )
            if role == KVConnectorRole.SCHEDULER:
                # Prefill scheduler: reuse Mooncake layerwise unchanged.
                self._prefill = MooncakeLayerwiseConnector(vllm_config, role, kv_cache_config)
                self.connector_scheduler = self._prefill.connector_scheduler
                self.connector_worker = None
            else:
                self._prefill = None
                self.connector_scheduler = None
                self.connector_worker = MooncakeToDramProducerWorker(
                    vllm_config, kv_cache_config, str(self.engine_id)
                )
        else:
            assert decode_offload_enabled, (
                "MooncakeLayerwiseToDramConnector consumer (D) must run with "
                "kv_offload_decode_config.enabled=true."
            )
            assert get_ascend_config().kv_offload_decode_config.use_fused_overlap, (
                "MooncakeLayerwiseToDramConnector consumer requires "
                "kv_offload_decode_config.use_fused_overlap=true"
            )
            host_backend = resolve_sfa_kv_offload_backend(
                kv_transfer_extra_config(vllm_config),
                use_fused_overlap_offload=True,
            )
            if host_backend != SFA_KV_OFFLOAD_BACKEND_MOONCAKE:
                raise RuntimeError(
                    "MooncakeLayerwiseToDramConnector consumer requires "
                    "sfa_kv_offload_backend=mooncake"
                )
            self._prefill = None
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = MooncakeToDramDecodeScheduler(
                    vllm_config, self.use_layerwise, kv_cache_config
                )
                self.connector_worker = None
            else:
                self.connector_scheduler = None
                self.connector_worker = MooncakeToDramDecodeWorker(
                    vllm_config, self.use_layerwise, kv_cache_config
                )

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def bind_runner_host_pool(self, pool: DSAHostKVPool) -> None:
        assert self.connector_worker is not None
        if not hasattr(self.connector_worker, "bind_runner_host_pool"):
            raise RuntimeError(
                "MooncakeLayerwiseToDramConnector worker cannot bind DSA Host pool"
            )
        self.connector_worker.bind_runner_host_pool(pool)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        if self.is_consumer:
            return self.connector_worker.get_finished(finished_req_ids)
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.is_producer and self.connector_worker is not None:
            self.connector_worker.wait_for_layer_load(layer_name)

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        """Buffer-reuse gate for AscendStore (Mooncake ToDram push TE done)."""
        if not self.is_producer or self.connector_worker is None:
            return
        self.connector_worker.wait_for_layer_reuse(layer_idx)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        if not self.has_connector_metadata():
            return
        if self.is_producer:
            self.connector_worker.save_kv_layer(
                layer_name, kv_layer, attn_metadata, self._get_connector_metadata()
            )
            return
        self.connector_worker.save_kv_layer(layer_name)

    def wait_for_save(self):
        if self.is_consumer and self.connector_worker is not None:
            self.connector_worker.wait_for_save()

    def set_req_ids(self, req_ids: list):
        if self.is_consumer and self.connector_worker is not None:
            self.connector_worker.set_req_ids(req_ids)

    def save_current_kv_tokens(self, *args, **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.save_current_kv_tokens(*args, **kwargs)

    def get_fused_overlap_cpu_kv_inputs(self, layer_name: str):
        assert self.connector_worker is not None
        return self.connector_worker.get_fused_overlap_cpu_kv_inputs(layer_name)

    def prepare_lru_resident_and_load(self, *args, **kwargs) -> bool:
        assert self.connector_worker is not None
        return self.connector_worker.prepare_lru_resident_and_load(*args, **kwargs)

    def get_num_cpu_blocks(self, req_ids: list[str]) -> dict[str, int] | None:
        if self.connector_worker is None or not self.is_consumer:
            return None
        return self.connector_worker.get_num_cpu_blocks(req_ids)

    def get_layerwise_reuse_layer_count(self) -> int | None:
        worker = self.connector_worker
        if worker is None or not self.is_producer:
            return None
        return getattr(worker, "total_layers", None)


MooncakeLayerwiseToDramConnector = MooncakeLayerwiseD2RHConnector
