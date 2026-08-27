# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Mooncake→DRAM asymmetric Indexer/Main Push + DONE."""

from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_to_dram_connector import (
    _append_block_transfers,
    _expand_block_ids_for_pd,
    _find_remote_meta_by_layer_idx,
    _indexer_layer_name,
    _map_remote_ids_by_indexer,
    _split_local_addrs_for_layer,
    align_indexer_ids_to_local_window,
    build_asymmetric_transfer_lists,
    ensure_last_layer_done_signals,
    indexer_token_scale,
    map_locals_to_indexer_pages,
    mooncake_to_dram_chunk_send_window,
    mooncake_to_dram_remote_port,
    mooncake_to_dram_tp_ratio,
)


def _meta(addrs, lens):
    return SimpleNamespace(kv_caches_base_addr=list(addrs), block_len=list(lens))


def test_indexer_layer_name():
    assert (
        _indexer_layer_name("model.layers.3.self_attn")
        == "model.layers.3.self_attn.indexer.k_cache"
    )


def test_find_remote_meta_by_layer_idx():
    remote = {
        "model.layers.2.self_attn.indexer.k_cache": _meta([100], [10]),
        "model.layers.2.self_attn": _meta([200, 201], [20, 20]),
        "model.layers.3.self_attn.indexer.k_cache": _meta([300], [10]),
    }
    name, meta = _find_remote_meta_by_layer_idx(remote, 2, want_indexer=True)
    assert name.endswith("indexer.k_cache")
    assert meta.kv_caches_base_addr == [100]
    name, meta = _find_remote_meta_by_layer_idx(remote, 2, want_indexer=False)
    assert name == "model.layers.2.self_attn"
    assert meta.kv_caches_base_addr == [200, 201]
    name, meta = _find_remote_meta_by_layer_idx(remote, 9, want_indexer=True)
    assert name is None and meta is None


def test_map_remote_ids_by_indexer_positional():
    full_indexer = [10, 11, 12, 13]
    full_main = [100, 101, 102]
    mapped = [11, 13]
    assert _map_remote_ids_by_indexer(mapped, full_indexer, full_main) == [101]
    assert _map_remote_ids_by_indexer([12, 13], full_indexer, full_main) == [102]


def test_expand_block_ids_align_and_kernel():
    assert _expand_block_ids_for_pd([2, 4], remote_block_size=256, local_block_size=128, kernel_scale=1) == [
        4,
        5,
        8,
        9,
    ]
    assert _expand_block_ids_for_pd([2], remote_block_size=128, local_block_size=128, kernel_scale=2) == [
        4,
        5,
    ]


def test_append_block_transfers_contiguous():
    src, dst, length = [], [], []
    _append_block_transfers(
        src,
        dst,
        length,
        src_base=1000,
        dst_base=2000,
        src_block_len=10,
        dst_block_len=10,
        local_block_ids=[1, 2, 3],
        remote_block_ids=[7, 8, 9],
    )
    assert src == [1000 + 1 * 10]
    assert dst == [2000 + 7 * 10]
    assert length == [3 * 10]


def test_append_block_transfers_mismatch_raises():
    import pytest

    with pytest.raises(RuntimeError, match="block_len mismatch"):
        _append_block_transfers(
            [],
            [],
            [],
            src_base=0,
            dst_base=0,
            src_block_len=8,
            dst_block_len=16,
            local_block_ids=[0],
            remote_block_ids=[0],
            leg="main_k",
        )


def test_append_block_transfers_indexer_page_prefix():
    """Prefill 128-token page → Decode PA row (512 tokens): id * dst_len (+ slot0).

    e2e short request indexer_mapped=[4] must hit base+4*dst_len (tensor row4).
    """
    src_len, dst_len = 32768, 131072
    assert indexer_token_scale(src_block_len=src_len, dst_block_len=dst_len) == 4

    src, dst, length = [], [], []
    _append_block_transfers(
        src,
        dst,
        length,
        src_base=1000,
        dst_base=2000,
        src_block_len=src_len,
        dst_block_len=dst_len,
        local_block_ids=[2, 3],
        remote_block_ids=[4, 5],
        leg="indexer",
        allow_indexer_page_prefix=True,
    )
    assert src == [1000 + 2 * src_len, 1000 + 3 * src_len]
    assert dst == [2000 + 4 * dst_len, 2000 + 5 * dst_len]
    assert length == [src_len, src_len]


def test_append_block_transfers_indexer_page_slots():
    """Multiple Prefill pages pack into one Decode PA row via page_slot."""
    src_len, dst_len = 32768, 131072
    src, dst, length = [], [], []
    _append_block_transfers(
        src,
        dst,
        length,
        src_base=1000,
        dst_base=2000,
        src_block_len=src_len,
        dst_block_len=dst_len,
        local_block_ids=[0, 1, 2, 3],
        remote_block_ids=[4, 4, 4, 4],
        leg="indexer",
        allow_indexer_page_prefix=True,
        page_slots=[0, 1, 2, 3],
    )
    assert src == [1000 + i * src_len for i in range(4)]
    assert dst == [2000 + 4 * dst_len + slot * src_len for slot in range(4)]
    assert length == [src_len] * 4


def test_map_locals_to_indexer_pages_physical_scale():
    ids, slots = map_locals_to_indexer_pages(
        [1, 2, 3, 4, 5],
        list(range(1, 11)),
        [10, 11, 12],
        scale=4,
    )
    assert ids == [10, 10, 10, 10, 11]
    assert slots == [0, 1, 2, 3, 0]


def test_build_asymmetric_transfer_lists_indexer_page_prefix():
    """Indexer uses id * dst_len (PA row); Main stays equal-lens."""
    layer = "model.layers.1.self_attn"
    src_idx, dst_idx = 32768, 131072
    remote = {
        "model.layers.1.self_attn.indexer.k_cache": _meta([9000], [dst_idx]),
        "model.layers.1.self_attn": _meta([7000, 7005], [8, 8]),
    }
    # e2e: indexer_mapped=[4] → dst = base + 4*dst_len (tensor row4)
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[100, 200, 300],
        local_lens=[8, 8, src_idx],
        local_block_ids=[0],
        remote_layers=remote,
        remote_indexer_ids=[4],
        remote_main_ids=[20],
        main_local_ids=[0],
        remote_port=10001,
        main_owner_port=10001,
    )
    assert len(src) == 3
    assert dst[0] == 9000 + 4 * dst_idx
    assert length[0] == src_idx
    assert dst[1] == 7000 + 20 * 8
    assert dst[2] == 7005 + 20 * 8


def test_build_asymmetric_indexer_page_packing():
    """Same Decode manager id + distinct page_slots from physical scale."""
    layer = "model.layers.2.self_attn"
    src_idx, dst_idx = 32768, 131072
    remote = {
        "model.layers.2.self_attn.indexer.k_cache": _meta([5000], [dst_idx]),
        "model.layers.2.self_attn": _meta([6000, 6001], [8, 8]),
    }
    # 4 Prefill pages → one Decode PA row (mgr id 7), slots 0..3.
    # Window may wrongly advertise duplicated ids without slots; full lists fix it.
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[1, 2, 3],
        local_lens=[8, 8, src_idx],
        local_block_ids=[10, 11, 12, 13],
        remote_layers=remote,
        remote_indexer_ids=[7, 7, 7, 7],
        remote_main_ids=[],
        main_local_ids=[],
        remote_port=10001,
        main_owner_port=10001,
        skip_main=True,
        full_local_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        full_indexer_ids=[7, 8],
    )
    assert length == [src_idx] * 4
    assert dst == [5000 + 7 * dst_idx + slot * src_idx for slot in range(4)]
    assert src == [3 + lid * src_idx for lid in (10, 11, 12, 13)]


def test_build_asymmetric_indexer_distinct_pa_rows():
    """Distinct manager ids map to distinct Decode PA rows (id * dst_len)."""
    layer = "model.layers.2.self_attn"
    src_idx, dst_idx = 32768, 131072
    remote = {
        "model.layers.2.self_attn.indexer.k_cache": _meta([5000], [dst_idx]),
        "model.layers.2.self_attn": _meta([6000, 6001], [8, 8]),
    }
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[1, 2, 3],
        local_lens=[8, 8, src_idx],
        local_block_ids=[0, 1, 2],
        remote_layers=remote,
        remote_indexer_ids=[4, 5, 6],
        remote_main_ids=[1, 1, 1],
        main_local_ids=[0, 1, 2],
        remote_port=10001,
        main_owner_port=10001,
        skip_main=True,
    )
    assert length == [src_idx, src_idx, src_idx]
    assert dst == [
        5000 + 4 * dst_idx,
        5000 + 5 * dst_idx,
        5000 + 6 * dst_idx,
    ]


def test_build_asymmetric_transfer_lists_p1group_d2group_tp0():
    """P co-located 3 tensors; D publishes Indexer + Main under different keys."""
    layer = "model.layers.1.self_attn"
    # Decode keys differ from a naive zip of P local meta (3 vs 2).
    remote = {
        "model.layers.1.self_attn.indexer.k_cache": _meta([9000], [5]),
        "model.layers.1.self_attn": _meta([7000, 7005], [8, 8]),
    }
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[100, 200, 300],
        local_lens=[8, 8, 5],
        local_block_ids=[0, 1],
        remote_layers=remote,
        remote_indexer_ids=[10, 11],
        remote_main_ids=[20, 21],
        main_local_ids=[0, 1],
        remote_port=10001,  # D-TP0 base
        main_owner_port=10001,
    )
    # Indexer (1 leg) + Main k/v (2 legs) = 3 contiguous groups
    assert len(src) == 3
    assert dst[0] == 9000 + 10 * 5  # indexer
    assert dst[1] == 7000 + 20 * 8  # main k
    assert dst[2] == 7005 + 20 * 8  # main v
    assert src[0] == 300 + 0 * 5
    assert src[1] == 100 + 0 * 8
    assert src[2] == 200 + 0 * 8


def test_build_asymmetric_transfer_lists_rejects_non_owner_main_target():
    import pytest

    layer = "model.layers.1.self_attn"
    remote = {
        "model.layers.1.self_attn.indexer.k_cache": _meta([9000], [5]),
        "model.layers.1.self_attn": _meta([7000, 7005], [8, 8]),
    }
    with pytest.raises(RuntimeError, match="Decode TP0"):
        build_asymmetric_transfer_lists(
            layer_name=layer,
            local_addrs=[100, 200, 300],
            local_lens=[8, 8, 5],
            local_block_ids=[0],
            remote_layers=remote,
            remote_indexer_ids=[10],
            remote_main_ids=[20],
            main_local_ids=[0],
            remote_port=10002,
            main_owner_port=10001,
        )


def test_build_asymmetric_transfer_lists_explicit_skip_main():
    """Explicit skip_main=True keeps Indexer-only TransferSync batch."""
    layer = "model.layers.1.self_attn"
    remote = {
        "model.layers.1.self_attn.indexer.k_cache": _meta([9000], [5]),
        "model.layers.1.self_attn": _meta([7000, 7005], [8, 8]),
    }
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[100, 200, 300],
        local_lens=[8, 8, 5],
        local_block_ids=[0],
        remote_layers=remote,
        remote_indexer_ids=[10],
        remote_main_ids=[20],
        main_local_ids=[0],
        remote_port=10002,
        main_owner_port=10001,
        skip_main=True,
    )
    assert len(src) == 1
    assert dst == [9000 + 10 * 5]
    assert src == [300]


def test_build_asymmetric_transfer_lists_split_d2d_vs_d2rh():
    """Indexer-only and Main-only legs must not share one TransferSync batch."""
    layer = "model.layers.0.self_attn"
    remote = {
        "model.layers.0.self_attn.indexer.k_cache": _meta([9000], [5]),
        "model.layers.0.self_attn": _meta([7000, 8000], [8, 8]),
    }
    common = dict(
        layer_name=layer,
        local_addrs=[100, 200, 300],
        local_lens=[8, 8, 5],
        local_block_ids=[0],
        remote_layers=remote,
        remote_indexer_ids=[10],
        remote_main_ids=[20],
        main_local_ids=[0],
        remote_port=10001,
        main_owner_port=10001,
    )
    idx_src, idx_dst, _ = build_asymmetric_transfer_lists(
        **common, skip_main=True, skip_indexer=False
    )
    main_src, main_dst, _ = build_asymmetric_transfer_lists(
        **common, skip_main=False, skip_indexer=True
    )
    assert len(idx_src) == 1
    assert idx_src == [300]
    assert idx_dst == [9000 + 10 * 5]
    assert len(main_src) == 2
    assert main_src == [100, 200]
    assert main_dst == [7000 + 20 * 8, 8000 + 20 * 8]
    # Combined would be 3; split legs stay type-homogeneous for ADXL.
    assert len(idx_src) + len(main_src) == 3


def test_build_asymmetric_0723_split_group_addrs():
    """0723 Prefill: indexer and main are independent groups, not addrs[2]."""
    layer = "model.layers.0.self_attn.attn"
    remote = {
        "model.layers.0.self_attn.indexer.k_cache": _meta([9000], [5]),
        "model.layers.0.self_attn.attn": _meta([7000, 8000], [8, 8]),
    }
    src, dst, length = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[],
        local_lens=[],
        local_block_ids=[0],
        remote_layers=remote,
        remote_indexer_ids=[10],
        remote_main_ids=[20],
        main_local_ids=[1],
        remote_port=10001,
        main_owner_port=10001,
        indexer_addrs=[300],
        indexer_lens=[5],
        main_addrs=[100, 200],
        main_lens=[8, 8],
    )
    assert src == [300, 100 + 1 * 8, 200 + 1 * 8]
    assert dst == [9000 + 10 * 5, 7000 + 20 * 8, 8000 + 20 * 8]
    assert length == [5, 8, 8]


def test_split_local_addrs_for_layer_two_groups():
    local = {
        "model.layers.2.self_attn.indexer.k_cache": _meta([300], [5]),
        "model.layers.2.self_attn.attn": _meta([100, 200], [8, 8]),
    }
    main_addrs, main_lens, idx_addrs, idx_lens = _split_local_addrs_for_layer(
        local, "model.layers.2.self_attn.attn"
    )
    assert main_addrs == [100, 200]
    assert main_lens == [8, 8]
    assert idx_addrs == [300]
    assert idx_lens == [5]


def test_split_local_addrs_for_layer_colocated():
    local = {"model.layers.0.self_attn.attn": _meta([100, 200, 300], [8, 8, 5])}
    main_addrs, main_lens, idx_addrs, idx_lens = _split_local_addrs_for_layer(
        local, "model.layers.0.self_attn.attn"
    )
    assert main_addrs == [100, 200]
    assert idx_addrs == [300]
    assert idx_lens == [5]


def test_build_asymmetric_layer_idx_not_same_name():
    """Indexer resolved by layer_idx even when Prefill name ≠ Decode key prefix."""
    # Prefill may use a slightly different attn module path; Decode uses standard keys.
    layer = "model.layers.4.self_attn"
    remote = {
        "model.layers.4.self_attn.indexer.k_cache": _meta([111], [4]),
        "model.layers.4.self_attn": _meta([222, 223], [6, 6]),
    }
    src, dst, _ = build_asymmetric_transfer_lists(
        layer_name=layer,
        local_addrs=[1, 2, 3],
        local_lens=[6, 6, 4],
        local_block_ids=[1],
        remote_layers=remote,
        remote_indexer_ids=[2],
        remote_main_ids=[3],
        main_local_ids=[1],
        remote_port=1,
        main_owner_port=1,
    )
    assert dst[0] == 111 + 2 * 4
    assert len(src) == 3


def test_ensure_last_layer_done_when_payload_empty():
    calls = []

    def cb(req_id, req_meta, group_idx, trans_flag=True):
        calls.append((req_id, group_idx, trans_flag, req_meta.remote_port))

    req_meta = SimpleNamespace(chunk_finish=True, remote_port=10002, trans_count=[1])
    signaled = ensure_last_layer_done_signals(
        send_request={"r1": req_meta},
        layer_idx=3,
        total_layers=4,
        already_signaled=set(),
        failed_reqs=set(),
        callback_func=cb,
    )
    assert signaled == {"r1"}
    assert calls == [("r1", 0, True, 10002)]


def test_ensure_last_layer_done_skips_non_last_and_already_signaled():
    calls = []

    def cb(req_id, req_meta, group_idx, trans_flag=True):
        calls.append(req_id)

    req_meta = SimpleNamespace(chunk_finish=True, remote_port=1, trans_count=[1])
    ensure_last_layer_done_signals(
        send_request={"r1": req_meta},
        layer_idx=1,
        total_layers=4,
        already_signaled=set(),
        failed_reqs=set(),
        callback_func=cb,
    )
    assert calls == []

    ensure_last_layer_done_signals(
        send_request={"r1": req_meta},
        layer_idx=3,
        total_layers=4,
        already_signaled={"r1"},
        failed_reqs=set(),
        callback_func=cb,
    )
    assert calls == []


def test_chunk_send_window_not_capped_by_indexer_len():
    """Reproduce e2e hang: indexer=3, main/local=10, chunked prefill 1024+218."""
    local = list(range(1, 11))
    indexer = [1, 2, 3]
    # Chunk 1: 1024 tokens, not finished → 8 local blocks (not 3).
    w1 = mooncake_to_dram_chunk_send_window(
        local_ids=local[:8],
        remote_indexer_ids=indexer,
        local_computed_tokens=1024,
        local_transed_tokens=0,
        local_bs=128,
        chunk_finish=False,
        full_local_ids=local,
    )
    assert w1 is not None
    send_local, send_idx = w1
    assert send_local == list(range(1, 9))
    assert len(send_idx) == 8
    assert send_idx[0] == 1 and send_idx[-1] == 2

    # Chunk 2: remaining tokens, finished → trailing locals + DONE-capable window.
    w2 = mooncake_to_dram_chunk_send_window(
        local_ids=local,
        remote_indexer_ids=indexer,
        local_computed_tokens=1242,
        local_transed_tokens=1024,
        local_bs=128,
        chunk_finish=True,
        full_local_ids=local,
    )
    assert w2 is not None
    send_local, send_idx = w2
    assert send_local == [9, 10]
    assert send_idx == [3, 3]


def test_chunk_send_window_physical_indexer_scale():
    """Explicit scale=4 packs 4 Prefill pages into each Decode indexer id."""
    local = list(range(1, 11))
    indexer = [1, 2, 3]
    w = mooncake_to_dram_chunk_send_window(
        local_ids=local[:8],
        remote_indexer_ids=indexer,
        local_computed_tokens=1024,
        local_transed_tokens=0,
        local_bs=128,
        chunk_finish=False,
        full_local_ids=local,
        indexer_scale=4,
    )
    assert w is not None
    send_local, send_idx = w
    assert send_local == list(range(1, 9))
    assert send_idx == [1, 1, 1, 1, 2, 2, 2, 2]


def test_chunk_send_window_empty_finish_keeps_request():
    # Already fully sent; final chunk_finish must not drop the req (DONE path).
    w = mooncake_to_dram_chunk_send_window(
        local_ids=[1, 2, 3],
        remote_indexer_ids=[1, 2, 3],
        local_computed_tokens=384,
        local_transed_tokens=384,
        local_bs=128,
        chunk_finish=True,
    )
    assert w == ([], [])
    assert (
        mooncake_to_dram_chunk_send_window(
            local_ids=[1, 2, 3],
            remote_indexer_ids=[1, 2, 3],
            local_computed_tokens=384,
            local_transed_tokens=384,
            local_bs=128,
            chunk_finish=False,
        )
        is None
    )


def test_align_indexer_ids_to_local_window():
    assert align_indexer_ids_to_local_window(
        [1, 2, 3, 4, 5, 6, 7, 8],
        list(range(1, 11)),
        [10, 11, 12],
    ) == [10, 10, 10, 10, 11, 11, 11, 11]


def test_mooncake_to_dram_port_map_p8_d2():
    """P TP8 → D TP2: TP0-3 → base, TP4-7 → base+1."""
    assert mooncake_to_dram_tp_ratio(tp_size=8, remote_tp_size=2) == 4
    base = 10010
    for tp in range(4):
        assert (
            mooncake_to_dram_remote_port(
                base_port=base, tp_rank=tp, tp_size=8, remote_tp_size=2
            )
            == base
        )
    for tp in range(4, 8):
        assert (
            mooncake_to_dram_remote_port(
                base_port=base, tp_rank=tp, tp_size=8, remote_tp_size=2
            )
            == base + 1
        )


def test_indexer_token_scale_helper():
    assert indexer_token_scale(src_block_len=8, dst_block_len=32) == 4
    assert indexer_token_scale(src_block_len=32, dst_block_len=32) == 1
    assert indexer_token_scale(src_block_len=0, dst_block_len=32) == 1
