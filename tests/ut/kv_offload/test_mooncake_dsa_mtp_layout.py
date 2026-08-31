# SPDX-License-Identifier: Apache-2.0
"""Blockwise MTP cache identities, complete transfers and wire compatibility."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread,
    MooncakeAgentMetadata,
    MooncakeConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_layout import (
    add_dsa_cache_descriptor,
    dsa_cache_key,
    infer_dsa_block_group_ids,
    project_dsa_remote_arrays,
    select_dsa_block_groups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaLocalResultKind,
    DsaStepRequest,
    RemoteEndpoint,
    RemoteSource,
)


class TestDsaCacheIdentity(unittest.TestCase):
    def test_main_and_indexer_share_physical_identity_not_wire_row(self):
        for prefix in ("model.layers.2", "model.layers.2.mtp_block", "mtp.0"):
            with self.subTest(prefix=prefix):
                self.assertEqual(dsa_cache_key(f"{prefix}.self_attn.attn", 2), "main:2")
                self.assertEqual(dsa_cache_key(f"{prefix}.self_attn.indexer.k_cache", 2), "indexer:2")

    def test_multiple_physical_draft_layers_are_distinct(self):
        self.assertEqual(dsa_cache_key("mtp.1.self_attn.attn", 2), "main:3")

    def test_unknown_or_negative_layer_is_rejected(self):
        for name in ("self_attn", "model.layers.bad", "model.layers", "model.layers.-1.attn"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dsa_cache_key(name, 2)

    def test_descriptor_records_real_append_offset(self):
        layout = {}
        add_dsa_cache_descriptor(
            layout,
            layer_name="model.layers.2.self_attn.attn",
            num_target_layers=2,
            wire_layer=7,
            first_position=0,
            tensor_count=2,
        )
        add_dsa_cache_descriptor(
            layout,
            layer_name="model.layers.2.self_attn.indexer.k_cache",
            num_target_layers=2,
            wire_layer=7,
            first_position=2,
            tensor_count=2,
        )
        self.assertEqual(layout, {"main:2": (7, 0, 2), "indexer:2": (7, 2, 2)})

    def test_colocated_indexer_has_a_separate_descriptor(self):
        layout = {}
        add_dsa_cache_descriptor(
            layout,
            layer_name="model.layers.2.self_attn.attn",
            num_target_layers=2,
            wire_layer=9,
            first_position=0,
            tensor_count=4,
        )
        self.assertEqual(layout, {"main:2": (9, 0, 2), "indexer:2": (9, 2, 2)})

    def test_duplicate_physical_identity_is_rejected(self):
        layout = {"main:2": (9, 0, 2)}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            add_dsa_cache_descriptor(
                layout,
                layer_name="mtp.0.self_attn.attn",
                num_target_layers=2,
                wire_layer=10,
                first_position=0,
                tensor_count=2,
            )
        self.assertEqual(layout, {"main:2": (9, 0, 2)})

    def test_invalid_component_count_is_rejected(self):
        for name, count in (("model.layers.0.attn", 1), ("model.layers.0.indexer.k_cache", 3)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                add_dsa_cache_descriptor(
                    {}, layer_name=name, num_target_layers=1, wire_layer=0, first_position=0, tensor_count=count
                )


class TestDsaBlockGroups(unittest.TestCase):
    def test_group_order_is_inferred_from_component_names(self):
        main = ["model.layers.0.attn", "model.layers.1.mtp_block.attn"]
        indexer = ["model.layers.0.indexer.k_cache", "model.layers.1.mtp_block.indexer.k_cache"]
        for groups, expected in (
            ([main, indexer], {"main": 0, "indexer": 1}),
            ([indexer, main], {"main": 1, "indexer": 0}),
            ([main + indexer], {"main": 0, "indexer": 0}),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(infer_dsa_block_group_ids(groups), expected)

    def test_separate_draft_manager_group_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "per-layer block-ID routing"):
            infer_dsa_block_group_ids([["model.layers.0.attn"], ["mtp.0.attn"]])

    def test_source_group_ids_override_legacy_first_last_order(self):
        self.assertEqual(select_dsa_block_groups(((11,), (22,)), {"main": 0, "indexer": 1}), ((11,), (22,)))
        self.assertEqual(select_dsa_block_groups(((11,),), {"main": 0, "indexer": 0}), ((11,), (11,)))

    def test_legacy_group_order_remains_available(self):
        self.assertEqual(select_dsa_block_groups(((11,), (22,)), None), ((22,), (11,)))

    def test_invalid_source_group_mapping_is_rejected(self):
        for mapping in ({"main": 0}, {"main": True, "indexer": 0}, {"main": 2, "indexer": 0}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                select_dsa_block_groups(((11,),), mapping)

    def test_prefill_publishes_group_ids_with_trimmed_prompt_blocks(self):
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.block_size = 16
        scheduler._get_transfer_block_ids = lambda groups, _length: tuple(group[:1] for group in groups)
        scheduler._get_swa_transfer_block_ids = lambda groups: groups
        scheduler._reqs_need_send = {}
        scheduler._dsa_debug_source_blocks = {}
        scheduler._dsa_main_group_idx = 0
        scheduler._dsa_indexer_group_idx = 1
        scheduler.engine_id = "P"
        scheduler.side_channel_host = "host"
        scheduler.side_channel_port = 5000
        scheduler.pcp_size = scheduler.dcp_size = scheduler.tp_size = 1
        scheduler.multi_nodes_meta_mapping = {}
        request = SimpleNamespace(
            request_id="req",
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[4],
        )
        delayed, params = scheduler.request_finished(request, ((11, 12), (21, 22)))
        self.assertTrue(delayed)
        self.assertEqual(params["remote_block_ids"], ((11,), (21,)))
        self.assertEqual(params["dsa_block_group_ids"], {"main": 0, "indexer": 1})


class TestDsaRemoteProjection(unittest.TestCase):
    def setUp(self):
        self.local = [[], [(0, 100, 8, 8, 1), (1, 200, 8, 8, 1)]]
        self.keys = {1: "main:2"}
        self.remote_arrays = (
            [[1], [], [30, 40, 50]],
            [[8], [], [8, 16, 24]],
            [[1], [], [1, 1, 1]],
            [[8], [], [8, 8, 8]],
        )
        self.remote_layout = {"main:2": (2, 1, 2)}

    def project(self):
        return project_dsa_remote_arrays(
            self.local, self.keys, self.remote_layout, self.remote_arrays, num_target_layers=2
        )

    def test_projection_uses_remote_identity_and_offset(self):
        result = self.project()
        self.assertEqual(result, ([[], [40, 50]], [[], [16, 24]], [[], [1, 1]], [[], [8, 8]]))
        self.assertEqual(self.remote_arrays[0], [[1], [], [30, 40, 50]])

    def test_missing_mtp_main_is_rejected(self):
        self.remote_layout = {"main:0": (0, 0, 1)}
        with self.assertRaisesRegex(ValueError, "missing required cache main:2"):
            self.project()

    def test_missing_mtp_indexer_is_rejected_even_with_main(self):
        self.keys = {1: "indexer:2"}
        with self.assertRaisesRegex(ValueError, "missing required cache indexer:2"):
            self.project()

    def test_shared_target_indexer_can_be_absent(self):
        self.keys = {1: "indexer:0"}
        self.remote_layout = {"main:0": (2, 1, 2)}
        self.assertTrue(all(not any(array) for array in self.project()))

    def test_missing_whole_target_layer_is_not_shared_indexer(self):
        self.keys = {1: "indexer:0"}
        with self.assertRaisesRegex(ValueError, "missing required cache indexer:0"):
            self.project()

    def test_mismatched_key_scale_component_count_is_rejected(self):
        self.remote_layout = {"main:2": (2, 1, 1)}
        with self.assertRaisesRegex(ValueError, "component count mismatch"):
            self.project()

    def test_incomplete_or_invalid_descriptors_are_rejected(self):
        for descriptor in ((20, 0, 2), (2, 2, 2), (-1, 0, 2), (True, 0, 2)):
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                self.remote_layout = {"main:2": descriptor}
                self.project()

    def test_each_parallel_address_array_is_checked(self):
        for position in range(4):
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "Incomplete"):
                arrays = list(self.remote_arrays)
                arrays[position] = [[]]
                project_dsa_remote_arrays(self.local, self.keys, self.remote_layout, tuple(arrays), num_target_layers=2)

    def test_old_peer_is_rejected_for_mtp(self):
        self.remote_layout = None
        with self.assertRaisesRegex(ValueError, "update both P and D"):
            self.project()

    def test_old_peer_keeps_target_only_positional_path(self):
        self.remote_layout = None
        self.keys = {1: "main:1"}
        self.assertIs(self.project(), self.remote_arrays)


class TestDsaMTPReceive(unittest.TestCase):
    def setUp(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.tp_rank = 0
        thread.num_layers = 1
        thread._dsa_main_owner = True
        thread._dsa_indexer_local_layout = [[(0, 100, 8, 8, 1)], [], [(0, 200, 8, 8, 1)]]
        thread._dsa_main_local_layout = [
            [(0, 300, 8, 8, 1), (1, 400, 8, 8, 1)],
            [(0, 500, 8, 8, 1), (1, 600, 8, 8, 1)],
            [],
        ]
        thread._dsa_indexer_cache_keys = {0: "indexer:0", 2: "indexer:1"}
        thread._dsa_main_cache_keys = {0: "main:0", 1: "main:1"}
        thread.remote_metadata_lock = threading.Lock()
        # P uses rows 0/1/4; D uses rows 0/1/2. The first P row is colocated.
        thread.kv_caches_base_addr = {"P": {5000: [[1000, 2000, 3000], [4000, 5000], [], [], [6000]]}}
        thread.remote_block_stride_per_addr = {"P": {5000: [[8, 8, 8], [8, 8], [], [], [8]]}}
        thread.remote_block_len_per_addr = {"P": {5000: [[8, 8, 8], [8, 8], [], [], [8]]}}
        thread.remote_block_size_scale = {"P": {5000: [[1, 1, 1], [1, 1], [], [], [1]]}}
        thread.remote_metadata_hosts = {"P": {5000: "host"}}
        thread.remote_te_port = {"P": {5000: 6000}}
        thread.remote_dsa_cache_layout = {
            "P": {5000: {"main:0": (0, 0, 2), "indexer:0": (0, 2, 1), "main:1": (1, 0, 2), "indexer:1": (4, 0, 1)}}
        }
        thread._get_remote_metadata = Mock()
        thread._send_done_recv_signal = Mock()
        thread._log_dsa_transfer_phase_diag = Mock()
        thread._log_dsa_destination_checksums = Mock()
        thread.engine = SimpleNamespace(batch_transfer_sync_read=Mock(return_value=0))
        self.thread = thread
        self.endpoint = RemoteEndpoint("host", 5000, "P")
        self.command = DsaStepRequest("req", RemoteSource("remote-req", (self.endpoint,), (1,), (1,)), (2,), (2,))
        self.results = []

    def receive(self):
        self.thread._execute_dsa_receive(self.command, self.endpoint, self.results.append)

    def test_all_mtp_components_are_read_before_complete(self):
        def read(*_args):
            self.assertEqual(self.results, [])
            return 0

        self.thread.engine.batch_transfer_sync_read.side_effect = read
        self.receive()
        calls = self.thread.engine.batch_transfer_sync_read.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, ("host:6000", [116, 216], [3008, 6008], [8, 8]))
        self.assertEqual(calls[1].args, ("host:6000", [316, 416, 516, 616], [1008, 2008, 4008, 5008], [8] * 4))
        self.assertEqual(self.results[0].kind, DsaLocalResultKind.RECEIVE_COMPLETE)
        self.thread._send_done_recv_signal.assert_called_once()

    def test_missing_mtp_is_detected_before_any_read(self):
        del self.thread.remote_dsa_cache_layout["P"][5000]["main:1"]
        with self.assertRaisesRegex(ValueError, "missing required cache main:1"):
            self.receive()
        self.thread.engine.batch_transfer_sync_read.assert_not_called()
        self.assertEqual(self.results, [])
        self.thread._send_done_recv_signal.assert_called_once()

    def test_non_owner_only_reads_indexer(self):
        self.thread._dsa_main_owner = False
        self.thread._dsa_main_local_layout = [[], [], []]
        self.thread._dsa_main_cache_keys = {}
        self.receive()
        self.thread.engine.batch_transfer_sync_read.assert_called_once()
        self.assertEqual(self.results[0].kind, DsaLocalResultKind.RECEIVE_COMPLETE)

    def test_shared_target_indexer_does_not_hide_mtp_indexer(self):
        del self.thread.remote_dsa_cache_layout["P"][5000]["indexer:0"]
        self.receive()
        self.assertEqual(self.thread.engine.batch_transfer_sync_read.call_args_list[0].args[2], [6008])
        self.assertEqual(self.results[0].kind, DsaLocalResultKind.RECEIVE_COMPLETE)

    def test_transfer_failure_never_reports_receive_complete(self):
        for returns in ((-1,), (0, -1)):
            with self.subTest(returns=returns):
                self.setUp()
                self.thread.engine.batch_transfer_sync_read.side_effect = returns
                self.receive()
                self.assertEqual(self.results[0].kind, DsaLocalResultKind.TRANSFER_FAILED)
                self.thread._send_done_recv_signal.assert_called_once()

    def test_indexer_source_pages_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "cannot cover all source pages"):
            self.thread._build_dsa_transfer_lists(
                [[(0, 100, 16, 16, 1)]], [[1000]], [[8]], [[1]], [[8]], (0, 1, 2), (0,)
            )

    def test_partial_final_indexer_row_is_allowed(self):
        result = self.thread._build_dsa_transfer_lists(
            [[(0, 100, 16, 16, 1)]], [[1000]], [[8]], [[1]], [[8]], (0, 1, 2), (0, 1)
        )
        self.assertEqual(result, ([100, 108, 116], [1000, 1008, 1016], [8, 8, 8]))

    def test_missing_main_array_entry_is_not_skipped(self):
        with self.assertRaisesRegex(ValueError, "incomplete DSA positional"):
            self.thread._build_dsa_transfer_lists([[(0, 100, 8, 8, 1)]], [[]], [[]], [[]], [[]], (0,), (0,))


class TestDsaHandshakeCompatibility(unittest.TestCase):
    def test_legacy_reader_accepts_extended_handshake(self):
        fields = []
        for field in msgspec.structs.fields(MooncakeAgentMetadata):
            if field.name == "dsa_cache_layout":
                continue
            fields.append(
                (field.name, field.type)
                if field.default is msgspec.NODEFAULT
                else (field.name, field.type, field.default)
            )
        legacy_type = msgspec.defstruct("LegacyMooncakeAgentMetadata", fields)
        metadata = MooncakeAgentMetadata(
            engine_id="P",
            te_rpc_port=6000,
            kv_group2layeridx={},
            block_size=128,
            kv_caches_base_addr=[[1000, 2000]],
            block_size_scale=[[1, 1]],
            num_blocks=4,
            block_lens=[[8, 8]],
            block_strides=[[8, 8]],
            dsa_cache_layout={"main:1": (0, 0, 2)},
        )
        legacy = msgspec.msgpack.decode(msgspec.msgpack.encode(metadata), type=legacy_type)
        self.assertEqual(legacy.kv_caches_base_addr, metadata.kv_caches_base_addr)

    def test_optional_descriptor_preserves_existing_address_arrays(self):
        fields = dict(
            engine_id="P",
            te_rpc_port=6000,
            kv_group2layeridx={},
            block_size=128,
            kv_caches_base_addr=[[1000, 2000]],
            block_size_scale=[[1, 1]],
            num_blocks=4,
            block_lens=[[8, 8]],
            block_strides=[[8, 8]],
        )
        old_payload = msgspec.msgpack.encode(fields)
        old_meta = msgspec.msgpack.decode(old_payload, type=MooncakeAgentMetadata)
        self.assertIsNone(old_meta.dsa_cache_layout)
        new_meta = MooncakeAgentMetadata(**fields, dsa_cache_layout={"main:1": (0, 0, 2)})
        round_trip = msgspec.msgpack.decode(msgspec.msgpack.encode(new_meta), type=MooncakeAgentMetadata)
        self.assertEqual(round_trip.dsa_cache_layout, {"main:1": (0, 0, 2)})
        self.assertEqual(round_trip.kv_caches_base_addr, old_meta.kv_caches_base_addr)


if __name__ == "__main__":
    unittest.main()
