# SPDX-License-Identifier: Apache-2.0
"""Physical MTP layer registration and D2RH completion regression tests."""

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_to_dram_connector import (
    MooncakeToDramProducerWorker,
    transfer_layerwise_d2rh,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def _cache_names(mtp_prefix="model.layers.2.mtp_block", *, include_mtp=True):
    prefixes = ["model.layers.0", "model.layers.1"]
    if include_mtp:
        prefixes.append(mtp_prefix)
    # Deliberately put Indexer first: the fallback callback needs Main first.
    return {f"{prefix}.self_attn.{component}": () for prefix in prefixes for component in ("indexer.k_cache", "attn")}


def _producer(caches, *, producer=True):
    worker = MooncakeToDramProducerWorker.__new__(MooncakeToDramProducerWorker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=2)),
        kv_transfer_config=SimpleNamespace(is_kv_producer=producer),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    worker.total_layers = 2
    worker._layer_transfer_finished_events = [threading.Event() for _ in range(2)] if producer else None
    worker._layer_transfer_pending_events = [threading.Event() for _ in range(2)] if producer else None
    for event in worker._layer_transfer_finished_events or ():
        event.set()
    worker._init_registered_layerwise_layout(caches)
    return worker


class TestLayerwiseMTPRegistration(unittest.TestCase):
    def test_split_mtp_components_count_as_one_physical_layer(self):
        for prefix in ("model.layers.2", "model.layers.2.mtp_block", "mtp.0"):
            with self.subTest(prefix=prefix):
                caches = _cache_names(prefix)
                worker = _producer(caches)
                self.assertEqual(worker.total_layers, 3)
                self.assertEqual(set(worker.index_to_name), {0, 1, 2})
                self.assertEqual(len(worker.index_to_name[2]), 2)
                self.assertTrue(worker.index_to_name[2][0].endswith(".attn"))
                self.assertEqual({name for names in worker.index_to_name.values() for name in names}, set(caches))
                self.assertEqual(len(worker._layer_transfer_finished_events), 3)
                self.assertEqual(len(worker._layer_transfer_pending_events), 3)
                self.assertTrue(all(event.is_set() for event in worker._layer_transfer_finished_events))
                self.assertFalse(any(event.is_set() for event in worker._layer_transfer_pending_events))

    def test_target_only_layout_is_unchanged(self):
        worker = _producer(_cache_names(include_mtp=False))
        self.assertEqual(worker.total_layers, 2)
        self.assertEqual(len(worker._layer_transfer_finished_events), 2)

    def test_partial_target_registration_does_not_shrink_runner_bound(self):
        worker = _producer({"model.layers.0.self_attn.attn": ()})
        self.assertEqual(worker.total_layers, 2)

    def test_multiple_physical_draft_layers_are_not_collapsed(self):
        caches = _cache_names("mtp.0")
        caches.update({"mtp.1.self_attn.attn": (), "mtp.1.self_attn.indexer.k_cache": ()})
        worker = _producer(caches)
        self.assertEqual(worker.total_layers, 4)
        self.assertEqual(len(worker.index_to_name[3]), 2)

    def test_registration_preserves_existing_event_objects(self):
        worker = _producer(_cache_names(include_mtp=False))
        previous = list(worker._layer_transfer_finished_events)
        worker._init_registered_layerwise_layout(_cache_names())
        self.assertEqual(worker._layer_transfer_finished_events[:2], previous)

    def test_consumer_does_not_allocate_producer_events(self):
        worker = _producer(_cache_names(), producer=False)
        self.assertEqual(worker.total_layers, 3)
        self.assertIsNone(worker._layer_transfer_finished_events)

    def test_duplicate_main_is_rejected(self):
        caches = _cache_names()
        caches["model.layers.2.mtp_block.other_attn"] = ()
        with self.assertRaises(AssertionError):
            _producer(caches)

    def test_mtp_callback_is_queued_and_repeated_draft_is_not(self):
        caches = _cache_names()
        worker = _producer(caches)
        worker.current_layer = 0
        worker.pd_head_ratio = 1
        worker.enable_c8_quant = worker.enable_kv_quant = False
        worker.layer_metadata = {name: SimpleNamespace(tensor_group_idx=[0]) for name in caches}
        worker.update_decoder_info = lambda _req_id, req: req
        worker.kv_send_layer_thread = SimpleNamespace(
            send_queue=queue.Queue(),
            layer_transfer_pending_events=worker._layer_transfer_pending_events,
            layer_transfer_finished_events=worker._layer_transfer_finished_events,
        )
        metadata = SimpleNamespace(
            requests={"request": SimpleNamespace(local_block_ids=[[1]], chunk_finish=True)},
            send_task=SendTask(),
        )
        attn_metadata = SimpleNamespace(reshape_cache_event=Mock())
        for names in worker.index_to_name.values():
            worker.save_kv_layer(names[1], [], attn_metadata, metadata)
            worker.save_kv_layer(names[0], [], attn_metadata, metadata)
        mtp_name = worker.index_to_name[2][0]
        worker.save_kv_layer(mtp_name, [], attn_metadata, metadata)
        sent = list(worker.kv_send_layer_thread.send_queue.queue)
        self.assertEqual([task.layer_idx for task in sent], [0, 1, 2])
        self.assertEqual(sent[-1].layer_name, mtp_name)
        self.assertTrue(worker._layer_transfer_pending_events[2].is_set())


class TestAscendStoreMTPRegistration(unittest.TestCase):
    def test_reuse_plan_matches_runner_after_adding_mtp(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.hf_config = SimpleNamespace(num_hidden_layers=2)
        worker.num_layers = 2
        worker.num_kv_cache_groups = 1
        worker.kv_cache_config = None
        worker.use_gva_layerwise = True
        worker._extra_config = {"layerwise_num_shared_buffers": 1}
        worker._init_layerwise_config()
        self.assertEqual(worker.independent_layers, [0, 1])
        worker._refresh_registered_layer_count(_cache_names())
        self.assertEqual(worker.independent_layers, [0, 2])
        self.assertEqual(len(worker.layer_load_tasks), 3)

    def test_multi_group_registration_retains_mtp_in_both_groups(self):
        for prefix in ("model.layers.2", "model.layers.2.mtp_block", "mtp.0"):
            with self.subTest(prefix=prefix):
                caches = _cache_names(prefix)
                worker = KVPoolWorker.__new__(KVPoolWorker)
                worker.hf_config = SimpleNamespace(num_hidden_layers=2)
                worker.num_layers = 2
                worker.num_kv_cache_groups = 2
                groups = [
                    SimpleNamespace(layer_names=[name for name in caches if ("indexer" in name) == indexer])
                    for indexer in (False, True)
                ]
                worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
                worker.use_gva_layerwise = False
                worker._extra_config = {}
                worker._init_layerwise_config()
                self.assertNotIn(2, worker.physical_layer_to_group_layers)
                worker._refresh_registered_layer_count(caches)
                self.assertEqual(worker.num_layers, 3)
                self.assertEqual(worker.physical_layer_to_group_layers[2], [(0, 2), (1, 2)])
                self.assertEqual(len(worker.layer_load_tasks), 3)
                self.assertEqual(len(worker.layer_save_tasks), 3)
                worker.kv_caches = {name: (SimpleNamespace(data_ptr=lambda: 100),) for name in caches}
                worker._get_cache_block_metadata = Mock(return_value=(16, 16, 64, 1))
                for attr in (
                    "group_kv_caches_base_addr",
                    "group_block_len",
                    "group_block_stride",
                    "group_layer_offsets",
                    "group_num_layers",
                ):
                    setattr(worker, attr, {})
                for group_id, group in enumerate(groups):
                    worker._infer_cache_group_metadata(group_id, group.layer_names)
                self.assertEqual(worker.group_num_layers, {0: 3, 1: 3})
                self.assertEqual(worker.group_layer_offsets, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]})

    def test_unchanged_layout_does_not_reset_events_or_tasks(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.hf_config = SimpleNamespace(num_hidden_layers=2)
        worker.num_layers = 2
        worker.num_kv_cache_groups = 2
        worker._init_layerwise_config = Mock()
        worker._refresh_registered_layer_count(_cache_names(include_mtp=False))
        worker._init_layerwise_config.assert_not_called()

    def test_partial_multi_group_layout_keeps_target_layer_bound(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.hf_config = SimpleNamespace(num_hidden_layers=4)
        worker.num_layers = 4
        worker.num_kv_cache_groups = 2
        worker._init_layerwise_config = Mock()
        worker._refresh_registered_layer_count(_cache_names(include_mtp=False))
        self.assertEqual(worker.num_layers, 4)
        worker._init_layerwise_config.assert_not_called()


class TestLayerwiseD2RHCompletion(unittest.TestCase):
    def setUp(self):
        self.thread = SimpleNamespace(
            total_layers=3,
            layer_transfer_finished_events=[threading.Event() for _ in range(3)],
            layer_transfer_pending_events=[threading.Event() for _ in range(3)],
            failed_reqs=set(),
            callback_func=Mock(),
        )
        self.request = SimpleNamespace(chunk_finish=True)
        self.task = SimpleNamespace(layer_idx=2, wait_event=Mock(), send_request={"req": self.request})

    def assert_released(self):
        self.assertTrue(self.thread.layer_transfer_finished_events[self.task.layer_idx].is_set())
        self.assertFalse(self.thread.layer_transfer_pending_events[self.task.layer_idx].is_set())

    def test_done_waits_for_mtp_and_both_legs(self):
        def transfer(_task, **_kwargs):
            self.thread.callback_func.assert_not_called()

        self.task.layer_idx = 1
        transfer_layerwise_d2rh(self.thread, self.task, transfer)
        self.thread.callback_func.assert_not_called()
        self.task.layer_idx = 2
        transfer_leg = Mock(side_effect=transfer)
        transfer_layerwise_d2rh(self.thread, self.task, transfer_leg)
        self.assertEqual(transfer_leg.call_count, 2)
        self.thread.callback_func.assert_called_once_with("req", self.request, 0, trans_flag=True)
        self.assert_released()

    def test_transfer_exception_signals_failure_once_and_releases(self):
        for fail_in in ("reshape", "indexer", "main"):
            with self.subTest(fail_in=fail_in):
                self.setUp()
                if fail_in == "reshape":
                    self.task.wait_event.synchronize.side_effect = RuntimeError("reshape failed")
                transfer_leg = Mock(
                    side_effect=[RuntimeError("indexer failed")]
                    if fail_in == "indexer"
                    else [None, RuntimeError("main failed")]
                    if fail_in == "main"
                    else None
                )
                transfer_layerwise_d2rh(self.thread, self.task, transfer_leg)
                self.thread.callback_func.assert_called_once_with("req", self.request, 0, trans_flag=False)
                self.assert_released()

    def test_failure_on_target_layer_is_remembered_until_mtp(self):
        self.task.layer_idx = 0
        transfer_layerwise_d2rh(self.thread, self.task, Mock(side_effect=RuntimeError("failed")))
        self.thread.callback_func.assert_not_called()
        self.assertIn("req", self.thread.failed_reqs)
        self.assert_released()
        self.task.layer_idx = 2
        transfer_layerwise_d2rh(self.thread, self.task, Mock())
        self.thread.callback_func.assert_called_once_with("req", self.request, 0, trans_flag=False)
        self.assert_released()

    def test_callback_exception_still_releases_source(self):
        self.thread.callback_func.side_effect = RuntimeError("callback failed")
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            transfer_layerwise_d2rh(self.thread, self.task, Mock())
        self.assert_released()

    def test_empty_payload_rank_still_signals(self):
        transfer_layerwise_d2rh(self.thread, self.task, Mock())
        self.thread.callback_func.assert_called_once_with("req", self.request, 0, trans_flag=True)

    def test_empty_request_task_releases_event(self):
        self.task.send_request = {}
        transfer_layerwise_d2rh(self.thread, self.task, Mock())
        self.thread.callback_func.assert_not_called()
        self.assert_released()

    def test_partial_chunk_does_not_emit_terminal_signal(self):
        self.request.chunk_finish = False
        transfer_layerwise_d2rh(self.thread, self.task, Mock())
        self.thread.callback_func.assert_not_called()
        self.assert_released()


if __name__ == "__main__":
    unittest.main()
