#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py
#
import copy
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.kv_offload.utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaLocalResult,
    DsaLocalResultKind,
    DsaTransferPhase,
    DsaWorkerResultMetadata,
)


def _num_waiting_requests(scheduler) -> int:
    return len(scheduler.waiting) + len(scheduler.skipped_waiting)


def _create_blockwise_dsa_scheduler(*, num_blocks=8, policy="fcfs"):
    extra = {
        "dsa_pd_offload": True,
        "prefill": {"tp_size": 2, "dp_size": 1, "pp_size": 1},
        "decode": {"tp_size": 2, "dp_size": 1, "pp_size": 1},
        "sfa_kv_offload_backend": "mooncake",
    }
    config = create_vllm_config(
        max_num_batched_tokens=16, block_size=128, kv_role="kv_consumer", kv_connector_extra_config=extra
    )
    config.scheduler_config.async_scheduling = False
    config.scheduler_config.policy = policy
    config.model_config.max_model_len = 16
    config.parallel_config.tensor_parallel_size = 2
    ascend_config = MagicMock(use_offload=True, kv_offload_mode="fused_overlap")

    def create_connector(*, config, role, kv_cache_config):
        assert role is KVConnectorRole.SCHEDULER
        return MooncakeConnector(config, role, kv_cache_config)

    with (
        patch.object(KVConnectorFactory, "create_connector", side_effect=create_connector),
        patch.multiple(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
            init_ascend_config=MagicMock(),
            get_ascend_config=MagicMock(return_value=ascend_config),
        ),
        patch.multiple(
            "vllm_ascend.distributed.kv_transfer.sfa_pd_cpu_offload.scheduler",
            create=True,
            init_ascend_config=MagicMock(),
            get_ascend_config=MagicMock(return_value=ascend_config),
            get_ip=MagicMock(return_value="127.0.0.1"),
        ),
    ):
        return create_scheduler(config, num_blocks=num_blocks, group_block_sizes=(2, 128))


def _dsa_result(command, rank, kind, *, failure_phase=None):
    return DsaLocalResult(
        command.request_id,
        rank,
        kind,
        failure_phase,
    )


def _update_scheduler(scheduler, output, *results):
    scheduled = [
        scheduler.requests[request_id]
        for request_id in output.num_scheduled_tokens
    ]
    runner_output = (
        create_model_runner_output(scheduled)
        if scheduled
        else copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    )
    if results:
        runner_output.kv_connector_output = KVConnectorOutput(
            kv_connector_worker_meta=DsaWorkerResultMetadata(
                tuple(results)
            )
        )
    scheduler.update_from_output(output, runner_output)


def test_basic_lifecycle():
    """Test lifecycle of a remote prefill."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks

    request = create_request(request_id=1, num_tokens=NUM_TOKENS, do_remote_prefill=True, block_size=BLOCK_SIZE)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()

    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0

    assert _num_waiting_requests(scheduler) == 1
    assert request in scheduler.skipped_waiting
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == NUM_TOKENS

    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.free_block_queue.num_free_blocks < START_FREE_BLOCK_QUEUE_SIZE
    assert len(block_pool.cached_block_hash_to_block) == 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    for block in blocks:
        assert block._block_hash is None

    # (1b): forward()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    assert not engine_core_outputs or not engine_core_outputs[0].outputs

    # STEP (2):
    # (2a): schedule(): nothing happens!
    scheduler_output = scheduler.schedule()
    assert _num_waiting_requests(scheduler) == 1
    assert len(scheduler.running) == 0

    # (2b): forward(): request finishes recv.
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(finished_recving={request_id})

    # (2c): update_from_output():
    engine_core_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
    assert _num_waiting_requests(scheduler) == 1
    assert request_id in scheduler.finished_recving_kv_req_ids

    # STEP (3):
    # (3a): schedule(): this should actually schedule.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    num_hashed_blocks = 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += 1 if block._block_hash is not None else 0
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS

    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
    num_computed_tokens = scheduled_req.num_computed_tokens
    total_prompt_tokens = len(scheduled_req.prompt_token_ids)
    assert num_scheduled_tokens == total_prompt_tokens - num_computed_tokens

    # (3b): execute_model()
    model_runner_output = create_model_runner_output([request])
    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    scheduler.update_from_output(scheduler_output, model_runner_output)
    scheduler.schedule()

    assert_scheduler_empty(scheduler)


def test_blockwise_dsa_is_one_shot_and_waits_for_all_tp_results():
    scheduler = _create_blockwise_dsa_scheduler()
    request = create_request(
        1,
        num_tokens=5,
        max_tokens=4,
        do_remote_prefill=True,
        block_size=128,
    )
    request.kv_transfer_params.update(
        remote_request_id="prefill-id-1",
        remote_block_ids=((101, 102, 103), (201,)),
    )
    scheduler.add_request(request)

    initial_output = scheduler.schedule()
    command = initial_output.kv_connector_metadata.requests[0]
    assert command.source.indexer_block_ids == (101, 102, 103)
    assert command.source.main_block_ids == (201,)
    assert command.indexer_hbm_block_ids
    assert command.main_host_block_ids
    assert not hasattr(initial_output.kv_connector_metadata, "d2h_plans")

    _update_scheduler(
        scheduler,
        initial_output,
        _dsa_result(
            command,
            0,
            DsaLocalResultKind.RECEIVE_COMPLETE,
        ),
    )
    partial_output = scheduler.schedule()
    assert partial_output.kv_connector_metadata.requests == ()

    _update_scheduler(
        scheduler,
        partial_output,
        _dsa_result(
            command,
            1,
            DsaLocalResultKind.RECEIVE_COMPLETE,
        ),
    )
    ready_output = scheduler.schedule()
    assert request.request_id in ready_output.num_scheduled_tokens


def test_blockwise_dsa_transfer_failure_falls_back_to_local_recompute():
    scheduler = _create_blockwise_dsa_scheduler()
    request = create_request(
        1,
        num_tokens=5,
        max_tokens=4,
        do_remote_prefill=True,
        block_size=128,
    )
    request.kv_transfer_params.update(
        remote_request_id="prefill-id-1",
        remote_block_ids=((101, 102, 103), (201,)),
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    command = output.kv_connector_metadata.requests[0]
    _update_scheduler(
        scheduler,
        output,
        _dsa_result(
            command,
            0,
            DsaLocalResultKind.TRANSFER_FAILED,
            failure_phase=DsaTransferPhase.INDEXER_D2D,
        ),
        _dsa_result(
            command,
            1,
            DsaLocalResultKind.RECEIVE_COMPLETE,
        ),
    )
    recompute_output = scheduler.schedule()
    assert recompute_output.num_scheduled_tokens[request.request_id] == len(
        request.prompt_token_ids
    )
