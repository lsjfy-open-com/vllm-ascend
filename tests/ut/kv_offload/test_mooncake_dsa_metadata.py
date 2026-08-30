# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

import importlib.machinery
import os
import pickle
import sys
import types
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import pytest

fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
installed_fake_torch_npu = "torch_npu" not in sys.modules
if installed_fake_torch_npu:
    sys.modules["torch_npu"] = fake_torch_npu
try:
    with patch.dict(os.environ, {"VLLM_PLUGINS": ""}):
        from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_dsa_metadata as dsa
finally:
    if installed_fake_torch_npu:
        sys.modules.pop("torch_npu", None)


def _source() -> dsa.RemoteSource:
    endpoint = dsa.RemoteEndpoint("prefill.example", 9000, "prefill-engine")
    return dsa.RemoteSource("prefill-request", (endpoint,), (1, 2), (3, 4))


def _lifecycle(action=dsa.DsaAction.RECEIVE_REMOTE, **overrides) -> dsa.LifecycleCommand:
    external = 32 if action is dsa.DsaAction.RECEIVE_REMOTE else 0
    return replace(dsa.LifecycleCommand(2, 3, action, 0, external, 16), **overrides)


def _request(action=dsa.DsaAction.RECEIVE_REMOTE) -> dsa.DsaStepRequest:
    source = _source() if action is dsa.DsaAction.RECEIVE_REMOTE else None
    destination = dsa.DestinationOwnership(7, 4, (1, 2), (0, 1))
    return dsa.DsaStepRequest("decode-request", source, destination, _lifecycle(action))


def _result(tp_rank=0, **overrides) -> dsa.DsaLocalResult:
    result = dsa.DsaLocalResult("decode-request", 2, 3, tp_rank, dsa.DsaLocalResultKind.RECEIVE_COMPLETE)
    return replace(result, **overrides)


@pytest.mark.parametrize(
    ("action", "carry_source", "overrides"),
    [
        (dsa.DsaAction.RECEIVE_REMOTE, False, {}),
        (dsa.DsaAction.PREPARE_REPLAY, True, {}),
    ],
)
def test_invalid_request_contracts_fail_closed(action, carry_source, overrides):
    with pytest.raises((TypeError, ValueError)):
        replace(
            _request(action),
            source=_source() if carry_source else None,
            lifecycle=_lifecycle(action, **overrides),
        )


@pytest.mark.parametrize(
    ("kind", "phase", "action"),
    [
        (dsa.DsaLocalResultKind.TRANSFER_FAILED, None, dsa.DsaAction.RECEIVE_REMOTE),
        (dsa.DsaLocalResultKind.RECEIVE_COMPLETE, dsa.DsaTransferPhase.INDEXER_D2D, dsa.DsaAction.RECEIVE_REMOTE),
        (dsa.DsaLocalResultKind.RECEIVE_COMPLETE, None, dsa.DsaAction.PREPARE_REPLAY),
    ],
)
def test_invalid_result_contracts_fail_closed(kind, phase, action):
    with pytest.raises((TypeError, ValueError)):
        dsa.validate_action_result(action, _result(kind=kind, failure_phase=phase))


@pytest.mark.parametrize(
    "field",
    ("execution_epoch", "command_seq", "num_computed_tokens", "num_external_tokens") + ("preserved_main_tokens",),
)
def test_lifecycle_scalars_are_nonnegative_integers(field):
    with pytest.raises((TypeError, ValueError)):
        _lifecycle(**{field: -1})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_request(), request_id=""),
        lambda: _lifecycle(action="receive_remote"),
        lambda: _result(tp_rank=-1),
        lambda: _result(kind="receive_complete"),
        lambda: _result(kind=dsa.DsaLocalResultKind.TRANSFER_FAILED, failure_phase="indexer_d2d"),
        lambda: dsa.DsaWorkerResultMetadata((object(),)),
        lambda: dsa.DsaWorkerResultMetadata().aggregate(object()),
    ],
)
def test_identity_and_enum_types_fail_closed(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_block_id_ownership_is_immutable_and_validated():
    request = _request()
    assert pickle.loads(pickle.dumps(request)) == request
    with pytest.raises(FrozenInstanceError):
        request.request_id = "other"  # type: ignore[misc]
    source = dsa.RemoteSource("r", [dsa.RemoteEndpoint("host", 1, "p")], [1, 2], [3, 4])
    destination = dsa.DestinationOwnership(7, 2, [1, 2], [0, 1])
    assert source.indexer_block_ids == destination.main_bound_host_block_ids == (1, 2)
    for factory in (
        lambda: dsa.DestinationOwnership(-1, 2, (1,), ()),
        lambda: dsa.DestinationOwnership(7, 0, (), ()),
        lambda: dsa.DestinationOwnership(7, 1, (0,), ()),
        lambda: dsa.DestinationOwnership(7, 1, (1, 1), ()),
        lambda: dsa.DestinationOwnership(7, 1, (1, 2), ()),
        lambda: dsa.RemoteSource("r", (), (1,), (3,)),
        lambda: dsa.RemoteSource("r", (object(),), (1,), (3,)),
        lambda: dsa.RemoteSource("r", (dsa.RemoteEndpoint("host", 1, "p"),), (), (3,)),
        lambda: dsa.RemoteSource("r", (dsa.RemoteEndpoint("host", 1, "p"),), (-1,), (3,)),
        lambda: dsa.RemoteEndpoint("", 1, "p"),
        lambda: dsa.RemoteEndpoint("host", 0, "p"),
        lambda: dsa.RemoteEndpoint("host", 65536, "p"),
        lambda: dsa.RemoteEndpoint("host", True, "p"),
        lambda: dsa.RemoteEndpoint("host", 1, ""),
    ):
        with pytest.raises((TypeError, ValueError)):
            factory()


def test_bound_main_token_ranges_fail_closed():
    with pytest.raises(ValueError, match="bound Main capacity"):
        dsa.validate_bound_main_capacity(
            replace(_request(), lifecycle=_lifecycle(preserved_main_tokens=33)),
            16,
        )
    plan = dsa.DsaD2HStepPlan("decode-request", 2, 0, 7, 4, (1,), 16, 17)
    with pytest.raises(ValueError, match="bound Main capacity"):
        dsa.validate_d2h_plan_capacity(plan, 16)


def test_worker_result_metadata_deduplicates_and_fails_closed():
    result = _result()
    merged = dsa.DsaWorkerResultMetadata((_result(1),)).aggregate(dsa.DsaWorkerResultMetadata((result,)))
    assert set(merged.results) == {result, _result(1)}
    assert dsa.DsaWorkerResultMetadata((result, result)).results == (result,)
    with pytest.raises(ValueError):
        dsa.DsaWorkerResultMetadata((result, replace(result, kind=dsa.DsaLocalResultKind.REPLAY_READY)))


def test_connector_metadata_validates_live_reservation_snapshot():
    metadata = dsa.DsaConnectorMetadata((_request(),), (), 8, (7,))
    assert pickle.loads(pickle.dumps(metadata)) == metadata
    for upper_bound, live_ids in ((-1, ()), (3, (2, 1)), (3, (1, 1)), (3, (3,))):
        with pytest.raises((TypeError, ValueError)):
            dsa.DsaConnectorMetadata((), (), upper_bound, live_ids)
    with pytest.raises(ValueError, match="live reservation"):
        dsa.DsaConnectorMetadata((_request(),), (), 8, ())


def test_step_local_d2h_metadata_is_independent_and_immutable():
    plan = dsa.DsaD2HStepPlan(
        "decode-request",
        2,
        0,
        7,
        4,
        (1, 2),
        16,
        24,
    )
    progress = dsa.D2HStepProgress(
        "decode-request",
        2,
        0,
        7,
        16,
        24,
        1,
    )

    metadata = dsa.DsaConnectorMetadata((), (plan,), 8, (7,))
    worker_metadata = dsa.DsaWorkerResultMetadata((), (progress,))

    assert metadata.d2h_plans == (plan,)
    assert worker_metadata.d2h_progress == (progress,)
    assert not hasattr(dsa.DsaAction, "FUSED_D2H")
    assert not hasattr(dsa.DsaLocalResultKind, "D2H_COMPLETE")
    assert pickle.loads(pickle.dumps(metadata)) == metadata
    with pytest.raises(FrozenInstanceError):
        plan.token_end = 32  # type: ignore[misc]


def test_step_local_d2h_metadata_rejects_conflicts_and_invalid_ranges():
    plan = dsa.DsaD2HStepPlan("decode-request", 2, 0, 7, 4, (1, 2), 16, 24)
    progress = dsa.D2HStepProgress("decode-request", 2, 0, 7, 16, 24, 1)

    for factory in (
        lambda: dsa.DsaD2HStepPlan("decode-request", 2, 0, 7, 4, (1,), 16, 16),
        lambda: dsa.DsaD2HStepPlan("decode-request", 2, 0, 7, 1, (1, 2), 16, 24),
        lambda: dsa.D2HStepProgress("decode-request", 2, 0, 7, 24, 16, 1),
        lambda: dsa.DsaConnectorMetadata((_request(),), (plan,), 8, (7,)),
        lambda: dsa.DsaConnectorMetadata((), (plan, replace(plan, token_end=32)), 8, (7,)),
        lambda: dsa.DsaWorkerResultMetadata(
            (),
            (progress, replace(progress, token_end=32)),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            factory()


def test_replay_ready_validates_skipped_d2h_bytes():
    result = _result(
        kind=dsa.DsaLocalResultKind.REPLAY_READY,
        skipped_d2h_bytes=768,
    )
    assert result.skipped_d2h_bytes == 768
    for invalid in (
        lambda: replace(result, skipped_d2h_bytes=-1),
        lambda: _result(skipped_d2h_bytes=1),
    ):
        with pytest.raises((TypeError, ValueError)):
            invalid()
