# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed cross-process metadata for blockwise DSA Mooncake transfers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

MAX_TCP_PORT = 65535


class DsaAction(str, Enum):
    RECEIVE_REMOTE = "receive_remote"
    PREPARE_REPLAY = "prepare_replay"
    QUIESCE = "quiesce"


class DsaLocalResultKind(str, Enum):
    RECEIVE_COMPLETE = "receive_complete"
    REPLAY_READY = "replay_ready"
    TRANSFER_FAILED = "transfer_failed"


class DsaTransferPhase(str, Enum):
    INDEXER_D2D = "indexer_d2d"
    MAIN_D2RH = "main_d2rh"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _normalize_block_ids(name: str, values: object, *, allow_zero: bool = True) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    block_ids = tuple(values)
    for block_id in block_ids:
        _require_nonnegative_integer(name, block_id)
        if not allow_zero and block_id == 0:
            raise ValueError(f"{name} must not contain reserved block 0")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(f"{name} must not contain duplicates")
    return block_ids


@dataclass(frozen=True, slots=True)
class RemoteEndpoint:
    remote_host: str
    remote_port: int
    remote_engine_id: str

    def __post_init__(self) -> None:
        _require_nonempty_string("remote_host", self.remote_host)
        _require_nonnegative_integer("remote_port", self.remote_port)
        if self.remote_port == 0 or self.remote_port > MAX_TCP_PORT:
            raise ValueError("remote_port must be in range 1..65535")
        _require_nonempty_string("remote_engine_id", self.remote_engine_id)


@dataclass(frozen=True, slots=True)
class RemoteSource:
    remote_request_id: str
    endpoints_by_prefill_rank: tuple[RemoteEndpoint, ...]
    indexer_block_ids: tuple[int, ...]
    main_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string("remote_request_id", self.remote_request_id)
        if not isinstance(self.endpoints_by_prefill_rank, (tuple, list)):
            raise TypeError("endpoints_by_prefill_rank must be a tuple or list")
        endpoints = tuple(self.endpoints_by_prefill_rank)
        if not endpoints:
            raise ValueError("endpoints_by_prefill_rank must not be empty")
        if not all(isinstance(endpoint, RemoteEndpoint) for endpoint in endpoints):
            raise TypeError("endpoints_by_prefill_rank must contain RemoteEndpoint values")
        object.__setattr__(self, "endpoints_by_prefill_rank", endpoints)
        object.__setattr__(
            self,
            "indexer_block_ids",
            _normalize_block_ids("indexer_block_ids", self.indexer_block_ids),
        )
        object.__setattr__(
            self,
            "main_block_ids",
            _normalize_block_ids("main_block_ids", self.main_block_ids),
        )
        if not self.indexer_block_ids or not self.main_block_ids:
            raise ValueError("RemoteSource block lists must not be empty")


@dataclass(frozen=True, slots=True)
class DestinationOwnership:
    main_reservation_id: int
    main_reservation_block_count: int
    main_bound_host_block_ids: tuple[int, ...]
    indexer_hbm_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("main_reservation_id", self.main_reservation_id)
        _require_nonnegative_integer(
            "main_reservation_block_count",
            self.main_reservation_block_count,
        )
        if self.main_reservation_block_count == 0:
            raise ValueError("main_reservation_block_count must be positive")
        main_block_ids = _normalize_block_ids(
            "main_bound_host_block_ids",
            self.main_bound_host_block_ids,
            allow_zero=False,
        )
        if len(main_block_ids) > self.main_reservation_block_count:
            raise ValueError("Main bound prefix exceeds reservation capacity")
        object.__setattr__(self, "main_bound_host_block_ids", main_block_ids)
        object.__setattr__(
            self,
            "indexer_hbm_block_ids",
            _normalize_block_ids("indexer_hbm_block_ids", self.indexer_hbm_block_ids),
        )


@dataclass(frozen=True, slots=True)
class LifecycleCommand:
    execution_epoch: int
    command_seq: int
    action: DsaAction
    num_computed_tokens: int
    num_external_tokens: int
    preserved_main_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, DsaAction):
            raise TypeError("action must be DsaAction")
        for name in (
            "execution_epoch",
            "command_seq",
            "num_computed_tokens",
            "num_external_tokens",
            "preserved_main_tokens",
        ):
            _require_nonnegative_integer(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class DsaStepRequest:
    request_id: str
    source: RemoteSource | None
    destination: DestinationOwnership
    lifecycle: LifecycleCommand

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        if self.source is not None and not isinstance(self.source, RemoteSource):
            raise TypeError("source must be RemoteSource")
        if not isinstance(self.destination, DestinationOwnership):
            raise TypeError("destination must be DestinationOwnership")
        if not isinstance(self.lifecycle, LifecycleCommand):
            raise TypeError("lifecycle must be LifecycleCommand")
        if self.lifecycle.action is DsaAction.RECEIVE_REMOTE and self.source is None:
            raise ValueError("RECEIVE_REMOTE requires a remote source")
        if self.lifecycle.action is DsaAction.RECEIVE_REMOTE and (
            not self.destination.indexer_hbm_block_ids or not self.destination.main_bound_host_block_ids
        ):
            raise ValueError("RECEIVE_REMOTE requires nonempty destination block lists")
        if self.lifecycle.action is not DsaAction.RECEIVE_REMOTE and self.source is not None:
            raise ValueError(f"{self.lifecycle.action.name} must not carry a remote source")


def validate_bound_main_capacity(request: DsaStepRequest, main_block_size: int) -> None:
    _require_nonnegative_integer("main_block_size", main_block_size)
    if main_block_size == 0:
        raise ValueError("main_block_size must be positive")
    bound_capacity = len(request.destination.main_bound_host_block_ids) * main_block_size
    lifecycle = request.lifecycle
    if lifecycle.preserved_main_tokens > bound_capacity:
        raise ValueError("preserved_main_tokens exceeds bound Main capacity")


@dataclass(frozen=True, slots=True)
class DsaD2HStepPlan:
    request_id: str
    execution_epoch: int
    d2h_step_seq: int
    main_reservation_id: int
    main_reservation_block_count: int
    main_bound_host_block_ids: tuple[int, ...]
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        for name in (
            "execution_epoch",
            "d2h_step_seq",
            "main_reservation_id",
            "main_reservation_block_count",
            "token_start",
            "token_end",
        ):
            _require_nonnegative_integer(name, getattr(self, name))
        if self.main_reservation_block_count == 0:
            raise ValueError("main_reservation_block_count must be positive")
        block_ids = _normalize_block_ids(
            "main_bound_host_block_ids",
            self.main_bound_host_block_ids,
            allow_zero=False,
        )
        if len(block_ids) > self.main_reservation_block_count:
            raise ValueError("Main bound prefix exceeds reservation capacity")
        if self.token_end <= self.token_start:
            raise ValueError("D2H token range must be nonempty")
        object.__setattr__(self, "main_bound_host_block_ids", block_ids)

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.request_id, self.execution_epoch, self.d2h_step_seq


def validate_d2h_plan_capacity(plan: DsaD2HStepPlan, main_block_size: int) -> None:
    _require_nonnegative_integer("main_block_size", main_block_size)
    if main_block_size == 0:
        raise ValueError("main_block_size must be positive")
    if plan.token_end > len(plan.main_bound_host_block_ids) * main_block_size:
        raise ValueError("D2H range exceeds bound Main capacity")


@dataclass(frozen=True, slots=True)
class D2HStepProgress:
    request_id: str
    execution_epoch: int
    d2h_step_seq: int
    main_reservation_id: int
    token_start: int
    token_end: int
    tp_rank: int

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        for name in (
            "execution_epoch",
            "d2h_step_seq",
            "main_reservation_id",
            "token_start",
            "token_end",
            "tp_rank",
        ):
            _require_nonnegative_integer(name, getattr(self, name))
        if self.token_end <= self.token_start:
            raise ValueError("D2H progress token range must be nonempty")

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.request_id,
            self.execution_epoch,
            self.d2h_step_seq,
            self.tp_rank,
        )


@dataclass(frozen=True, slots=True)
class DsaConnectorMetadata(KVConnectorMetadata):
    """Immutable commands issued to Decode workers for one engine step."""

    requests: tuple[DsaStepRequest, ...] = ()
    d2h_plans: tuple[DsaD2HStepPlan, ...] = ()
    reservation_id_upper_bound: int = 0
    live_reservation_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            "reservation_id_upper_bound",
            self.reservation_id_upper_bound,
        )
        if not isinstance(self.live_reservation_ids, (tuple, list)):
            raise TypeError("live_reservation_ids must be a tuple or list")
        live_reservation_ids = tuple(self.live_reservation_ids)
        for reservation_id in live_reservation_ids:
            _require_nonnegative_integer("live_reservation_ids", reservation_id)
            if reservation_id >= self.reservation_id_upper_bound:
                raise ValueError("live reservation IDs must be below the upper bound")
        if live_reservation_ids != tuple(sorted(set(live_reservation_ids))):
            raise ValueError("live reservation IDs must be strictly increasing")

        requests_by_id: dict[str, DsaStepRequest] = {}
        for request in self.requests:
            if request.destination.main_reservation_id not in live_reservation_ids:
                raise ValueError(f"DSA command for {request.request_id!r} must reference a live reservation")
            existing = requests_by_id.get(request.request_id)
            if existing is not None and existing != request:
                raise ValueError(f"conflicting DSA step requests for {request.request_id!r}")
            requests_by_id[request.request_id] = request
        plans_by_id: dict[str, DsaD2HStepPlan] = {}
        for plan in self.d2h_plans:
            if not isinstance(plan, DsaD2HStepPlan):
                raise TypeError("d2h_plans must contain DsaD2HStepPlan values")
            if plan.main_reservation_id not in live_reservation_ids:
                raise ValueError(f"DSA D2H plan for {plan.request_id!r} must reference a live reservation")
            if plan.request_id in requests_by_id:
                raise ValueError(f"DSA request {plan.request_id!r} must not carry lifecycle and D2H metadata")
            existing = plans_by_id.get(plan.request_id)
            if existing is not None and existing != plan:
                raise ValueError(f"conflicting DSA D2H plans for {plan.request_id!r}")
            plans_by_id[plan.request_id] = plan
        object.__setattr__(
            self,
            "requests",
            tuple(requests_by_id[key] for key in sorted(requests_by_id)),
        )
        object.__setattr__(
            self,
            "d2h_plans",
            tuple(plans_by_id[key] for key in sorted(plans_by_id)),
        )
        object.__setattr__(self, "live_reservation_ids", live_reservation_ids)


@dataclass(frozen=True, slots=True)
class DsaLocalResult:
    request_id: str
    execution_epoch: int
    command_seq: int
    tp_rank: int
    kind: DsaLocalResultKind
    failure_phase: DsaTransferPhase | None = None
    skipped_d2h_bytes: int = 0

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        _require_nonnegative_integer("execution_epoch", self.execution_epoch)
        _require_nonnegative_integer("command_seq", self.command_seq)
        _require_nonnegative_integer("tp_rank", self.tp_rank)
        _require_nonnegative_integer("skipped_d2h_bytes", self.skipped_d2h_bytes)
        if not isinstance(self.kind, DsaLocalResultKind):
            raise TypeError("kind must be DsaLocalResultKind")
        if self.failure_phase is not None and not isinstance(self.failure_phase, DsaTransferPhase):
            raise TypeError("failure_phase must be DsaTransferPhase")
        if self.kind is DsaLocalResultKind.TRANSFER_FAILED and self.failure_phase is None:
            raise ValueError("TRANSFER_FAILED requires a failure phase")
        if self.kind is not DsaLocalResultKind.TRANSFER_FAILED and self.failure_phase is not None:
            raise ValueError(f"{self.kind.name} must not carry a failure phase")
        if self.kind is not DsaLocalResultKind.REPLAY_READY and self.skipped_d2h_bytes:
            raise ValueError(f"{self.kind.name} must not carry skipped D2H bytes")

    @property
    def identity(self) -> tuple[str, int, int, int]:
        return (
            self.request_id,
            self.execution_epoch,
            self.command_seq,
            self.tp_rank,
        )


_ACTION_RESULT_KINDS = MappingProxyType(
    {
        DsaAction.RECEIVE_REMOTE: frozenset(
            (
                DsaLocalResultKind.RECEIVE_COMPLETE,
                DsaLocalResultKind.TRANSFER_FAILED,
            )
        ),
        DsaAction.PREPARE_REPLAY: frozenset((DsaLocalResultKind.REPLAY_READY,)),
        DsaAction.QUIESCE: frozenset(),
    }
)


def validate_action_result(action: DsaAction, result: DsaLocalResult) -> None:
    if result.kind not in _ACTION_RESULT_KINDS[action]:
        raise ValueError(f"{result.kind.name} is not valid for action {action.name}")


def _merge_results(
    groups: Iterable[Iterable[DsaLocalResult]],
) -> tuple[DsaLocalResult, ...]:
    merged: dict[tuple[str, int, int, int], DsaLocalResult] = {}
    for results in groups:
        for result in results:
            if not isinstance(result, DsaLocalResult):
                raise TypeError("results must contain DsaLocalResult values")
            existing = merged.get(result.identity)
            if existing is not None and existing != result:
                raise ValueError(f"conflicting DSA local results for identity {result.identity}")
            merged[result.identity] = result
    return tuple(merged[identity] for identity in sorted(merged))


def _merge_d2h_progress(
    groups: Iterable[Iterable[D2HStepProgress]],
) -> tuple[D2HStepProgress, ...]:
    merged: dict[tuple[str, int, int, int], D2HStepProgress] = {}
    for progress_group in groups:
        for progress in progress_group:
            if not isinstance(progress, D2HStepProgress):
                raise TypeError("d2h_progress must contain D2HStepProgress values")
            existing = merged.get(progress.identity)
            if existing is not None and existing != progress:
                raise ValueError(f"conflicting D2H progress for identity {progress.identity}")
            merged[progress.identity] = progress
    return tuple(merged[identity] for identity in sorted(merged))


@dataclass(frozen=True, slots=True)
class DsaWorkerResultMetadata(KVConnectorWorkerMetadata):
    """Same-step rank-aware terminal facts returned by Decode workers."""

    results: tuple[DsaLocalResult, ...] = ()
    d2h_progress: tuple[D2HStepProgress, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", _merge_results((self.results,)))
        object.__setattr__(
            self,
            "d2h_progress",
            _merge_d2h_progress((self.d2h_progress,)),
        )

    def aggregate(self, other: KVConnectorWorkerMetadata) -> DsaWorkerResultMetadata:
        if not isinstance(other, DsaWorkerResultMetadata):
            raise TypeError("aggregate expects another DsaWorkerResultMetadata instance")
        return DsaWorkerResultMetadata(
            _merge_results((self.results, other.results)),
            _merge_d2h_progress((self.d2h_progress, other.d2h_progress)),
        )
