"""Shared-proposal query x release factorial replay primitives."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple


FACTORIAL_REPLAY_VERSION = "rgd_query_release_factorial_v5"
FACTORIAL_RUN_SCHEMA = "rgd_query_release_factorial_run_v3"
FACTORIAL_PROPOSAL_SCHEMA = "rgd_factorial_proposal_bank_v3"
FACTORIAL_EVENT_SCHEMA = "rgd_event_log_v3"


@dataclass(frozen=True)
class FactorialArm:
    name: str
    query_gate_enabled: bool
    release_guard_enabled: bool


@dataclass(frozen=True)
class QueryAdmissionContext:
    """Proposal-blind inputs available before a slow request is issued."""

    frame: int
    fast_action: int
    query_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class QueryAdmissionDecision:
    admit: bool
    audit: Mapping[str, Any]


class QueryAdmissionPolicy(Protocol):
    def decide(self, context: QueryAdmissionContext) -> QueryAdmissionDecision:
        ...


_COMPONENT_GATE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("latency_survival", "latency_survival_pass"),
    ("maneuver_breadth", "maneuver_breadth_pass"),
    ("corrective_headroom", "corrective_headroom_pass"),
    ("state_need", "state_need_pass"),
)
_NON_ABLATABLE_GATE_FIELDS: Tuple[str, ...] = (
    "domain_contract_pass",
    "executor_available_pass",
    "latency_prediction_pass",
    "absolute_feasibility_pass",
)


@dataclass(frozen=True)
class ComponentAblationArm:
    """One prespecified query-gate component removal for the v13 study."""

    name: str
    display_name: str
    removed_components: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {name for name, _ in _COMPONENT_GATE_FIELDS}
        removed = tuple(str(item) for item in self.removed_components)
        if not str(self.name or "") or not str(self.display_name or ""):
            raise ValueError("component-ablation arm names must be nonempty")
        if len(removed) != len(set(removed)) or not set(removed).issubset(allowed):
            raise ValueError("component-ablation arm removes an unknown component")


# The leave-one-out arms identify all four serial predicates. The H x N arm is
# a prespecified interaction check because both predicates govern whether a
# potentially corrective maneuver is worth escalating.
COMPONENT_ABLATION_ARMS: Tuple[ComponentAblationArm, ...] = (
    ComponentAblationArm("full", "Full RGD"),
    ComponentAblationArm("without_l", "w/o L", ("latency_survival",)),
    ComponentAblationArm("without_a", "w/o A", ("maneuver_breadth",)),
    ComponentAblationArm("without_h", "w/o H", ("corrective_headroom",)),
    ComponentAblationArm("without_n", "w/o N", ("state_need",)),
    ComponentAblationArm(
        "without_h_and_n",
        "w/o H,N",
        ("corrective_headroom", "state_need"),
    ),
)


class ComponentAblationQueryPolicy:
    """Remove only named serial predicates from a frozen query gate."""

    def __init__(self, arm: ComponentAblationArm) -> None:
        self.arm = arm

    def decide(self, context: QueryAdmissionContext) -> QueryAdmissionDecision:
        gate = context.query_metadata.get("recoverability_gate")
        if not isinstance(gate, Mapping):
            raise ValueError("component ablation requires recoverability-gate metadata")
        values = dict(gate)
        required = [
            *_NON_ABLATABLE_GATE_FIELDS,
            *(field for _, field in _COMPONENT_GATE_FIELDS),
            "serial_gate_pass",
        ]
        missing = [field for field in required if not isinstance(values.get(field), bool)]
        if missing:
            raise ValueError(
                "component ablation requires boolean gate fields: "
                + ", ".join(sorted(missing))
            )

        non_ablatable_pass = all(bool(values[field]) for field in _NON_ABLATABLE_GATE_FIELDS)
        component_passes = {
            name: bool(values[field]) for name, field in _COMPONENT_GATE_FIELDS
        }
        retained_components = tuple(
            name for name, _ in _COMPONENT_GATE_FIELDS if name not in self.arm.removed_components
        )
        admit = bool(
            non_ablatable_pass
            and all(component_passes[name] for name in retained_components)
        )
        full_equivalent = bool(admit == bool(values["serial_gate_pass"]))
        if not self.arm.removed_components and not full_equivalent:
            raise ValueError("full component-ablation arm diverges from the runtime serial gate")

        audit: Dict[str, Any] = {
            "component_ablation_policy": "serial_predicate_removal",
            "component_ablation_arm": self.arm.name,
            "component_ablation_removed_components": ";".join(self.arm.removed_components),
            "component_ablation_non_ablatable_pass": bool(non_ablatable_pass),
            "component_ablation_full_equivalent_to_serial_gate": bool(full_equivalent),
        }
        for name, passed in component_passes.items():
            audit[f"component_ablation_{name}_pass"] = bool(passed)
            audit[f"component_ablation_{name}_retained"] = bool(
                name in retained_components
            )
        return QueryAdmissionDecision(admit=admit, audit=audit)


FACTORIAL_ARMS: Tuple[FactorialArm, ...] = (
    FactorialArm("full", True, True),
    FactorialArm("query_only", True, False),
    FactorialArm("release_only", False, True),
    FactorialArm("neither", False, False),
)

# ``neither`` is the no-gate control within the 2x2 query/release factorial:
# it still executes every frozen proposal.  The formal comparison additionally
# needs a genuine Fast-only arm that never issues a slow request.  Keep the
# legacy four-arm tuple stable for existing artifacts and expose the explicit
# five-arm design separately.
FAST_ONLY_ARM = FactorialArm("fast_only", False, False)
FORMAL_FACTORIAL_ARMS: Tuple[FactorialArm, ...] = FACTORIAL_ARMS + (FAST_ONLY_ARM,)


@dataclass(frozen=True)
class ProposalRecord:
    seed: int
    source_frame: int
    request_id: str
    raw_slow_action: int
    latency_steps: int
    outcome: str = "valid"
    response_text: str = ""
    response_sha256: str = ""
    source_artifact: str = ""

    def __post_init__(self) -> None:
        if int(self.seed) < 0 or int(self.source_frame) < 0:
            raise ValueError("proposal seed and source frame must be nonnegative")
        if not str(self.request_id or ""):
            raise ValueError("proposal request ID must be nonempty")
        if int(self.raw_slow_action) not in range(5):
            raise ValueError("proposal action must be in the discrete action universe")
        if int(self.latency_steps) < 0:
            raise ValueError("proposal latency must be nonnegative")
        if self.outcome not in {"valid", "timeout", "failure"}:
            raise ValueError(f"unsupported proposal outcome: {self.outcome}")
        if self.response_sha256 and len(str(self.response_sha256)) != 64:
            raise ValueError("proposal response SHA256 must be a full digest")


def canonical_proposal_bank_payload(
    records_by_seed: Mapping[int, Mapping[int, ProposalRecord]],
) -> list[Dict[str, Any]]:
    """Return the portable bank identity, retaining empty seed blocks."""
    payload = []
    for raw_seed in sorted(records_by_seed):
        seed = int(raw_seed)
        records = []
        for frame, record in sorted(records_by_seed[raw_seed].items()):
            if int(frame) != int(record.source_frame) or seed != int(record.seed):
                raise ValueError("proposal-bank map keys do not match record identity")
            row = asdict(record)
            # Provenance paths are authenticated separately by content hash in
            # the manifest and must not make the bank identity machine-specific.
            row.pop("source_artifact", None)
            records.append(row)
        payload.append({"seed": seed, "records": records})
    return payload


def proposal_bank_sha256(
    records_by_seed: Mapping[int, Mapping[int, ProposalRecord]],
) -> str:
    payload = canonical_proposal_bank_payload(records_by_seed)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_factorial_arm(
    cfg: Mapping[str, Any],
    arm: FactorialArm,
) -> Dict[str, Any]:
    """Return one arm config while holding controller and safety settings fixed."""
    resolved = copy.deepcopy(dict(cfg))
    replay = copy.deepcopy(
        dict(resolved.get("closed_loop_latency_replay", {}) or {})
    )
    replay["enable"] = True
    replay["target_systems"] = ["slow"]
    revalidation = copy.deepcopy(
        dict(replay.get("release_opportunity_revalidation", {}) or {})
    )
    revalidation["enable"] = bool(arm.release_guard_enabled)
    alignment = copy.deepcopy(
        dict(revalidation.get("action_cost_alignment", {}) or {})
    )
    alignment["enable"] = bool(arm.release_guard_enabled)
    revalidation["action_cost_alignment"] = alignment
    replay["release_opportunity_revalidation"] = revalidation
    resolved["closed_loop_latency_replay"] = replay
    resolved["capture_release_snapshots_online"] = True
    resolved["require_release_snapshot_on_release"] = True
    resolved["factorial_replay_version"] = FACTORIAL_REPLAY_VERSION
    resolved["factorial_arm"] = asdict(arm)
    return resolved


def _gate_pass(metadata: Mapping[str, Any]) -> bool:
    gate = dict(metadata.get("recoverability_gate", {}) or {})
    return bool(gate.get("serial_gate_pass", False))


def _query_state_action_mapping(
    state: Any,
    *,
    raw_slow_action: int,
    fast_action: int,
) -> tuple[int, bool, str]:
    """Fail closed when a logged proposal is illegal on an arm trajectory."""
    raw_available = getattr(state, "legal_actions", None)
    if raw_available is None:
        getter = getattr(state, "get_available_actions", None)
        raw_available = getter() if callable(getter) else None
    if raw_available is None and isinstance(state, Mapping):
        raw_available = state.get("available_actions")
    if raw_available is None:
        return int(fast_action), False, "query_action_universe_missing"
    available = {int(action) for action in list(raw_available)}
    if int(raw_slow_action) in available:
        return int(raw_slow_action), True, "raw_action_available"
    if int(fast_action) not in available:
        raise RuntimeError("factorial Fast fallback is absent from the query action set")
    return int(fast_action), False, "raw_action_unavailable_fast_fallback"


class ProposalReplayAgent:
    """Replay a frozen response stream over a continuously recomputed Fast policy."""

    def __init__(
        self,
        inner: Any,
        proposals_by_frame: Mapping[int, ProposalRecord],
        *,
        arm: FactorialArm,
        bank_sha256: str,
        query_admission_policy: Optional[QueryAdmissionPolicy] = None,
    ) -> None:
        self.inner = inner
        self.proposals_by_frame = {
            int(frame): record for frame, record in proposals_by_frame.items()
        }
        if set(self.proposals_by_frame) != {
            int(record.source_frame) for record in self.proposals_by_frame.values()
        }:
            raise ValueError("proposal-bank frame keys do not match record frames")
        request_ids = [
            str(record.request_id or "")
            for record in self.proposals_by_frame.values()
        ]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("proposal bank contains duplicate request IDs")
        self.arm = arm
        self.bank_sha256 = str(bank_sha256)
        self.query_admission_policy = query_admission_policy
        self.frame = 0
        self.last_system_used = "fast"
        self.candidate_count = 0
        self.issued_count = 0
        self.gate_rejected_count = 0
        self.timeout_count = 0
        self._diagnostic_watchdogs: Dict[str, int] = {}

    @property
    def fast_thinker(self):
        return self.inner.fast_thinker

    @property
    def orchestrator(self):
        return self.inner.orchestrator

    def snapshot_policy_state(self) -> Dict[str, Any]:
        return self.inner.snapshot_policy_state()

    def snapshot_release_policy_state(self) -> Dict[str, Any]:
        snapshot = getattr(self.inner, "snapshot_release_policy_state", None)
        return snapshot() if callable(snapshot) else self.inner.snapshot_policy_state()

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        self.inner.restore_policy_state(snapshot)

    def record_executed_action(self, action: int) -> None:
        self.inner.record_executed_action(int(action))

    def uses_native_async_policy_pacing(self) -> bool:
        """Factorial response timing is simulated in frames, never wall time."""
        return False

    def prepare_frame(self, frame: int) -> Optional[Dict[str, Any]]:
        prepare = getattr(self.inner, "prepare_frame", None)
        return prepare(int(frame)) if callable(prepare) else None

    def pending_slow_requests(self) -> list[Dict[str, Any]]:
        pending = getattr(self.inner, "pending_slow_requests", None)
        return list(pending() or []) if callable(pending) else []

    def end_episode(self, reason: str = "episode_end") -> list[Dict[str, Any]]:
        end = getattr(self.inner, "end_episode", None)
        return list(end(str(reason)) or []) if callable(end) else []

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def decide(self, state: Any) -> Tuple[int, str, Dict[str, Any]]:
        frame = int(self.frame)
        # This counter is a local liveness diagnostic only.  The authoritative
        # request terminal event is emitted by the closed-loop replay queue in
        # runtime_support; no controller action is changed here.
        due_request_ids = [
            request_id
            for request_id, due_frame in self._diagnostic_watchdogs.items()
            if int(due_frame) <= frame
        ]
        for request_id in due_request_ids:
            self._diagnostic_watchdogs.pop(request_id, None)
        self.timeout_count += len(due_request_ids)
        fast_action, fast_response, raw_metadata = self.inner.decide(state)
        metadata = dict(raw_metadata or {})
        metadata.update(
            {
                "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
                "factorial_arm": self.arm.name,
                "factorial_query_gate_enabled": bool(
                    self.arm.query_gate_enabled
                ),
                "factorial_release_guard_enabled": bool(
                    self.arm.release_guard_enabled
                ),
                "factorial_proposal_bank_sha256": self.bank_sha256,
                "factorial_candidate_query": False,
                "factorial_query_issued": False,
                "closed_loop_latency_issuance_event": False,
                "closed_loop_latency_issued_request_id": "",
                "closed_loop_latency_issued_response_outcome": "",
                "closed_loop_latency_terminal_event": False,
                "closed_loop_latency_terminal_request_id": "",
                "closed_loop_latency_terminal_response_outcome": "",
            }
        )
        proposal = self.proposals_by_frame.get(frame)
        self.frame += 1
        if proposal is None:
            self.last_system_used = "fast"
            return int(fast_action), str(fast_response), metadata

        self.candidate_count += 1
        admission_audit: Dict[str, Any] = {}
        if self.query_admission_policy is None:
            gate_passed = _gate_pass(metadata)
        else:
            # The learned policy runs before any ProposalRecord field is added.
            # A shallow immutable copy is sufficient because policy inputs are
            # read-only by contract and the deployed extractor uses a whitelist.
            context = QueryAdmissionContext(
                frame=frame,
                fast_action=int(fast_action),
                query_metadata=MappingProxyType(dict(metadata)),
            )
            decision = self.query_admission_policy.decide(context)
            if not isinstance(decision, QueryAdmissionDecision):
                raise TypeError(
                    "query admission policy must return QueryAdmissionDecision"
                )
            gate_passed = bool(decision.admit)
            admission_audit = dict(decision.audit or {})
            if any(
                not str(key).startswith(
                    ("discrepancy_gate_", "component_ablation_")
                )
                for key in admission_audit
            ):
                raise ValueError(
                    "query admission audit keys must use an approved policy prefix"
                )
        metadata.update(
            {
                "factorial_candidate_query": True,
                "factorial_candidate_request_id": proposal.request_id,
                "factorial_query_gate_pass": bool(gate_passed),
                "factorial_shared_raw_slow_action": int(
                    proposal.raw_slow_action
                ),
                "factorial_shared_latency_steps": int(proposal.latency_steps),
                "factorial_shared_response_sha256": str(
                    proposal.response_sha256
                ),
                "factorial_shared_response_outcome": proposal.outcome,
            }
        )
        metadata.update(admission_audit)
        if self.arm.name == FAST_ONLY_ARM.name:
            self.gate_rejected_count += 1
            metadata["factorial_query_rejection_reason"] = "fast_only_control"
            self.last_system_used = "fast"
            return int(fast_action), str(fast_response), metadata

        if self.arm.query_gate_enabled and not gate_passed:
            self.gate_rejected_count += 1
            metadata["factorial_query_rejection_reason"] = "query_gate_failed"
            self.last_system_used = "fast"
            return int(fast_action), str(fast_response), metadata

        mapped_slow_action, raw_action_available, mapping_reason = (
            _query_state_action_mapping(
                state,
                raw_slow_action=int(proposal.raw_slow_action),
                fast_action=int(fast_action),
            )
        )

        record_external_request = getattr(
            self.orchestrator,
            "record_external_slow_request",
            None,
        )
        if not callable(record_external_request):
            raise RuntimeError(
                "factorial replay requires orchestrator.record_external_slow_request()"
        )
        record_external_request()
        self.issued_count += 1
        self._diagnostic_watchdogs[proposal.request_id] = frame + 2
        metadata.update(
            {
                "factorial_query_issued": True,
                "factorial_policy_state_synchronized": True,
                "factorial_query_action_available": bool(raw_action_available),
                "factorial_query_mapped_action": int(mapped_slow_action),
                "factorial_query_mapping_reason": str(mapping_reason),
                "closed_loop_latency_request_id": proposal.request_id,
                "closed_loop_latency_issuance_event": True,
                "closed_loop_latency_issued_request_id": proposal.request_id,
                "closed_loop_latency_issued_response_outcome": proposal.outcome,
                "closed_loop_latency_terminal_event": False,
                "closed_loop_latency_terminal_request_id": "",
                "closed_loop_latency_terminal_response_outcome": "",
                "closed_loop_latency_response_outcome": proposal.outcome,
                "closed_loop_latency_terminal_outcome": "pending",
                "closed_loop_latency_timeout_event": False,
                "closed_loop_latency_failure_event": False,
                "slow_request_id": proposal.request_id,
                "closed_loop_scripted_latency_steps": int(
                    proposal.latency_steps
                ),
                "query_state_fast_proposal_action": int(fast_action),
                "query_state_slow_pre_guard_action": int(
                    proposal.raw_slow_action
                ),
                "query_state_slow_released_action": int(
                    mapped_slow_action
                ),
                "query_state_route_divergence": bool(
                    int(mapped_slow_action) != int(fast_action)
                ),
                "slow_request_attempted": True,
            }
        )
        if proposal.outcome != "valid":
            metadata.update(
                {
                    "system_used": "slow",
                    "route_label": "factorial_shared_proposal_pending",
                    "slow_request_valid_return": False,
                    "slow_request_failed": False,
                    "slow_reasoning_success": False,
                    "slow_reasoning_failure_reason": "",
                }
            )
            self.last_system_used = "slow"
            return (
                int(fast_action),
                f"[proposal {proposal.outcome} pending]",
                metadata,
            )

        metadata.update(
            {
                "system_used": "slow",
                "route_label": "factorial_shared_proposal_replay",
                "slow_request_valid_return": True,
                "slow_request_failed": False,
                "slow_reasoning_success": True,
                "slow_reasoning_mode": "frozen_response_replay",
            }
        )
        self.last_system_used = "slow"
        response = proposal.response_text or f"[proposal {proposal.request_id}]"
        return int(mapped_slow_action), response, metadata
