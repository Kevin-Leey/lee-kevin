"""Closed-loop latency-sweep entry point and scope contract.

The runner distinguishes paper-facing protocol executions from small smoke
checks before a simulator job is launched.  Keeping this classification here
prevents a short diagnostic run from being written with a formal label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SMOKE_SCOPE = "smoke"
FORMAL_LATENCY_SWEEP_SCOPE = "formal_closed_loop_latency_sweep"
FORMAL_MECHANISM_TRACE_SCOPE = "formal_mechanism_trace_acquisition"

_FORMAL_LATENCIES = (0.0, 0.7, 1.7, 2.7)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["rgd_fixed_policy"])
    parser.add_argument("--latencies", nargs="+", type=float, default=list(_FORMAL_LATENCIES))
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--policy-frequency", type=float, default=10.0)
    parser.add_argument(
        "--scope",
        choices=(
            "auto",
            SMOKE_SCOPE,
            FORMAL_LATENCY_SWEEP_SCOPE,
            FORMAL_MECHANISM_TRACE_SCOPE,
        ),
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.seeds <= 0 or args.policy_frequency <= 0:
        parser.error("seeds and policy frequency must be positive")
    if args.scope != "auto" and args.scope != SMOKE_SCOPE:
        if args.scope == FORMAL_LATENCY_SWEEP_SCOPE and not _matches_formal_latency_sweep(args):
            parser.error("formal closed-loop latency sweep requires the locked protocol matrix")
        if args.scope == FORMAL_MECHANISM_TRACE_SCOPE and not _matches_formal_mechanism_trace(args):
            parser.error("formal mechanism trace acquisition requires the locked cohort")
    return args


def _same_latencies(values: Sequence[float], expected: Sequence[float]) -> bool:
    return len(values) == len(expected) and all(
        abs(float(actual) - float(target)) <= 1e-12
        for actual, target in zip(values, expected)
    )


def _matches_formal_latency_sweep(args: argparse.Namespace) -> bool:
    return (
        list(args.groups) == ["rgd_fixed_policy"]
        and _same_latencies(args.latencies, _FORMAL_LATENCIES)
        and int(args.seeds) == 30
        and abs(float(args.policy_frequency) - 10.0) <= 1e-12
    )


def _matches_formal_mechanism_trace(args: argparse.Namespace) -> bool:
    return (
        list(args.groups) == ["always_fast"]
        and _same_latencies(args.latencies, (0.0,))
        and int(args.seed_start) == 6000
        and int(args.seeds) == 20
        and abs(float(args.policy_frequency) - 10.0) <= 1e-12
    )


def _is_formal_closed_loop_latency_sweep(args: argparse.Namespace) -> bool:
    return _scope_label(args) == FORMAL_LATENCY_SWEEP_SCOPE


def _is_formal_mechanism_trace_acquisition(args: argparse.Namespace) -> bool:
    return _scope_label(args) == FORMAL_MECHANISM_TRACE_SCOPE


def _is_formal_evidence_acquisition(args: argparse.Namespace) -> bool:
    return _scope_label(args) != SMOKE_SCOPE


def _scope_label(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "scope", "auto"))
    if requested != "auto":
        return requested
    if _matches_formal_latency_sweep(args):
        return FORMAL_LATENCY_SWEEP_SCOPE
    if _matches_formal_mechanism_trace(args):
        return FORMAL_MECHANISM_TRACE_SCOPE
    return SMOKE_SCOPE


def _scope_resolution(args: argparse.Namespace) -> str:
    return "explicit" if str(args.scope) != "auto" else "auto_protocol_match"


def _write_report(root: Path, rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    scope = _scope_label(args)
    manifest = {
        "schema": "closed_loop_latency_sweep_manifest_v1",
        "scope": scope,
        "scope_request": str(args.scope),
        "scope_resolution": _scope_resolution(args),
        "groups": list(args.groups),
        "latencies_s": [float(value) for value in args.latencies],
        "seed_start": int(args.seed_start),
        "seeds": int(args.seeds),
        "policy_frequency_hz": float(args.policy_frequency),
        "is_formal_closed_loop_latency_sweep": scope == FORMAL_LATENCY_SWEEP_SCOPE,
        "is_formal_mechanism_trace_acquisition": scope == FORMAL_MECHANISM_TRACE_SCOPE,
        "is_formal_evidence_acquisition": scope != SMOKE_SCOPE,
        "row_count": len(rows),
    }
    path = root / "closed_loop_latency_sweep_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _write_report(Path("."), [], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
