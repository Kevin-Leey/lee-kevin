"""Build the minimal snapshot set for release and component rollouts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_release_state_rollouts import (  # noqa: E402
    DEFAULT_DELAYS,
    _load_trace,
    _selected_queries,
    _selection_contract,
)
from tools.analyze_rgd_component_ablation import ARM_SPECS, _select_frames  # noqa: E402
from tools.run_main_table_runtime import load_formal_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.7)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_formal_protocol(args.protocol)
    threshold = float(
        (protocol.get("tvt_submission_contract", {}) or {})["ttc_delay_threshold"]
    )
    policy_frequency = 10.0
    delay_steps = int(math.ceil(float(args.delay) * policy_frequency))
    targets: dict[str, list[int]] = {}
    component_states = 0
    release_states = 0

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        records, _ = _load_trace(args.trace_root, seed)
        selected = _selected_queries(
            records,
            float(args.delay),
            ttc_delay_threshold=threshold,
        )
        release_specs, _ = _selection_contract(
            selected,
            seed=seed,
            record_count=len(records),
            delays=DEFAULT_DELAYS,
            horizon=args.horizon,
            policy_frequency=policy_frequency,
        )
        seed_targets = {int(release_frame) for _, _, _, release_frame in release_specs}
        release_states += len(seed_targets)

        component_targets = set()
        for spec in ARM_SPECS:
            for query_frame in _select_frames(records, spec, float(args.delay)):
                release_frame = int(query_frame + delay_steps)
                if release_frame + args.horizon <= len(records):
                    component_targets.add(release_frame)
        component_states += len(component_targets)
        seed_targets.update(component_targets)
        targets[str(seed)] = sorted(seed_targets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(targets, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seeds": args.seeds,
                "unique_targets": sum(len(frames) for frames in targets.values()),
                "release_targets_before_union": release_states,
                "component_targets_before_union": component_states,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
