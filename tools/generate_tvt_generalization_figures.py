"""Generate MetaDrive transfer and multi-LLM generalization figures for TVT."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("paper2/generated_figures")
METADRIVE_SUMMARY = Path("results/metadrive_result/analysis/metadrive_summary.csv")
MULTI_LLM_SUMMARY = Path("results/multi_llm_probe/analysis/multi_llm_executor_probe_summary.csv")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)


def style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_metadrive_transfer(rows: Sequence[Dict[str, str]]) -> Path:
    """Grouped bars: success and calls/episode by allocator on MetaDrive envs."""
    style()
    # Prefer short envs first.
    env_order = ["metadrive-highway-v0", "metadrive-merge-v0", "metadrive-intersection-v0", "metadrive-roundabout-v0"]
    group_order = ["rgd_fixed_policy", "always_fast", "random_budget", "risk_budget", "uncertainty_budget"]
    labels = {
        "rgd_fixed_policy": "RGD",
        "always_fast": "Fast-only",
        "random_budget": "Random",
        "risk_budget": "TTC-risk",
        "uncertainty_budget": "Uncertainty",
    }
    colors = {
        "rgd_fixed_policy": "#0072B2",
        "always_fast": "#4D4D4D",
        "random_budget": "#E69F00",
        "risk_budget": "#D55E00",
        "uncertainty_budget": "#009E73",
    }
    envs = [e for e in env_order if any(r.get("env") == e for r in rows)]
    groups = [g for g in group_order if any(r.get("group") == g for r in rows)]
    if not envs or not groups:
        raise ValueError("MetaDrive summary has no usable env/group rows")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={"wspace": 0.32})

    # Success
    ax = axes[0]
    x = np.arange(len(envs))
    width = 0.14
    for i, g in enumerate(groups):
        vals = []
        for env in envs:
            hit = [r for r in rows if r.get("env") == env and r.get("group") == g]
            vals.append(_f(hit[0], "success_rate") if hit else 0.0)
        ax.bar(
            x + (i - (len(groups) - 1) / 2) * width,
            vals,
            width=width,
            color=colors.get(g, "#777"),
            label=labels.get(g, g),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, [e.replace("metadrive-", "").replace("-v0", "") for e in envs])
    ax.set_ylabel("Collision-free completion")
    ax.set_ylim(0, max(0.45, max((_f(r, "success_rate") for r in rows), default=0.3) * 1.25))
    clean(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # Calls / episode (online only)
    ax = axes[1]
    online = [g for g in groups if g != "always_fast"]
    for i, g in enumerate(online):
        vals = []
        for env in envs:
            hit = [r for r in rows if r.get("env") == env and r.get("group") == g]
            # prefer attempts_per_episode, else slow_call_rate * horizon proxy
            if hit:
                if "attempts_per_episode" in hit[0] and hit[0]["attempts_per_episode"] not in ("", None):
                    vals.append(_f(hit[0], "attempts_per_episode"))
                else:
                    vals.append(_f(hit[0], "slow_call_rate") * 100.0)
            else:
                vals.append(0.0)
        ax.bar(
            x + (i - (len(online) - 1) / 2) * width,
            vals,
            width=width,
            color=colors.get(g, "#777"),
            label=labels.get(g, g),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x, [e.replace("metadrive-", "").replace("-v0", "") for e in envs])
    ax.set_ylabel("Slow exposure (calls/ep. or rate×100)")
    clean(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    out = OUT / "fig_metadrive_transfer.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return out


def fig_multi_llm(rows: Sequence[Dict[str, str]]) -> Path:
    style()
    # sort by success then inverse call rate
    ordered = sorted(rows, key=lambda r: (-_f(r, "success_rate"), _f(r, "slow_call_rate")))
    labels = [str(r.get("label") or r.get("model")) for r in ordered]
    success = [_f(r, "success_rate") for r in ordered]
    calls = [_f(r, "slow_call_rate") for r in ordered]
    preserve = [_f(r, "route_action_preservation_rate") for r in ordered]
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, success, color=colors[: len(labels)], edgecolor="white", width=0.72)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Completion (mean over probe envs)")
    ax.set_ylim(0, max(0.35, max(success + [0.1]) * 1.25))
    clean(ax)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    ax = axes[1]
    ax.scatter(calls, success, s=54, c=colors[: len(labels)], edgecolor="white", linewidth=0.6, zorder=3)
    for i, lab in enumerate(labels):
        ax.text(calls[i] + 0.003, success[i] + 0.004, lab, fontsize=6.4, color=colors[i % len(colors)])
    ax.set_xlabel("Slow-call rate")
    ax.set_ylabel("Completion")
    clean(ax)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    if preserve and max(preserve) > 0:
        ax.text(0.02, 0.03, f"mean route preserve={np.mean(preserve):.2f}", transform=ax.transAxes, fontsize=6.3)

    out = OUT / "fig_multi_llm_generalization.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    return out


def summarise_metadrive_bundle(bundle_root: Path, out_csv: Path) -> Path:
    """Aggregate group run rows under a MetaDrive formal bundle into one summary CSV."""
    rows_out: List[Dict[str, Any]] = []
    for group_dir in sorted(bundle_root.glob("*/")):
        group = group_dir.name
        rows_path = group_dir / f"{group}_run_rows.csv"
        if not rows_path.is_file():
            continue
        rows = _read_csv(rows_path)
        envs = sorted({r.get("env", "") for r in rows})
        for env in envs:
            sub = [r for r in rows if r.get("env") == env]
            if not sub:
                continue
            n = len(sub)
            def avg(key: str) -> float:
                vals = [_f(r, key) for r in sub]
                return float(sum(vals) / max(1, len(vals)))

            success_keys = [k for k in ("success_rate", "episode_success", "success") if any(k in r for r in sub)]
            success = avg(success_keys[0]) if success_keys else 0.0
            # if binary success column
            if success == 0.0 and any("success" in r for r in sub):
                success = avg("success")
            call = avg("slow_call_rate")
            attempts = avg("slow_attempts") if any("slow_attempts" in r for r in sub) else call
            distance = avg("avg_driving_distance") if any("avg_driving_distance" in r for r in sub) else avg("driving_distance")
            speed = avg("avg_speed_all_frames") if any("avg_speed_all_frames" in r for r in sub) else avg("mean_speed")
            rows_out.append(
                {
                    "group": group,
                    "env": env,
                    "n": n,
                    "success_rate": success,
                    "slow_call_rate": call,
                    "attempts_per_episode": attempts,
                    "avg_driving_distance": distance,
                    "avg_speed_all_frames": speed,
                    "route_action_preservation_rate": avg("route_action_preservation_rate"),
                    "safety_override_rate": avg("safety_override_rate"),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        raise FileNotFoundError(f"no group rows under {bundle_root}")
    fieldnames = list(rows_out[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return out_csv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md_bundle = Path("results/metadrive_result/formal_run/2026-07-15/metadrive_stress")
    if md_bundle.is_dir():
        try:
            summarise_metadrive_bundle(md_bundle, METADRIVE_SUMMARY)
            print("metadrive_summary", METADRIVE_SUMMARY)
            if METADRIVE_SUMMARY.is_file():
                print("fig_metadrive", fig_metadrive_transfer(_read_csv(METADRIVE_SUMMARY)))
        except Exception as exc:  # noqa: BLE001
            print("metadrive_pending", exc)
    else:
        print("metadrive_bundle_missing", md_bundle)

    if MULTI_LLM_SUMMARY.is_file():
        print("fig_multi_llm", fig_multi_llm(_read_csv(MULTI_LLM_SUMMARY)))
    else:
        print("multi_llm_summary_pending", MULTI_LLM_SUMMARY)


if __name__ == "__main__":
    main()
