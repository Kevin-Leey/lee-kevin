"""Run multi-LLM executor diagnostic for the RGD route layer."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from dilu.driver_agent.qwen_adapter import RETRYABLE_HTTP_STATUSES


METRICS = [
    "collision_rate",
    "success_rate",
    "avg_driving_distance",
    "avg_speed_all_frames",
    "avg_speed_safety_qualified",
    "slow_call_rate",
    "slow_call_success_rate",
    "llm_action_preservation_rate",
    "avg_runtime_per_frame",
    "latency_p95",
    "route_action_preservation_rate",
    "safety_override_rate",
    "independent_selective_routing_gain",
]


def _resolved_retry_contract(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the retry settings that are persisted in a runtime manifest."""
    max_attempts = max(1, int((cfg or {}).get("LLM_MAX_ATTEMPTS", 3) or 3))
    initial_backoff = max(
        0.0, float((cfg or {}).get("LLM_RETRY_BACKOFF_S", 0.5) or 0.5)
    )
    return {
        "max_attempts": max_attempts,
        "max_attempts_including_initial_request": max_attempts,
        "initial_backoff_s": initial_backoff,
        "schedule": "exponential",
        "retryable_http_statuses": list(RETRYABLE_HTTP_STATUSES),
        "retryable_transport_failures": ["requests.Timeout", "requests.ConnectionError"],
    }


def _safe_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:80]


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[Dict[str, Any]], metric: str) -> float:
    values = [_safe_float(row.get(metric)) for row in rows]
    return float(mean(values)) if values else 0.0


def _load_model_specs(config_path: Path, models_cli: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    specs = list(cfg.get("MULTI_LLM_MODELS") or [])
    if models_cli:
        wanted = set(models_cli)
        filtered = []
        for spec in specs:
            key = str(spec.get("model", "")).strip()
            label = str(spec.get("label", "")).strip()
            if key in wanted or label in wanted or f"{spec.get('provider')}:{key}" in wanted:
                filtered.append(spec)
        if not filtered:
            # fall back to simple siliconflow model names from CLI
            for model in models_cli:
                if ":" in model:
                    provider, name = model.split(":", 1)
                    filtered.append({"provider": provider, "model": name, "label": name})
                else:
                    filtered.append({"provider": "siliconflow", "model": model, "label": model})
        return filtered
    if not specs:
        return [
            {"provider": "siliconflow", "model": "Qwen/Qwen3-8B", "label": "Qwen3-8B"},
            {"provider": "siliconflow", "model": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5-7B"},
        ]
    return specs


def _write_model_protocol(
    base_protocol: Path,
    output_path: Path,
    spec: Dict[str, Any],
    base_cfg: Dict[str, Any],
    *,
    min_observation_frames: int = 1,
) -> None:
    protocol = yaml.safe_load(base_protocol.read_text(encoding="utf-8-sig"))
    if not isinstance(protocol, dict):
        raise ValueError(f"protocol must be a YAML mapping: {base_protocol}")
    group = ((protocol.get("groups", {}) or {}).get("rgd_fixed_policy", {}) or {})
    overrides = group.setdefault("runtime_overrides", {})
    runtime_config = protocol.setdefault("runtime_config", {})

    provider = str(spec.get("provider", "siliconflow") or "siliconflow").strip().lower()
    model = str(spec.get("model", "")).strip()
    base_url = str(spec.get("base_url", "") or "").strip().rstrip("/")
    retry_contract = _resolved_retry_contract(base_cfg)

    for target in (overrides, runtime_config):
        target["LLM_PROVIDER"] = provider
        target["QWEN_TEMPERATURE"] = 0.0
        target["QWEN_REQUEST_TIMEOUT_S"] = float(base_cfg.get("QWEN_REQUEST_TIMEOUT_S", 60.0) or 60.0)
        target["rgd_min_observation_frames"] = max(1, int(min_observation_frames))
        target["LLM_MAX_ATTEMPTS"] = int(retry_contract["max_attempts"])
        target["LLM_RETRY_BACKOFF_S"] = float(retry_contract["initial_backoff_s"])
        if provider in {"openai_compatible", "openai", "bizdecipher"}:
            target["OPENAI_CHAT_MODEL"] = model
            target["BIZDECIPHER_CHAT_MODEL"] = model
            target["OPENAI_BASE_URL"] = base_url or str(base_cfg.get("OPENAI_BASE_URL", "") or "https://bizdecipher.com/v1")
            target["BIZDECIPHER_BASE_URL"] = target["OPENAI_BASE_URL"]
            # Some third-party models reject forced json_object.
            target["LLM_FORCE_JSON_RESPONSE"] = False
        else:
            target["SILICONFLOW_CHAT_MODEL"] = model
            target["SILICONFLOW_BASE_URL"] = str(base_cfg.get("SILICONFLOW_BASE_URL", "") or "https://api.siliconflow.cn/v1")
            target["LLM_FORCE_JSON_RESPONSE"] = True

        for credential_key in (
            "OPENAI_API_KEY",
            "BIZDECIPHER_API_KEY",
            "SILICONFLOW_API_KEY",
            "TOKENPLAN_API_KEY",
        ):
            target.pop(credential_key, None)
        target.pop("QWEN_ENABLE_THINKING", None)
        target.pop("QWEN_THINKING_BUDGET", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_model(
    *,
    spec: Dict[str, Any],
    protocol_path: Path,
    result_root: Path,
    mode: str,
    groups: Sequence[str],
    envs: Sequence[str],
    seeds: int,
    seed_start: int,
    simulation_duration: Optional[int],
    run_stamp_prefix: str,
) -> Path:
    slug = _safe_slug(str(spec.get("label") or spec.get("model")))
    stamp = f"{run_stamp_prefix}/{slug}"
    log_dir = result_root / mode / run_stamp_prefix / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{slug}.log"
    cmd = [
        sys.executable,
        "tools/run_main_table.py",
        "--protocol",
        str(protocol_path),
        "--mode",
        mode,
        "--partition",
        "nonformal",
        "--allow-nonformal",
        "--groups",
        *list(groups),
        "--envs",
        *list(envs),
        "--seeds",
        str(seeds),
        "--seed-start",
        str(seed_start),
        "--episodes",
        "1",
        "--result-root",
        str(result_root),
        "--run-stamp",
        stamp,
    ]
    if simulation_duration is not None:
        cmd.extend(["--simulation-duration", str(int(simulation_duration))])
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(cmd, check=True, stdout=log_handle, stderr=subprocess.STDOUT)
    # Primary rows path for RGD fixed policy
    return result_root / mode / stamp / "rgd_fixed_policy" / "rgd_fixed_policy_run_rows.csv"


def _summarise_model(
    spec: Dict[str, Any],
    rows_path: Path,
    *,
    expected_envs: Optional[Sequence[str]] = None,
    expected_seeds: Optional[Sequence[int]] = None,
    retry_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = _read_csv(rows_path)
    observed_pairs = {
        (str(row.get("env", "")), int(row.get("seed_idx", -1))) for row in rows
    }
    if expected_envs is not None and expected_seeds is not None:
        expected_pairs = {
            (str(env), int(seed)) for env in expected_envs for seed in expected_seeds
        }
        if observed_pairs != expected_pairs:
            raise ValueError("incomplete model matrix")

    expected_retry = dict(retry_contract or _resolved_retry_contract({}))
    records: List[Dict[str, Any]] = []
    for row in rows:
        result_dir = Path(str(row.get("result_dir", "") or ""))
        manifest_path = result_dir / "runtime_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing runtime manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        backend = dict(manifest.get("llm_backend", {}) or {})
        if (
            str(backend.get("provider", "")) != str(spec.get("provider", ""))
            or str(backend.get("requested_model", "")) != str(spec.get("model", ""))
        ):
            raise ValueError("LLM manifest drift")
        observed_retry = dict(backend.get("retry_contract", {}) or {})
        retry_fields = [
            "retryable_http_statuses",
            "retryable_transport_failures",
            "max_attempts",
            "max_attempts_including_initial_request",
            "initial_backoff_s",
            "schedule",
        ]
        for field in retry_fields:
            value = expected_retry[field]
            if observed_retry.get(field) != value:
                raise ValueError(f"retry contract drift: {field}")
        for trace_path in sorted(result_dir.glob("ep_*/*_reasoning_records.json")):
            payload = json.loads(trace_path.read_text(encoding="utf-8-sig"))
            records.extend(list(payload.get("analysis_records", payload.get("records", [])) or []))

    slow_attempts = [
        record
        for record in records
        if str(record.get("system_used", "")) in {"slow", "fast_after_slow_failure"}
    ]
    successes = [record for record in slow_attempts if bool(record.get("slow_reasoning_success", False))]
    failures = [record for record in slow_attempts if not bool(record.get("slow_reasoning_success", False))]
    parse_failures = [
        record
        for record in failures
        if str(record.get("slow_reasoning_failure_reason", "")).startswith("structured_parse_failed")
    ]
    fallbacks = [
        record for record in slow_attempts if str(record.get("system_used", "")) == "fast_after_slow_failure"
    ]
    result: Dict[str, Any] = {
        "label": str(spec.get("label") or spec.get("model")),
        "provider": str(spec.get("provider", "")),
        "model": str(spec.get("model", "")),
        "runs": len(rows),
        "env_count": len({str(row.get("env", "")) for row in rows}),
        "seed_count": len({str(row.get("seed_idx", "")) for row in rows}),
        "rows_path": str(rows_path),
        "slow_attempts": len(slow_attempts),
        "slow_successes": len(successes),
        "slow_failures": len(failures),
        "slow_parse_failures": len(parse_failures),
        "slow_fallbacks": len(fallbacks),
        "slow_call_success_rate": (
            float(len(successes) / len(slow_attempts)) if slow_attempts else 0.0
        ),
        "slow_call_rate": (
            float(len(slow_attempts) / len(records)) if records else 0.0
        ),
    }
    for metric in METRICS:
        if metric not in {"slow_call_success_rate", "slow_call_rate"}:
            result[metric] = _mean(rows, metric)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-LLM executor diagnostic for the RGD route layer.")
    parser.add_argument("--base-protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--result-root", type=Path, default=Path("results/multi_llm_probe"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multi_llm_probe/analysis"))
    parser.add_argument("--mode", default="formal_run", choices=["quick_check", "formal_run"])
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["rgd_fixed_policy", "always_fast", "random_budget"],
    )
    parser.add_argument(
        "--envs",
        nargs="*",
        default=[
            "highway-v0",
            "metadrive-highway-v0",
            "metadrive-merge-v0",
        ],
    )
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=60)
    parser.add_argument("--simulation-duration", type=int, default=None)
    parser.add_argument("--run-stamp-prefix", default="2026-07-15/multi_llm_generalization")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--summarise-only", action="store_true")
    parser.add_argument("--smoke-api-only", action="store_true")
    return parser.parse_args()


def _smoke_api(spec: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    from dilu.utils.config import setup_api
    from dilu.driver_agent.qwen_adapter import QwenChatModel

    cfg = dict(base_cfg)
    provider = str(spec.get("provider", "siliconflow")).strip().lower()
    model = str(spec.get("model", "")).strip()
    cfg["LLM_PROVIDER"] = provider
    if provider in {"openai_compatible", "openai", "bizdecipher"}:
        cfg["OPENAI_CHAT_MODEL"] = model
        cfg["OPENAI_API_KEY"] = str(spec.get("api_key") or base_cfg.get("OPENAI_API_KEY") or "")
        cfg["OPENAI_BASE_URL"] = str(spec.get("base_url") or base_cfg.get("OPENAI_BASE_URL") or "https://bizdecipher.com/v1")
        cfg["LLM_FORCE_JSON_RESPONSE"] = False
    else:
        cfg["SILICONFLOW_CHAT_MODEL"] = model
        cfg["LLM_FORCE_JSON_RESPONSE"] = True
    setup_api(cfg)
    client = QwenChatModel(temperature=0.0, max_tokens=32)
    text = client.complete(
        "Return compact JSON only.",
        '{"task":"ping","reply":"ok"}',
    )
    return {"label": spec.get("label"), "model": model, "provider": provider, "ok": True, "preview": text[:120]}


def main() -> int:
    args = parse_args()
    base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig")) or {}
    specs = _load_model_specs(args.config, args.models)

    if args.smoke_api_only:
        reports = []
        for spec in specs:
            try:
                reports.append(_smoke_api(spec, base_cfg))
            except Exception as exc:  # noqa: BLE001 - report all model failures
                reports.append(
                    {
                        "label": spec.get("label"),
                        "model": spec.get("model"),
                        "provider": spec.get("provider"),
                        "ok": False,
                        "error": str(exc)[:300],
                    }
                )
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return 0 if all(bool(item.get("ok")) for item in reports) else 2

    protocol_dir = args.result_root / "protocols"
    model_rows: Dict[str, Path] = {}

    def prepare_or_run(spec: Dict[str, Any]) -> Path:
        slug = _safe_slug(str(spec.get("label") or spec.get("model")))
        protocol_path = protocol_dir / f"{slug}.yaml"
        if not args.summarise_only:
            _write_model_protocol(args.base_protocol, protocol_path, spec, base_cfg)
            return _run_model(
                spec=spec,
                protocol_path=protocol_path,
                result_root=args.result_root,
                mode=args.mode,
                groups=args.groups,
                envs=args.envs,
                seeds=args.seeds,
                seed_start=args.seed_start,
                simulation_duration=args.simulation_duration,
                run_stamp_prefix=args.run_stamp_prefix,
            )
        return (
            args.result_root
            / args.mode
            / args.run_stamp_prefix
            / slug
            / "rgd_fixed_policy"
            / "rgd_fixed_policy_run_rows.csv"
        )

    if args.summarise_only or int(args.parallel_workers) <= 1:
        for spec in specs:
            model_rows[str(spec.get("label") or spec.get("model"))] = prepare_or_run(spec)
    else:
        max_workers = min(max(1, int(args.parallel_workers)), len(specs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spec = {executor.submit(prepare_or_run, spec): spec for spec in specs}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                model_rows[str(spec.get("label") or spec.get("model"))] = future.result()

    summaries: List[Dict[str, Any]] = []
    run_rows: Dict[str, str] = {}
    for spec in specs:
        key = str(spec.get("label") or spec.get("model"))
        rows_path = model_rows[key]
        if not rows_path.is_file():
            raise FileNotFoundError(f"missing run rows for {key}: {rows_path}")
        run_rows[key] = str(rows_path)
        summaries.append(_summarise_model(spec, rows_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "multi_llm_executor_probe_summary.csv"
    json_path = args.output_dir / "multi_llm_executor_probe.json"
    _write_csv(
        summary_path,
        summaries,
        ["label", "provider", "model", "runs", "env_count", "seed_count", *METRICS, "rows_path"],
    )
    payload = {
        "design": {
            "question": "Whether RGD remains useful across LLM families and simulator families.",
            "unit": "closed-loop episode seed",
            "envs": list(args.envs),
            "groups": list(args.groups),
            "seeds": int(args.seeds),
            "seed_start": int(args.seed_start),
            "fixed_factor": "RGD route layer, action schema, temperature 0.0, shared safety map",
            "varied_factor": "online LLM model behind the slow executor",
            "scope": "executor/simulator generalization diagnostic under locked RGD allocation",
        },
        "models": [dict(spec) for spec in specs],
        "run_rows": run_rows,
        "summary": summaries,
    }
    # redact keys in saved JSON
    for model in payload["models"]:
        if "api_key" in model:
            model["api_key"] = "<redacted>"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_path), "json": str(json_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
