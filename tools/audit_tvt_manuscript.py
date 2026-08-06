"""Audit the TVT manuscript against protocol-registered, versioned evidence."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER = ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"
MAIN_PDF = PAPER / "main.pdf"
BIB_FILE = PAPER / "references.bib"
CONFIG_FILE = ROOT / "config.yaml"
PROTOCOL_FILE = ROOT / "formal_protocol.yaml"

TEMP_SUFFIXES = (".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(load_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(load_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return payload


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cite_keys(tex: str) -> set[str]:
    return {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex, flags=re.DOTALL)
        for key in group.split(",")
        if key.strip()
    }


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def dotted(mapping: Mapping[str, Any], key: str) -> Any:
    return nested(mapping, *str(key).split("."))


def parse_seed_partition(value: Any, *, field_name: str) -> List[int]:
    """Resolve the inclusive seed notations used by the formal protocol."""
    if isinstance(value, str):
        bounds = value.strip().split("-", 1)
        if len(bounds) != 2:
            raise ValueError(f"{field_name} must use inclusive start-end notation")
        try:
            start, end = (int(item.strip()) for item in bounds)
        except ValueError as exc:
            raise ValueError(f"{field_name} contains a non-integer seed bound") from exc
        count = end - start + 1
    elif isinstance(value, Mapping):
        try:
            start = int(value["start"])
            end = int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integer start/end values") from exc
        count = int(value.get("count", end - start + 1))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) != 2:
            raise ValueError(f"{field_name} bounds must contain exactly two values")
        start, end = (int(item) for item in value)
        count = end - start + 1
    else:
        raise ValueError(f"{field_name} has no supported seed-range encoding")
    if start > end or count != end - start + 1:
        raise ValueError(f"{field_name} is not a valid inclusive seed partition")
    return list(range(start, end + 1))


def resolve_contract_path(root: Path, value: Any, *, field_name: str) -> Path:
    """Resolve a repository-relative contract path without permitting escapes."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is not declared")
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative: {text}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes the repository root: {text}") from exc
    return resolved


def values_equal(observed: Any, expected: Any) -> bool:
    if expected is None:
        return observed in (None, "")
    if isinstance(expected, bool):
        if isinstance(observed, str):
            return observed.strip().lower() == str(expected).lower()
        return observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(observed) == str(expected)


def compare_summary_rows(
    csv_path: Path,
    manifest: Mapping[str, Any],
    manifest_key: str,
) -> List[str]:
    errors: List[str] = []
    csv_rows = load_csv(csv_path)
    manifest_rows = manifest.get(manifest_key)
    if not isinstance(manifest_rows, list):
        return [f"manifest field {manifest_key!r} is not a row list"]
    if len(csv_rows) != len(manifest_rows):
        return [
            f"{csv_path.name} has {len(csv_rows)} rows but manifest {manifest_key} has "
            f"{len(manifest_rows)}"
        ]
    for index, (csv_row, manifest_row) in enumerate(zip(csv_rows, manifest_rows)):
        if not isinstance(manifest_row, Mapping):
            errors.append(f"manifest {manifest_key}[{index}] is not a mapping")
            continue
        for key, expected in manifest_row.items():
            if key in csv_row and not values_equal(csv_row[key], expected):
                errors.append(
                    f"{csv_path.name} row {index} field {key!r} differs from its manifest"
                )
                break
    return errors


def audit_evidence_provenance(
    submission: Mapping[str, Any],
    root: Path = ROOT,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Validate every paper-facing artifact before any result-specific audit."""
    errors: List[str] = []
    resolved: Dict[str, Dict[str, Any]] = {}
    current_method = str(submission.get("rgd_method_version", "") or "")
    prior_method = str(submission.get("prior_paper_result_method_version", "") or "")
    registry = submission.get("evidence_artifacts")
    if not isinstance(registry, Mapping):
        return ["formal protocol omits tvt_submission_contract.evidence_artifacts"], resolved
    compatibility = registry.get("compatibility")
    if not isinstance(compatibility, Mapping):
        return ["evidence artifact compatibility policy is not declared"], resolved

    required_method = str(compatibility.get("required_method_version", "") or "")
    if not current_method or required_method != current_method:
        errors.append(
            "evidence compatibility required_method_version does not match rgd_method_version"
        )
    if str(compatibility.get("prior_method_version", "") or "") != prior_method:
        errors.append("evidence compatibility prior_method_version does not match the protocol")
    for policy in ("cross_method_version_policy", "missing_manifest_method_version_policy"):
        if compatibility.get(policy) != "reject":
            errors.append(f"evidence compatibility {policy} must be reject")

    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        errors.append("evidence artifact registry is empty")
        return errors, resolved

    for name, raw_spec in artifacts.items():
        artifact_error_count = len(errors)
        if not isinstance(raw_spec, Mapping):
            errors.append(f"evidence artifact {name!r} has no mapping contract")
            continue
        spec = dict(raw_spec)
        if spec.get("paper_facing") is not True:
            continue
        status = str(spec.get("status", "") or "")
        declared_method = str(spec.get("method_version", "") or "")
        if status != "current":
            errors.append(
                f"paper-facing artifact {name!r} is {status or 'unversioned'}, not current for "
                f"{current_method}; {prior_method or 'prior-method'} evidence cannot be mixed in"
            )
            continue
        if declared_method == prior_method and prior_method:
            errors.append(
                f"paper-facing artifact {name!r} declares legacy method {prior_method}; refusing "
                f"to mix it with {current_method}"
            )
            continue
        if declared_method != current_method:
            errors.append(
                f"paper-facing artifact {name!r} declares method {declared_method!r}, expected "
                f"{current_method!r}"
            )
            continue

        try:
            artifact_root = resolve_contract_path(
                root, spec.get("artifact"), field_name=f"evidence_artifacts.{name}.artifact"
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        manifest_name = str(spec.get("manifest", "") or "")
        if not manifest_name or Path(manifest_name).is_absolute() or len(Path(manifest_name).parts) != 1:
            errors.append(f"evidence artifact {name!r} must declare a local manifest filename")
            continue
        manifest_path = artifact_root / manifest_name
        data_files = [str(item) for item in (spec.get("data_files") or [])]
        required_paths = [manifest_path] + [artifact_root / item for item in data_files]
        missing = [path for path in required_paths if not path.is_file()]
        if missing:
            errors.extend(
                f"missing current {name} evidence file: {path.relative_to(root.resolve())}"
                for path in missing
            )
            continue
        try:
            manifest = load_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {name} manifest: {exc}")
            continue

        observed_method = str(manifest.get("method_version", "") or "")
        if not observed_method:
            errors.append(
                f"{name} manifest omits method_version; unversioned evidence is rejected for "
                f"{current_method}"
            )
        elif observed_method == prior_method and prior_method:
            errors.append(
                f"{name} manifest is legacy {prior_method}; refusing to mix it with {current_method}"
            )
        elif observed_method != current_method:
            errors.append(
                f"{name} manifest method_version={observed_method!r}, expected {current_method!r}"
            )

        seed_contract = str(spec.get("seed_contract", "") or "")
        if seed_contract:
            try:
                expected_seeds = parse_seed_partition(
                    submission.get(seed_contract), field_name=f"tvt_submission_contract.{seed_contract}"
                )
                observed_seeds = [int(value) for value in manifest.get("seeds", [])]
                if observed_seeds != expected_seeds:
                    errors.append(
                        f"{name} manifest seed cohort differs from protocol {seed_contract}"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))

        requirements = spec.get("required_manifest_values") or {}
        if not isinstance(requirements, Mapping):
            errors.append(f"{name} required_manifest_values is not a mapping")
        else:
            for key, expected in requirements.items():
                observed = dotted(manifest, str(key))
                if not values_equal(observed, expected):
                    errors.append(
                        f"{name} manifest {key}={observed!r}, expected protocol value {expected!r}"
                    )

        summary_name = str(spec.get("summary_file", "") or "")
        summary_key = str(spec.get("summary_manifest_key", "") or "")
        if summary_name and summary_key:
            errors.extend(compare_summary_rows(artifact_root / summary_name, manifest, summary_key))
        if len(errors) == artifact_error_count:
            resolved[str(name)] = {
                "root": artifact_root,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "spec": spec,
            }
    return errors, resolved


def pdf_pages(path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def audit_runtime_alignment(
    config: Mapping[str, Any], protocol: Mapping[str, Any]
) -> List[str]:
    errors: List[str] = []
    runtime = dict(protocol.get("runtime_config", {}) or {})
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    pairs = {
        "RGD threshold": (
            config.get("rgd_decision_threshold"),
            runtime.get("rgd_decision_threshold"),
        ),
        "request timeout": (
            config.get("QWEN_REQUEST_TIMEOUT_S"),
            runtime.get("QWEN_REQUEST_TIMEOUT_S"),
        ),
        "slow token limit": (
            nested(config, "slow_thinking", "max_tokens"),
            nested(runtime, "slow_thinking", "max_tokens"),
        ),
        "hidden slower bridge": (
            nested(config, "hidden_slower_bridge", "enable"),
            nested(runtime, "hidden_slower_bridge", "enable"),
        ),
        "delay seconds": (
            nested(config, "closed_loop_latency_replay", "extra_latency_s"),
            nested(runtime, "closed_loop_latency_replay", "extra_latency_s"),
        ),
        "delay steps": (
            nested(config, "closed_loop_latency_replay", "delay_steps"),
            nested(runtime, "closed_loop_latency_replay", "delay_steps"),
        ),
    }
    for name, (config_value, protocol_value) in pairs.items():
        if not values_equal(config_value, protocol_value):
            errors.append(
                f"{name} differs between config and protocol: "
                f"{config_value!r} != {protocol_value!r}"
            )
    if not values_equal(pairs["delay seconds"][1], submission.get("main_delay_s")):
        errors.append("runtime delay seconds differs from the submission contract")
    if not values_equal(pairs["delay steps"][1], submission.get("main_delay_steps")):
        errors.append("runtime delay steps differs from the submission contract")
    if config.get("rgd_threshold_selection_rule") != runtime.get("rgd_threshold_selection_rule"):
        errors.append("threshold-selection rule differs between config and formal protocol")
    if submission.get("supplementary_material") is not False:
        errors.append("TVT submission contract must declare supplementary_material: false")

    paper_evidence = dict(config.get("paper_evidence_protocol", {}) or {})
    partition_pairs = (
        ("calibration_seeds", "calibration_seeds"),
        ("descriptive_main_seeds", "main_seeds"),
        ("latency_endpoint_seeds", "latency_endpoint_seeds"),
        ("mechanism_evaluation_seeds", "mechanism_evaluation_seeds"),
    )
    for config_key, protocol_key in partition_pairs:
        try:
            config_seeds = parse_seed_partition(
                paper_evidence.get(config_key), field_name=f"config.paper_evidence_protocol.{config_key}"
            )
            protocol_seeds = parse_seed_partition(
                submission.get(protocol_key),
                field_name=f"tvt_submission_contract.{protocol_key}",
            )
            if config_seeds != protocol_seeds:
                errors.append(f"seed partition differs for {config_key}/{protocol_key}")
        except ValueError as exc:
            errors.append(str(exc))
    if paper_evidence.get("mechanism_cohort_status") != submission.get("mechanism_cohort_status"):
        errors.append("mechanism cohort status differs between config and formal protocol")

    rollout_cfg = dict(paper_evidence.get("release_rollout", {}) or {})
    rollout_contract = dict(submission.get("release_rollout", {}) or {})
    for key in ("horizon_steps", "gamma", "corrective_margin", "utility", "bootstrap_unit", "bootstrap_draws"):
        if not values_equal(rollout_cfg.get(key), rollout_contract.get(key)):
            errors.append(f"release-rollout {key} differs between config and protocol")
    if not values_equal(paper_evidence.get("ttc_delay_threshold"), submission.get("ttc_delay_threshold")):
        errors.append("TTC-delay threshold differs between config and protocol")

    component_cfg = dict(paper_evidence.get("component_ablation", {}) or {})
    component_contract = dict(submission.get("component_ablation", {}) or {})
    component_keys = (
        "artifact",
        "design",
        "components",
        "delay_s",
        "horizon_steps",
        "gamma",
        "corrective_margin",
        "opportunity_floor",
        "priority_threshold",
        "budget",
        "cooldown_frames",
        "removed_component_value",
        "remove_support_hard_gate_when_A_removed",
        "absolute_alternative_feasibility_non_ablatable",
        "alternative_metric_source",
        "headroom_metric_source",
        "viable_cost_threshold",
        "threshold_policy",
        "bootstrap_unit",
        "bootstrap_draws",
    )
    for key in component_keys:
        if not values_equal(component_cfg.get(key), component_contract.get(key)):
            errors.append(f"component-ablation {key} differs between config and protocol")
    try:
        component_config_seeds = parse_seed_partition(
            component_cfg.get("seeds"), field_name="config component-ablation seeds"
        )
        component_protocol_seeds = parse_seed_partition(
            component_contract.get("seed_range"), field_name="protocol component-ablation seed_range"
        )
        validation_seeds = parse_seed_partition(
            submission.get("component_validation_seeds"),
            field_name="tvt_submission_contract.component_validation_seeds",
        )
        if component_config_seeds != component_protocol_seeds or component_protocol_seeds != validation_seeds:
            errors.append("component-ablation seed contracts are not aligned")
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def audit_result_shapes(
    protocol: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], int]:
    errors: List[str] = []
    transfer_rows = 0
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})

    main = resolved.get("main_results")
    if main:
        rows = load_csv(main["root"] / main["spec"]["data_files"][0])
        expected_groups = set(nested(protocol, "paper_baselines", "main_text_groups") or [])
        if {row.get("group") for row in rows} != expected_groups:
            errors.append("main-results groups differ from the protocol main-text groups")
        expected_episodes = len(parse_seed_partition(submission.get("main_seeds"), field_name="main_seeds"))
        if any(int(row.get("episodes", -1)) != expected_episodes for row in rows):
            errors.append("main-results episode denominator differs from the main seed partition")

    transfer = resolved.get("transfer_results")
    if transfer:
        rows = load_csv(transfer["root"] / transfer["spec"]["data_files"][0])
        transfer_rows = len(rows)
        expected_rows = int(nested(submission, "table_vii", "total_episodes") or 0)
        if len(rows) != expected_rows:
            errors.append(f"transfer evidence has {len(rows)} rows, expected {expected_rows}")
        keys = {
            (row.get("group"), row.get("lanes_count"), row.get("vehicles_density"), row.get("seed"))
            for row in rows
        }
        if len(keys) != len(rows):
            errors.append("transfer evidence contains duplicate execution keys")
        if any(str(row.get("replay_delay_positive", "false")).lower() == "true" for row in rows):
            errors.append("transfer evidence contains positive replay delay")

    executor = resolved.get("executor_diagnostics")
    if executor:
        rows = load_csv(executor["root"] / executor["spec"]["data_files"][0])
        observed_labels = {row.get("label") for row in rows}
        expected_labels = {
            item.get("label")
            for item in (nested(submission, "table_vii", "slow_executor_models") or [])
        }
        if observed_labels != expected_labels:
            errors.append("executor diagnostics do not cover exactly the protocol models")

    component = resolved.get("component_ablation")
    release = resolved.get("release_rollout")
    if component and release:
        try:
            component_schema = str(component["manifest"].get("schema", "") or "")
            if component_schema == "identifiable_gate_v12_component_ablation_evidence_v1":
                from tools.export_identifiable_gate_v12_evidence import (
                    verify_published_bundle,
                )

                if component["root"].resolve() != release["root"].resolve():
                    raise ValueError(
                        "v12 release and component evidence must share one atomically "
                        "published bundle root"
                    )
                verify_published_bundle(component["root"])
            else:
                from tools.verify_rgd_component_ablation import (
                    verify as verify_component_ablation,
                )

                verify_component_ablation(component["root"], release["root"])
        except Exception as exc:  # pragma: no cover - surfaced as an audit finding.
            errors.append(f"component-ablation derivation check failed: {exc}")
    return errors, transfer_rows


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []
    for path in (MAIN_TEX, MAIN_PDF, BIB_FILE, CONFIG_FILE, PROTOCOL_FILE):
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    tex = load_text(MAIN_TEX)
    bib = load_text(BIB_FILE)
    config = load_yaml(CONFIG_FILE)
    protocol = load_yaml(PROTOCOL_FILE)
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})

    citations = cite_keys(tex)
    bibliography = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
    missing_citations = sorted(citations - bibliography)
    unused_citations = sorted(bibliography - citations)
    if missing_citations:
        errors.append(f"citation keys missing from bibliography: {', '.join(missing_citations)}")
    if unused_citations:
        errors.append(f"uncited bibliography entries: {', '.join(unused_citations)}")

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    references = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex))
    if duplicate_labels:
        errors.append(f"duplicate labels: {', '.join(duplicate_labels)}")
    undefined_references = sorted(references - set(labels))
    if undefined_references:
        errors.append(f"undefined references: {', '.join(undefined_references)}")

    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    for relative in figures:
        if not (PAPER / relative).is_file():
            errors.append(f"missing figure: {relative}")

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if not abstract_match:
        errors.append("abstract environment not found")
    else:
        abstract = re.sub(r"\\[A-Za-z]+", " ", abstract_match.group(1))
        word_count = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract))
        number_count = len(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", abstract))
        if not 150 <= word_count <= 250:
            errors.append(f"abstract word count outside 150--250: {word_count}")
        if number_count > 18:
            errors.append(f"abstract contains too many numeric tokens: {number_count}")

    errors.extend(audit_runtime_alignment(config, protocol))
    evidence_errors, resolved = audit_evidence_provenance(submission, ROOT)
    errors.extend(evidence_errors)
    shape_errors, transfer_rows = audit_result_shapes(protocol, resolved)
    errors.extend(shape_errors)

    secret_text = load_text(CONFIG_FILE)
    if re.search(r"(?i)\bsk-[A-Za-z0-9_-]{12,}\b", secret_text):
        errors.append("config.yaml contains a plaintext API credential")
    if re.search(r"\\(?:input|include)\{[^}]*supp", tex, flags=re.IGNORECASE):
        errors.append("manuscript includes supplementary material")

    pages = pdf_pages(MAIN_PDF)
    if pages is None:
        warnings.append("pdfinfo unavailable; PDF page count was not checked")
    elif not 10 <= pages <= 16:
        errors.append(f"compiled PDF has {pages} pages, expected 10--16")
    for path in PAPER.iterdir():
        if path.is_file() and path.name.lower().endswith(TEMP_SUFFIXES):
            errors.append(f"temporary LaTeX file remains in paper/: {path.name}")

    print("TVT MANUSCRIPT AND VERSIONED-EVIDENCE AUDIT")
    print(
        f"citations={len(citations)} figures={len(figures)} labels={len(labels)} "
        f"current_artifacts={len(resolved)} transfer_rows={transfer_rows} pdf_pages={pages}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("PASS: manuscript, runtime protocol, and versioned evidence artifacts are aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
