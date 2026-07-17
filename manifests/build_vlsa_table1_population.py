#!/usr/bin/env python3
"""Build the immutable 1,600-case VLSA Table 1 population manifest.

The builder intentionally uses only the Python standard library. The emitted
JSONL order and encoding are part of the protocol contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


CONFIG_SCHEMA = "vlsa_table1_translational_protocol.v1"
MANIFEST_SCHEMA = "vlsa_table1_population_case.v1"
RECEIPT_SCHEMA = "vlsa_table1_population_receipt.v1"


class ManifestBuildError(RuntimeError):
    """Raised when the frozen protocol cannot produce its exact population."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_config(config_path: Path) -> tuple[dict[str, Any], str]:
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ManifestBuildError("unexpected protocol config schema")
    if config["population"]["expected_cases"] != 1600:
        raise ManifestBuildError("Table 1 protocol must contain 1,600 cases")
    if config["result_contract"]["expected_results"] != 3200:
        raise ManifestBuildError("paired protocol must contain 3,200 results")
    return config, sha256_bytes(raw)


def resolved_task_index(
    config: dict[str, Any],
    suite: str,
    safety_level: str,
    logical_task_index: int,
) -> int:
    remaps = config["population"]["task_remaps"]
    return int(
        remaps.get(suite, {})
        .get(safety_level, {})
        .get(str(logical_task_index), logical_task_index)
    )


def _source_binding(
    repo_root: Path,
    *,
    suite: str,
    safety_level: str,
    task_name: str,
) -> dict[str, str]:
    bddl_relative = (
        Path("safelibero/libero/libero/bddl_files")
        / suite
        / f"{task_name}.bddl"
    )
    init_relative = (
        Path("safelibero/libero/libero/init_files")
        / suite
        / f"{task_name}_level_{safety_level}.pruned_init"
    )
    bddl_path = repo_root / bddl_relative
    init_path = repo_root / init_relative
    missing = [
        str(path.relative_to(repo_root))
        for path in (bddl_path, init_path)
        if not path.is_file()
    ]
    if missing:
        raise ManifestBuildError(f"missing frozen SafeLIBERO files: {missing}")
    return {
        "bddl_path": bddl_relative.as_posix(),
        "bddl_sha256": sha256_path(bddl_path),
        "initial_states_path": init_relative.as_posix(),
        "initial_states_sha256": sha256_path(init_path),
    }


def build_rows(
    *,
    repo_root: Path,
    config_path: Path,
) -> list[dict[str, Any]]:
    config, config_sha256 = load_config(config_path)
    population = config["population"]
    execution = config["execution"]
    task_bindings: dict[tuple[str, str, str], dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    ordinal = 0

    for suite in population["suite_order"]:
        task_names = population["tasks"][suite]
        for safety_level in population["safety_levels"]:
            for logical_task_index in population["logical_task_indices"]:
                resolved_index = resolved_task_index(
                    config, suite, safety_level, logical_task_index
                )
                try:
                    task_name = task_names[resolved_index]
                except IndexError as error:
                    raise ManifestBuildError(
                        f"invalid task remap for {suite}/{safety_level}/"
                        f"{logical_task_index}: {resolved_index}"
                    ) from error
                binding_key = (suite, safety_level, task_name)
                binding = task_bindings.get(binding_key)
                if binding is None:
                    binding = _source_binding(
                        repo_root,
                        suite=suite,
                        safety_level=safety_level,
                        task_name=task_name,
                    )
                    task_bindings[binding_key] = binding

                suite_short = (
                    suite[len("safelibero_") :]
                    if suite.startswith("safelibero_")
                    else suite
                )
                group_id = (
                    f"vlsa-t1-{suite_short}-{safety_level.lower()}-"
                    f"t{logical_task_index}"
                )
                for episode_index in range(
                    int(population["episode_start_inclusive"]),
                    int(population["episode_stop_exclusive"]),
                ):
                    case_id = f"{group_id}-e{episode_index:02d}"
                    # Reserve a disjoint 1,024-seed block for each case.  The
                    # longest 550-step episode queries at most 110 chunks, so
                    # paired cases never reuse another case's flow-noise key.
                    noise_seed = int(
                        execution["policy_noise_seed_base"]
                    ) + int(
                        execution["policy_noise_case_stride"]
                    ) * ordinal
                    rows.append(
                        {
                            "schema_version": MANIFEST_SCHEMA,
                            "protocol_id": config["protocol_id"],
                            "protocol_config_sha256": config_sha256,
                            "source_commit": config["source"][
                                "upstream_commit"
                            ],
                            "case_ordinal": ordinal,
                            "case_id": case_id,
                            "pair_group_id": case_id,
                            "task_level_group_id": group_id,
                            "suite": suite,
                            "safety_level": safety_level,
                            "logical_task_index": logical_task_index,
                            "resolved_task_index": resolved_index,
                            "task_name": task_name,
                            "episode_index": episode_index,
                            "environment_seed": int(
                                population["environment_seed"]
                            ),
                            "max_steps": int(
                                population["max_steps"][suite]
                            ),
                            "settle_actions": int(
                                execution["settle_actions"]
                            ),
                            "action_space": config["action_space"]["mode"],
                            "model_action_horizon": int(
                                execution["model_action_horizon"]
                            ),
                            "replan_steps": int(
                                execution["replan_steps"]
                            ),
                            "policy_noise_seed": noise_seed,
                            "policy_noise_schedule_id": (
                                f"pi05-query-noise-v1:{noise_seed}"
                            ),
                            "required_arms": list(config["arms"]),
                            "semantic_label_requirement": (
                                "frozen_codex_label_bound_to_agentview_sha256"
                            ),
                            **binding,
                        }
                    )
                    ordinal += 1

    expected = int(population["expected_cases"])
    if len(rows) != expected:
        raise ManifestBuildError(
            f"generated {len(rows)} cases, expected {expected}"
        )
    if len({row["case_id"] for row in rows}) != expected:
        raise ManifestBuildError("case IDs are not unique")
    expected_group_size = int(
        population["expected_cases_per_task_level_group"]
    )
    group_counts: dict[str, int] = {}
    for row in rows:
        key = row["task_level_group_id"]
        group_counts[key] = group_counts.get(key, 0) + 1
    if (
        len(group_counts) != 32
        or set(group_counts.values()) != {expected_group_size}
    ):
        raise ManifestBuildError(
            f"invalid task-level group sizes: {group_counts}"
        )
    return rows


def manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build_receipt(
    *,
    config_path: Path,
    rows: list[dict[str, Any]],
    payload: bytes,
) -> dict[str, Any]:
    _, config_sha256 = load_config(config_path)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": rows[0]["protocol_id"],
        "protocol_config_sha256": config_sha256,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "manifest_rows": len(rows),
        "manifest_sha256": sha256_bytes(payload),
        "first_case_id": rows[0]["case_id"],
        "last_case_id": rows[-1]["case_id"],
        "task_level_groups": len(
            {row["task_level_group_id"] for row in rows}
        ),
        "paired_results_required": sum(
            len(row["required_arms"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "configs/vlsa_table1_translational.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()

    rows = build_rows(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
    )
    payload = manifest_bytes(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    receipt = build_receipt(
        config_path=args.config.resolve(),
        rows=rows,
        payload=payload,
    )
    if args.receipt_output is not None:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    json.dump(receipt, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
