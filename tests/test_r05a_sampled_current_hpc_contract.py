from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "hpc" / "run_r05a_canary.sh"
H100_WRAPPER = ROOT / "scripts" / "hpc" / "run_r05a_sampled_current_canary.sh"
CPU_WRAPPER = ROOT / "scripts" / "hpc" / "validate_r05a_sampled_current_canary.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_sampled_current_canary.sh"
H100_SLURM = ROOT / "slurm" / "r05a_sampled_current_canary_h100.sbatch"
CPU_SLURM = ROOT / "slurm" / "r05a_sampled_current_canary_validate_cpu.sbatch"
APPARATUS_CONFIG = (
    ROOT / "configs" / "experiments" / "r05a_sampled_current_canary_apparatus.json"
)
SCHEMA = ROOT / "schemas" / "r05a-sampled-current-canary-envelope.schema.json"
PUBLISHER = ROOT / "main" / "crfs_oracle" / "r05a_sampled_current_canary.py"

_PACKAGE_NAME = "r05a_sampled_current_contract_test_package"
_PACKAGE = types.ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = []
sys.modules[_PACKAGE_NAME] = _PACKAGE
for _module_name, _path in (
    (
        "r05a_full_lifetime_telemetry",
        ROOT / "main" / "crfs_oracle" / "r05a_full_lifetime_telemetry.py",
    ),
    ("r05a_sampled_current_canary", PUBLISHER),
):
    _qualified = f"{_PACKAGE_NAME}.{_module_name}"
    _spec = importlib.util.spec_from_file_location(_qualified, _path)
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_qualified] = _module
    _spec.loader.exec_module(_module)
publication = sys.modules[f"{_PACKAGE_NAME}.r05a_sampled_current_canary"]


def _assignment_dict(tree: ast.AST, name: str) -> ast.Dict:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    raise AssertionError(f"dictionary assignment {name!r} was not found")


def _function_return_dict(tree: ast.Module, name: str) -> ast.Dict:
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise AssertionError(f"function {name!r} must return one literal dictionary")
    return returns[0].value


def _dict_keys(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("envelope dictionaries must use literal string keys")
        keys.add(key.value)
    return keys


def _dict_value(node: ast.Dict, key: str) -> ast.AST:
    for item_key, value in zip(node.keys, node.values):
        if isinstance(item_key, ast.Constant) and item_key.value == key:
            return value
    raise AssertionError(f"dictionary key {key!r} was not found")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _released_apparatus_config(
    *, run_id: str, accepted_implementation_commit: str = "a" * 40
) -> dict:
    config = json.loads(APPARATUS_CONFIG.read_text(encoding="utf-8"))
    config["ready_to_run"] = True
    config["blocked_on"] = []
    config["execution_release"] = {
        "schema_version": "1.0",
        "artifact_role": "r05a_single_canary_execution_release",
        "decision_artifact": publication.RELEASE_DECISION_PATH,
        "accepted_implementation_commit": accepted_implementation_commit,
        "run_id": run_id,
        "single_submission": True,
        "source_host": publication.SOURCE_NODE,
        "resources": dict(publication.SOURCE_RESOURCE_CONTRACT),
        "release_only_parent_required": True,
        "allowed_release_diff_paths": list(publication.RELEASE_ONLY_PATHS),
        "automatic_resubmission_allowed": False,
        "automatic_next_experiment_allowed": False,
    }
    return config


def _source_contract_fixture(root: Path, *, source_job_id: str = "123") -> dict:
    experiment_root = (root / "experiments").resolve()
    experiment_root.mkdir()
    run_id = "r05a-source-contract-fixture"
    run_root = experiment_root / run_id
    case_dir = run_root / publication.CASE_ID
    case_dir.mkdir(parents=True)
    held_path = run_root / "held-gpu-submission.json"
    resources = dict(publication.SOURCE_RESOURCE_CONTRACT)
    commit = "a" * 40
    held = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_held_gpu_submission",
        "status": "sbatch_returned_held_gpu_id",
        "run_id": run_id,
        "git_commit": commit,
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "source_node": "worker-1",
        "resources": resources,
        "released_at_receipt_time": False,
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
        "timestamp_utc": "2026-07-15T00:00:00Z",
    }
    _write_json(held_path, held)
    artifact_paths = {
        "run_root": str(run_root),
        "case_dir": str(case_dir),
        "payload": str(case_dir / "canary-payload.json"),
        "host_telemetry": str(case_dir / "host-cgroup-sampled-current.tsv"),
        "gpu_samples": str(case_dir / "gpu-memory-samples.csv"),
        "allocation_tests_log": str(case_dir / "allocation-focused-tests.log"),
        "hidden_candidate": str(case_dir / ".results.candidate.json"),
        "result": str(case_dir / "results.json"),
        "validation_receipt": str(run_root / "cpu-afterany-validation.json"),
    }
    repository_hashes = {
        relative: publication.file_sha256(ROOT / relative)
        for relative in publication.BOUND_REPOSITORY_PATHS
    }
    contract = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_source_contract",
        "status": "gpu_held_sources_bound_before_cpu_submission",
        "run_id": run_id,
        "git_commit": commit,
        "git_dirty": False,
        "source_node": "worker-1",
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "resources": resources,
        "artifact_paths": artifact_paths,
        "frozen_bindings": dict(publication.FROZEN_BINDINGS),
        "repository_file_sha256": repository_hashes,
        "held_gpu_submission_path": str(held_path),
        "held_gpu_submission_sha256": publication.file_sha256(held_path),
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
        "timestamp_utc": "2026-07-15T00:00:01Z",
    }
    contract_path = run_root / "source-contract.json"
    _write_json(contract_path, contract)
    return {
        "experiment_root": experiment_root,
        "run_root": run_root,
        "case_dir": case_dir,
        "held_path": held_path,
        "held": held,
        "contract_path": contract_path,
        "contract": contract,
        "commit": commit,
        "source_job_id": source_job_id,
    }


def _schema_subset_errors(value, schema: dict | bool, root: dict, path: str = "$" ) -> list[str]:
    """Dependency-free evaluator for every JSON-Schema keyword used here."""

    if schema is False:
        return [f"{path}: forbidden by schema"]
    if schema is True:
        return []
    if "$ref" in schema:
        reference = schema["$ref"]
        prefix = "#/$defs/"
        if not reference.startswith(prefix):
            return [f"{path}: unsupported reference {reference}"]
        return _schema_subset_errors(value, root["$defs"][reference[len(prefix) :]], root, path)
    errors: list[str] = []
    if "const" in schema and (
        value != schema["const"] or type(value) is not type(schema["const"])
    ):
        errors.append(f"{path}: const mismatch")
    if "enum" in schema and not any(
        value == item and type(value) is type(item) for item in schema["enum"]
    ):
        errors.append(f"{path}: enum mismatch")
    expected_type = schema.get("type")
    type_matches = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in {int, float},
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if expected_type is not None and not type_matches[expected_type]:
        return errors + [f"{path}: type mismatch for {expected_type}"]
    if type(value) is dict:
        required = set(schema.get("required", []))
        missing = required - set(value)
        if missing:
            errors.append(f"{path}: missing {sorted(missing)}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(
                    _schema_subset_errors(item, properties[key], root, f"{path}.{key}")
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key {key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(
                    _schema_subset_errors(
                        item, schema["additionalProperties"], root, f"{path}.{key}"
                    )
                )
    if type(value) is list:
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        prefix_items = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix_items):
            if index < len(value):
                errors.extend(
                    _schema_subset_errors(
                        value[index], item_schema, root, f"{path}[{index}]"
                    )
                )
        remaining_schema = schema.get("items")
        if remaining_schema is not None:
            for index in range(len(prefix_items), len(value)):
                errors.extend(
                    _schema_subset_errors(
                        value[index], remaining_schema, root, f"{path}[{index}]"
                    )
                )
    if type(value) is str:
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: string too short")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: pattern mismatch")
    if type(value) in {int, float}:
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if "oneOf" in schema:
        matches = sum(
            not _schema_subset_errors(value, item, root, path)
            for item in schema["oneOf"]
        )
        if matches != 1:
            errors.append(f"{path}: oneOf matched {matches} branches")
    for clause in schema.get("allOf", []):
        if "if" in clause:
            if not _schema_subset_errors(value, clause["if"], root, path):
                errors.extend(
                    _schema_subset_errors(value, clause.get("then", {}), root, path)
                )
        else:
            errors.extend(_schema_subset_errors(value, clause, root, path))
    return errors


class R05ASampledCurrentHPCContractTest(unittest.TestCase):
    maxDiff = None

    def test_shell_entrypoints_parse_and_required_launchers_are_executable(self) -> None:
        paths = (
            RUNNER,
            H100_WRAPPER,
            CPU_WRAPPER,
            SUBMITTER,
            H100_SLURM,
            CPU_SLURM,
        )
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)
                self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)

    def test_historical_runner_default_is_preserved_and_new_path_is_opt_in(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(': "${R05A_FULL_LIFETIME_TELEMETRY:=false}"', source)
        helper_guard = source.index('if [ "$R05A_FULL_LIFETIME_TELEMETRY" = true ]; then')
        helper_source = source.index('. "$FULL_LIFETIME_HELPER"', helper_guard)
        self.assertLess(helper_guard, helper_source)
        self.assertIn(
            'else\n    "$OPENPI_PYTHON" scripts/serve_policy.py', source
        )
        self.assertIn(
            'if [ "$R05A_FULL_LIFETIME_TELEMETRY" = true ]; then\n'
            '    exec "$OPENPI_PYTHON" scripts/serve_policy.py',
            source,
        )

        monitor = source.index("crfs_monitor_cgroup_v2_full_lifetime")
        ready = source.index('test "$telemetry_ready" = true', monitor)
        setup = source.index("TRANSFORMERS_OVERLAY=", ready)
        focused_tests = source.index("crfs_run_r05a_allocation_tests", setup)
        policy_launch = source.index(
            'crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_SERVER_LAUNCH"',
            focused_tests,
        )
        self.assertLess(monitor, ready)
        self.assertLess(ready, setup)
        self.assertLess(setup, focused_tests)
        self.assertLess(focused_tests, policy_launch)

        payload = source.index('test -f "$PAYLOAD"')
        server_kill = source.index('kill "$SERVER_PID"', payload)
        policy_cleanup = source.index(
            'crfs_write_cgroup_v2_full_lifetime_marker '
            '"$POLICY_SERVER_CLEANUP_COMPLETE"',
            server_kill,
        )
        gpu_stop = source.index(': >"$GPU_MONITOR_STOP"', policy_cleanup)
        gpu_wait = source.index('wait "$GPU_MONITOR_PID"', gpu_stop)
        gpu_cleanup = source.index(
            'crfs_write_cgroup_v2_full_lifetime_marker '
            '"$GPU_MONITOR_CLEANUP_COMPLETE"',
            gpu_wait,
        )
        workload_cleanup = source.index(
            'crfs_write_cgroup_v2_full_lifetime_marker "$WORKLOAD_CLEANUP_COMPLETE"',
            gpu_cleanup,
        )
        stop = source.index(': >"$HOST_TELEMETRY_STOP"', workload_cleanup)
        host_wait = source.index('wait "$HOST_MONITOR_PID"', stop)
        self.assertLess(server_kill, policy_cleanup)
        self.assertLess(policy_cleanup, gpu_stop)
        self.assertLess(gpu_stop, gpu_wait)
        self.assertLess(gpu_wait, gpu_cleanup)
        self.assertLess(gpu_cleanup, workload_cleanup)
        self.assertLess(workload_cleanup, stop)
        self.assertLess(stop, host_wait)

    def test_h100_source_validates_exact_receipts_and_never_publishes_result(self) -> None:
        wrapper = H100_WRAPPER.read_text(encoding="utf-8")
        for fragment in (
            'SLURM_ARRAY_JOB_ID:?exact source array job id is required',
            'test "$SLURM_ARRAY_TASK_ID" = 0',
            '.ready_to_run == true and .blocked_on == []',
            'r05a_sampled_current_canary_source_contract',
            '.source_contract_sha256 == $source_sha',
            '.dependency == ("afterany:" + $gpu)',
            'export R05A_FULL_LIFETIME_TELEMETRY=true',
            'exec "$RUNNER"',
            'SOURCE_CONTRACT=$RUN_ROOT/source-contract.json',
            'SUBMISSION=$RUN_ROOT/submission.json',
        ):
            self.assertIn(fragment, wrapper)
        self.assertNotIn("publish_crfs", wrapper)
        self.assertNotIn('SOURCE_CONTRACT=${SOURCE_CONTRACT:-', wrapper)
        self.assertNotIn('SUBMISSION=${SUBMISSION:-', wrapper)

        runner = RUNNER.read_text(encoding="utf-8")
        result_guard = runner.index('test ! -e "$RESULT"')
        sampled_exit = runner.index("exit 0", result_guard)
        legacy_peak = runner.index("FAILURE_STAGE=host_cgroup_memory_peak", sampled_exit)
        self.assertLess(result_guard, sampled_exit)
        self.assertLess(sampled_exit, legacy_peak)

    def test_cpu_afterany_is_the_only_publisher_and_has_no_gpu(self) -> None:
        slurm = CPU_SLURM.read_text(encoding="utf-8")
        self.assertNotIn("--gres", slurm)
        self.assertIn("#SBATCH --cpus-per-task=2", slurm)
        self.assertIn("#SBATCH --mem=8G", slurm)
        self.assertIn("#SBATCH --time=00:15:00", slurm)
        wrapper = CPU_WRAPPER.read_text(encoding="utf-8")
        for fragment in (
            'test "${SLURM_MEM_PER_NODE:-0}" = 8192',
            'CUDA_VISIBLE_DEVICES:-NoDevFiles',
            '"$job_raw" = "${SOURCE_JOB_ID}_0"',
            'test "$source_state" = COMPLETED',
            'test "$source_exit" = 0:0',
            'publish_crfs_r05a_sampled_current_canary.py',
            '--expected-source-contract-sha256',
        ):
            self.assertIn(fragment, wrapper)

    def test_submission_is_held_bound_cpu_registered_receipted_then_released(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        held = source.index("sbatch --parsable --hold")
        held_receipt = source.index('mv "$held_tmp" "$run_root/held-gpu-submission.json"', held)
        contract = source.index('mv "$source_tmp" "$run_root/source-contract.json"', held_receipt)
        cpu = source.index("cpu_submission=$(", contract)
        external_sha = source.index(
            'EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha"', cpu
        )
        submission = source.index('mv "$submission_tmp" "$run_root/submission.json"', cpu)
        release = source.index('scontrol release "$gpu_job_id"', submission)
        self.assertLess(held, held_receipt)
        self.assertLess(held_receipt, contract)
        self.assertLess(contract, cpu)
        self.assertLess(cpu, external_sha)
        self.assertLess(external_sha, submission)
        self.assertLess(submission, release)
        self.assertIn("--nodelist=worker-1", source)
        self.assertIn("#SBATCH --mem=64G", H100_SLURM.read_text(encoding="utf-8"))
        self.assertNotIn("128G", source + H100_SLURM.read_text(encoding="utf-8"))
        self.assertNotIn("scancel", source)

    def test_implementation_commit_is_fail_closed_before_remote_submission(self) -> None:
        config = json.loads(APPARATUS_CONFIG.read_text(encoding="utf-8"))
        if config["ready_to_run"] is False:
            self.assertGreater(len(config["blocked_on"]), 0)
            environment = {**os.environ, "RUN_ID": "must-not-be-consumed"}
            expected_error = "not released for H100 submission"
        else:
            self.assertEqual(config["blocked_on"], [])
            registered = config["execution_release"]["run_id"]
            environment = {
                **os.environ,
                "RUN_ID": registered + "-alternate-must-not-be-consumed",
                "EXPECTED_RELEASE_COMMIT": "0" * 40,
            }
            expected_error = "not the exact registered canary identity"
        completed = subprocess.run(
            ["bash", str(SUBMITTER)],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(expected_error, completed.stderr)

    def test_exact_execution_release_validator_rejects_identity_and_resource_drift(self) -> None:
        run_id = "r05a-exact-release-fixture"
        valid = _released_apparatus_config(run_id=run_id)
        observed = publication._validate_execution_release(valid, run_id=run_id)
        self.assertEqual(observed, valid["execution_release"])

        mutations = {
            "alternate_run_id": lambda value: value["execution_release"].__setitem__(
                "run_id", run_id + "-alternate"
            ),
            "path_segment_run_id": lambda value: value["execution_release"].__setitem__(
                "run_id", ".."
            ),
            "extra_key": lambda value: value["execution_release"].__setitem__(
                "unexpected", False
            ),
            "invalid_commit": lambda value: value["execution_release"].__setitem__(
                "accepted_implementation_commit", "not-a-commit"
            ),
            "worker_change": lambda value: value["execution_release"].__setitem__(
                "source_host", "worker-2"
            ),
            "release_memory_change": lambda value: value["execution_release"][
                "resources"
            ].__setitem__("host_memory_mib", 131072),
            "apparatus_memory_change": lambda value: value["resource_contract"].__setitem__(
                "host_memory_mib", 131072
            ),
            "allowlist_change": lambda value: value["execution_release"][
                "allowed_release_diff_paths"
            ].append("main/unsafe.py"),
            "resubmission_enabled": lambda value: value["execution_release"].__setitem__(
                "automatic_resubmission_allowed", True
            ),
            "next_gate_enabled": lambda value: value["execution_release"].__setitem__(
                "automatic_next_experiment_allowed", True
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = json.loads(json.dumps(valid))
                mutate(changed)
                with self.assertRaises(ValueError):
                    publication._validate_execution_release(changed, run_id=run_id)

        unreleased = json.loads(APPARATUS_CONFIG.read_text(encoding="utf-8"))
        unreleased["ready_to_run"] = False
        unreleased["blocked_on"] = ["release_identity_not_selected"]
        unreleased.pop("execution_release", None)
        with self.assertRaisesRegex(ValueError, "not released"):
            publication._validate_execution_release(unreleased, run_id=run_id)

    def test_release_commit_requires_single_parent_allowlist_and_config_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "R05A Test"],
                check=True,
            )
            config_path = repository / publication.APPARATUS_CONFIG_PATH
            decision_path = repository / publication.RELEASE_DECISION_PATH
            config_path.parent.mkdir(parents=True)
            decision_path.parent.mkdir(parents=True)
            parent_config = {
                "ready_to_run": False,
                "blocked_on": ["release_identity_not_selected"],
                "stable_science": {"solver": "frozen", "iterations": 128},
            }
            _write_json(config_path, parent_config)
            decision_path.write_text("execution unreleased\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "implementation"],
                check=True,
            )
            implementation = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()

            released = dict(parent_config)
            released["ready_to_run"] = True
            released["blocked_on"] = []
            execution_release = _released_apparatus_config(
                run_id="fixture",
                accepted_implementation_commit=implementation,
            )["execution_release"]
            released["execution_release"] = execution_release
            _write_json(config_path, released)
            decision_path.write_text(
                "execution unreleased\n"
                + publication._release_decision_appendix(execution_release),
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "release"],
                check=True,
            )
            release = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(repository), "update-ref", publication.RELEASE_BRANCH_REF, release],
                check=True,
            )
            changed = publication._validate_release_commit(
                repository,
                observed_commit=release,
                execution_release=execution_release,
            )
            self.assertEqual(
                changed,
                {publication.APPARATUS_CONFIG_PATH, publication.RELEASE_DECISION_PATH},
            )

            exact_release_decision = decision_path.read_text(encoding="utf-8")
            decision_mutations = {
                "trailing_claim": exact_release_decision
                + "unauthorized trailing claim\n",
                "parent_rewrite": exact_release_decision.replace(
                    "execution unreleased", "execution already authorized", 1
                ),
                "run_id_mismatch": exact_release_decision.replace(
                    "Immutable run ID: `fixture`",
                    "Immutable run ID: `fixture-drift`",
                    1,
                ),
            }
            for name, tampered_decision in decision_mutations.items():
                with self.subTest(decision_tamper=name):
                    decision_path.write_text(tampered_decision, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "exact appendix"):
                        publication._validate_release_commit(
                            repository,
                            observed_commit=release,
                            execution_release=execution_release,
                        )
            decision_path.write_text(exact_release_decision, encoding="utf-8")

            subprocess.run(
                ["git", "-C", str(repository), "checkout", "-q", "-b", "bad-content", implementation],
                check=True,
            )
            changed_release = dict(released)
            changed_release["stable_science"] = {"solver": "changed", "iterations": 128}
            _write_json(config_path, changed_release)
            decision_path.write_text("bad content release\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "bad content"],
                check=True,
            )
            bad_content = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(repository), "update-ref", publication.RELEASE_BRANCH_REF, bad_content],
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "non-release apparatus content"):
                publication._validate_release_commit(
                    repository,
                    observed_commit=bad_content,
                    execution_release=execution_release,
                )

            subprocess.run(
                ["git", "-C", str(repository), "checkout", "-q", "-b", "bad-path", implementation],
                check=True,
            )
            _write_json(config_path, released)
            decision_path.write_text("bad path release\n", encoding="utf-8")
            outside = repository / "main" / "unsafe.py"
            outside.parent.mkdir(parents=True)
            outside.write_text("unsafe = True\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "bad path"],
                check=True,
            )
            bad_path = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(repository), "update-ref", publication.RELEASE_BRANCH_REF, bad_path],
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "non-release files"):
                publication._validate_release_commit(
                    repository,
                    observed_commit=bad_path,
                    execution_release=execution_release,
                )

            with (
                mock.patch.object(
                    publication,
                    "_git_output",
                    side_effect=(
                        bad_path,
                        bad_path,
                        f"{bad_path} {implementation} {'c' * 40}",
                    ),
                ),
                self.assertRaisesRegex(ValueError, "direct implementation child"),
            ):
                publication._validate_release_commit(
                    repository,
                    observed_commit=bad_path,
                    execution_release=execution_release,
                )

    def test_release_identity_checks_precede_every_side_effect_boundary(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        local_identity_checks = (
            'test "$RUN_ID" = "$REGISTERED_RUN_ID"',
            'test "$(git rev-parse HEAD)" = "$EXPECTED_RELEASE_COMMIT"',
            'git rev-parse refs/remotes/origin/agent/crfs-oracle-harness',
            'git rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT"',
            'release commit changed non-release apparatus content',
            'release decision is not the exact canonical appendix',
        )
        preflight = source.index("scripts/hpc/preflight.sh")
        ssh = source.index('ssh "$HOST" bash -s --')
        for fragment in local_identity_checks:
            with self.subTest(local=fragment):
                position = source.index(fragment)
                self.assertLess(position, preflight)
                self.assertLess(position, ssh)

        remote = source[source.index("set -euo pipefail", ssh) :]
        mkdir = remote.index('mkdir "$run_root"')
        first_sbatch = remote.index("sbatch --parsable --hold")
        for fragment in (
            "remote origin release ref mismatch",
            "remote release is not the direct implementation child",
            "remote release changed non-release apparatus content",
            "remote release decision is not the exact canonical appendix",
            "remote execution release contract changed",
            'test ! -e "$run_root"',
            "sampled-current canary not submitted: user queue is not empty",
            "worker-1 FreeMem=",
        ):
            with self.subTest(remote=fragment):
                position = remote.index(fragment)
                self.assertLess(position, mkdir)
                self.assertLess(position, first_sbatch)

        decision = publication.RELEASE_DECISION_PATH
        self.assertIn(decision, publication.BOUND_REPOSITORY_PATHS)
        self.assertIn(decision, source)
        for wrapper in (H100_WRAPPER, CPU_WRAPPER):
            wrapper_source = wrapper.read_text(encoding="utf-8")
            self.assertIn("execution_release.run_id", wrapper_source)
            self.assertIn("accepted_implementation_commit", wrapper_source)

    def test_builder_literal_keys_cover_every_closed_schema_object(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        tree = ast.parse(PUBLISHER.read_text(encoding="utf-8"))
        envelope = _assignment_dict(tree, "envelope")
        self.assertEqual(_dict_keys(envelope), set(schema["required"]))

        inline_objects = (
            "source_job",
            "allocation_tests",
            "scientific_payload",
            "semantic_validation",
            "telemetry",
            "simulator_use",
            "publication",
            "interpretation",
        )
        for name in inline_objects:
            with self.subTest(object=name):
                node = _dict_value(envelope, name)
                if isinstance(node, ast.Name):
                    node = _assignment_dict(tree, node.id)
                self.assertIsInstance(node, ast.Dict)
                self.assertEqual(
                    _dict_keys(node), set(schema["properties"][name]["required"])
                )

        telemetry = _dict_value(envelope, "telemetry")
        assert isinstance(telemetry, ast.Dict)
        gpu = _dict_value(telemetry, "gpu")
        assert isinstance(gpu, ast.Dict)
        self.assertEqual(_dict_keys(gpu), set(schema["$defs"]["gpuTelemetry"]["required"]))
        host = _function_return_dict(tree, "_host_envelope")
        self.assertEqual(_dict_keys(host), set(schema["$defs"]["hostTelemetry"]["required"]))

    def test_realistic_built_envelope_satisfies_the_closed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_id = "r05a-realistic-envelope-fixture"
            run_root = root / run_id
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            paths = {
                "payload": case_dir / "canary-payload.json",
                "host": case_dir / "host-cgroup-sampled-current.tsv",
                "gpu": case_dir / "gpu-memory-samples.csv",
                "tests": case_dir / "allocation-focused-tests.log",
                "contract": run_root / "source-contract.json",
                "submission": run_root / "submission.json",
                "result": case_dir / "results.json",
                "receipt": run_root / "cpu-afterany-validation.json",
            }
            for key in ("host", "gpu", "tests", "contract", "submission"):
                paths[key].write_text(key + "\n", encoding="utf-8")
            commit = "b" * 40
            simulator = {
                "setup": "one_reset_plus_20_dummy_settle_control_steps",
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
            }
            apparatus = {
                key: (True if key != "memory_pending" else True)
                for key in publication.PRE_MEMORY_APPARATUS_KEYS
            }
            result_without_memory = {
                "run_id": run_id,
                "case_id": publication.CASE_ID,
                "status": "completed_nonconverged",
                "config_hash": publication.FROZEN_BINDINGS[
                    "scientific_config_projection_sha256"
                ],
                "provenance": {
                    "git_commit": commit,
                    "git_dirty": False,
                    "slurm_job_id": "123",
                    "slurm_array_job_id": "123",
                    "slurm_array_task_id": "0",
                    "partition": "main",
                    "host": "worker-1",
                    "allocation_gpu_uuid": "GPU-test-uuid",
                },
                "apparatus": apparatus,
                "solver": {"first": {"trace": {}}, "duplicate": {"trace": {}}},
                "pairing": {"checks": {"exact": True}},
                "simulator_use": simulator,
            }
            payload = {
                "schema_version": "1.0",
                "payload_type": "r05a_inverse_flow_allocation_canary_payload",
                "complete_result_requires_memory_finalization": True,
                "result_without_memory": result_without_memory,
            }
            _write_json(paths["payload"], payload)
            contract = {
                "run_id": run_id,
                "git_commit": commit,
                "repository_file_sha256": {
                    publication.APPARATUS_CONFIG_PATH: publication.file_sha256(
                        APPARATUS_CONFIG
                    ),
                    publication.ENVELOPE_SCHEMA_PATH: publication.file_sha256(SCHEMA),
                },
            }
            host_summary = {
                "kernel_osrelease": "5.15.0-130-generic",
                "membership_path": "/slurm/job_123/step_batch/user/task_0",
                "mount_root": "/",
                "mount_point": "/sys/fs/cgroup",
                "job_scope_path": "/sys/fs/cgroup/slurm/job_123",
                "raw_trace_path": str(paths["host"]),
                "raw_trace_sha256": publication.file_sha256(paths["host"]),
                "sample_interval_requested_ms": 100,
                "sample_count": 2,
                "first_sample_monotonic_ns": 1_000_000_000,
                "last_sample_monotonic_ns": 1_100_000_000,
                "maximum_adjacent_gap_ns": 100_000_000,
                "host_cgroup_sampled_current_high_water_bytes": 6_000_000,
                "memory_max_before_bytes": 68_719_476_736,
                "memory_max_after_bytes": 68_719_476_736,
                "memory_events_before": {"max": 0, "oom": 0, "oom_kill": 0},
                "memory_events_after": {"max": 0, "oom": 0, "oom_kill": 0},
                "memory_events_deltas": {"max": 0, "oom": 0, "oom_kill": 0},
                "native_memory_peak_state": "missing",
                "native_memory_peak_value_bytes": None,
                "lifecycle_monotonic_ns": {
                    "monitor_ready": 1_000_000_001,
                    "policy_launch": 1_000_000_002,
                    "policy_cleanup_complete": 1_000_000_003,
                    "gpu_monitor_cleanup_complete": 1_000_000_004,
                    "workload_cleanup_complete": 1_000_000_005,
                    "monitor_stop_observed": 1_000_000_006,
                },
                "contract_passed": True,
            }
            counts = {
                "test_inverse_flow_control.py": 17,
                "test_inverse_flow_sampler.py": 8,
                "test_inverse_flow_policy.py": 10,
                "test_r05a_canary.py": 12,
            }
            fake_legacy = types.ModuleType(f"{_PACKAGE_NAME}.r05a_canary")
            fake_legacy.ALLOCATION_TEST_COUNTS = counts
            fake_legacy.ALLOCATION_TEST_REGISTRY_SHA256 = (
                "89bf8a509dafefcce2bd18cd6cf8e0233a8728165b6f033adc7f39a45c7e1432"
            )
            fake_legacy.EXPECTED_RESULT_STATUSES = {
                "completed_converged",
                "completed_nonconverged",
                "completed_apparatus_failure",
            }
            fake_legacy._parse_allocation_test_log = lambda path: (
                publication.file_sha256(path),
                counts,
            )
            fake_legacy._read_gpu_samples = lambda path: (
                2,
                "GPU-test-uuid",
                100.0,
                200.0,
                publication.file_sha256(path),
            )
            fake_legacy.validate_r05a_canary_result = lambda _value: list(
                publication.LEGACY_MEMORY_ONLY_ERRORS
            )
            fake_legacy._trace_from_record = lambda _value, name: {"name": name}
            fake_legacy._artifact_cuda_memory_peaks = lambda *_traces: (1024, 2048)

            original_loader = publication._load_hashed_object

            def released_config_loader(path, *, label):
                if Path(path) == APPARATUS_CONFIG:
                    config = _released_apparatus_config(
                        run_id=run_id,
                        accepted_implementation_commit="a" * 40,
                    )
                    return config, publication.file_sha256(APPARATUS_CONFIG)
                return original_loader(path, label=label)

            def git_result(arguments, **_kwargs):
                return SimpleNamespace(
                    stdout=(commit + "\n" if "rev-parse" in arguments else "")
                )

            qualified_legacy = f"{_PACKAGE_NAME}.r05a_canary"
            previous_legacy = sys.modules.get(qualified_legacy)
            previous_attribute = getattr(_PACKAGE, "r05a_canary", None)
            sys.modules[qualified_legacy] = fake_legacy
            setattr(_PACKAGE, "r05a_canary", fake_legacy)
            try:
                with (
                    mock.patch.object(
                        publication,
                        "_validate_source_contract",
                        return_value=(contract, run_root, case_dir),
                    ),
                    mock.patch.object(
                        publication,
                        "_validate_submission",
                        return_value=(
                            {},
                            publication.file_sha256(paths["submission"]),
                        ),
                    ),
                    mock.patch.object(
                        publication,
                        "parse_full_lifetime_telemetry",
                        return_value=host_summary,
                    ),
                    mock.patch.object(
                        publication, "_load_hashed_object", side_effect=released_config_loader
                    ),
                    mock.patch.object(publication.subprocess, "run", side_effect=git_result),
                    mock.patch.object(
                        publication,
                        "_validate_release_commit",
                        return_value=frozenset(publication.RELEASE_ONLY_PATHS[:2]),
                    ),
                    mock.patch.object(publication, "_validate_schema", return_value=[]),
                ):
                    envelope = publication.build_sampled_current_envelope(
                        payload_path=paths["payload"],
                        host_telemetry_path=paths["host"],
                        gpu_samples_path=paths["gpu"],
                        allocation_tests_log=paths["tests"],
                        source_contract_path=paths["contract"],
                        expected_source_contract_sha256="c" * 64,
                        submission_path=paths["submission"],
                        source_job_id="123",
                        source_job_state="COMPLETED",
                        source_exit_code="0:0",
                        publisher_job_id="456",
                        result_path=paths["result"],
                        receipt_path=paths["receipt"],
                        repo_root=ROOT,
                    )
            finally:
                if previous_legacy is None:
                    sys.modules.pop(qualified_legacy, None)
                else:
                    sys.modules[qualified_legacy] = previous_legacy
                if previous_attribute is None:
                    delattr(_PACKAGE, "r05a_canary")
                else:
                    setattr(_PACKAGE, "r05a_canary", previous_attribute)

            schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
            self.assertEqual(_schema_subset_errors(envelope, schema, schema), [])

    def test_python_and_both_shell_source_binding_sets_match(self) -> None:
        python_paths = set(publication.BOUND_REPOSITORY_PATHS)

        shell = SUBMITTER.read_text(encoding="utf-8")
        blocks = re.findall(
            r"(?:BOUND_REPOSITORY_PATHS|bound_paths)=\(\n(.*?)\n\)",
            shell,
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            shell_paths = {line.strip() for line in block.splitlines() if line.strip()}
            self.assertEqual(shell_paths, python_paths)

    def test_real_source_and_submission_contracts_pass_then_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _source_contract_fixture(Path(directory))
            contract_path = fixture["contract_path"]
            source_job_id = fixture["source_job_id"]
            contract_sha = publication.file_sha256(contract_path)
            with mock.patch.object(
                publication, "EXPERIMENT_ROOT", fixture["experiment_root"]
            ):
                contract, run_root, case_dir = publication._validate_source_contract(
                    contract_path,
                    expected_sha256=contract_sha,
                    repo_root=ROOT,
                    source_job_id=source_job_id,
                )
            self.assertEqual(contract["run_id"], fixture["contract"]["run_id"])
            self.assertEqual(run_root, fixture["run_root"])
            self.assertEqual(case_dir, fixture["case_dir"])

            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            submission_path = run_root / "submission.json"
            submission = {
                "schema_version": "2.0",
                "artifact_role": "r05a_sampled_current_canary_atomic_submission",
                "status": "cpu_afterany_registered_gpu_held",
                "run_id": contract["run_id"],
                "git_commit": contract["git_commit"],
                "source_node": "worker-1",
                "gpu_slurm_array_job_id": source_job_id,
                "gpu_slurm_array_task_id": 0,
                "cpu_afterany_job_id": "456",
                "dependency": f"afterany:{source_job_id}",
                "source_contract_path": str(contract_path),
                "source_contract_sha256": contract_sha,
                "expected_result": str(result),
                "expected_validation_receipt": str(receipt),
                "scientific_claim_allowed": False,
                "probe_training_authorized": False,
                "timestamp_utc": "2026-07-15T00:00:02Z",
            }
            _write_json(submission_path, submission)
            observed, submission_sha = publication._validate_submission(
                submission_path,
                contract=contract,
                contract_path=contract_path,
                contract_sha256=contract_sha,
                source_job_id=source_job_id,
                publisher_job_id="456",
                result_path=result,
                receipt_path=receipt,
            )
            self.assertEqual(observed, submission)
            self.assertEqual(submission_sha, publication.file_sha256(submission_path))

            with mock.patch.object(
                publication, "EXPERIMENT_ROOT", fixture["experiment_root"]
            ), self.assertRaisesRegex(ValueError, "externally supplied SHA-256"):
                publication._validate_source_contract(
                    contract_path,
                    expected_sha256="0" * 64,
                    repo_root=ROOT,
                    source_job_id=source_job_id,
                )

            bad_submission = dict(submission)
            bad_submission["cpu_afterany_job_id"] = "457"
            _write_json(submission_path, bad_submission)
            with self.assertRaisesRegex(
                ValueError, "atomic submission receipt cpu_afterany_job_id changed"
            ):
                publication._validate_submission(
                    submission_path,
                    contract=contract,
                    contract_path=contract_path,
                    contract_sha256=contract_sha,
                    source_job_id=source_job_id,
                    publisher_job_id="456",
                    result_path=result,
                    receipt_path=receipt,
                )

    def test_coupled_resource_and_held_receipt_tampering_still_fails(self) -> None:
        mutations = ("resource", "held")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = _source_contract_fixture(Path(directory))
                contract = dict(fixture["contract"])
                if mutation == "resource":
                    contract["resources"] = {
                        **contract["resources"],
                        "host_memory_mib": 131072,
                    }
                else:
                    held = dict(fixture["held"])
                    held["source_node"] = "worker-2"
                    _write_json(fixture["held_path"], held)
                    contract["held_gpu_submission_sha256"] = publication.file_sha256(
                        fixture["held_path"]
                    )
                _write_json(fixture["contract_path"], contract)
                with mock.patch.object(
                    publication, "EXPERIMENT_ROOT", fixture["experiment_root"]
                ), self.assertRaises(ValueError):
                    publication._validate_source_contract(
                        fixture["contract_path"],
                        expected_sha256=publication.file_sha256(
                            fixture["contract_path"]
                        ),
                        repo_root=ROOT,
                        source_job_id=fixture["source_job_id"],
                    )

    def test_symlinked_artifact_and_run_root_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = root / "payload.json"
            outside = root / "outside.json"
            outside.write_text("outside\n", encoding="utf-8")
            expected.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "non-symlink regular file"):
                publication._resolve_exact(expected, expected, label="payload")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            experiment_root = root / "experiments"
            outside_run = root / "outside-run"
            outside_case = outside_run / publication.CASE_ID
            outside_case.mkdir(parents=True)
            run_id = "r05a-symlink-run"
            contract_path = outside_run / "source-contract.json"
            _write_json(contract_path, {"run_id": run_id})
            experiment_root.mkdir()
            (experiment_root / run_id).symlink_to(outside_run, target_is_directory=True)
            linked_contract = experiment_root / run_id / "source-contract.json"
            with mock.patch.object(
                publication, "EXPERIMENT_ROOT", experiment_root
            ), self.assertRaisesRegex(ValueError, "non-symlink directory"):
                publication._validate_source_contract(
                    linked_contract,
                    expected_sha256=publication.file_sha256(linked_contract),
                    repo_root=ROOT,
                    source_job_id="123",
                )

    def test_receipt_write_failure_rolls_back_new_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_root = root / "run"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            payload = case_dir / "payload.json"
            tests_log = case_dir / "tests.log"
            host = case_dir / "host.tsv"
            gpu = case_dir / "gpu.csv"
            source = run_root / "source.json"
            submission = run_root / "submission.json"
            for path in (payload, tests_log, host, gpu, source, submission):
                path.write_text(path.name + "\n", encoding="utf-8")
            envelope = {
                "scientific_payload": {
                    "path": str(payload),
                    "sha256": publication.file_sha256(payload),
                },
                "allocation_tests": {
                    "log_path": str(tests_log),
                    "log_sha256": publication.file_sha256(tests_log),
                },
                "telemetry": {
                    "host": {
                        "raw_trace_path": str(host),
                        "raw_trace_sha256": publication.file_sha256(host),
                    },
                    "gpu": {
                        "raw_samples_path": str(gpu),
                        "raw_samples_sha256": publication.file_sha256(gpu),
                    },
                },
                "publication": {
                    "source_contract_receipt_path": str(source),
                    "source_contract_receipt_sha256": publication.file_sha256(source),
                    "atomic_submission_receipt_path": str(submission),
                    "atomic_submission_receipt_sha256": publication.file_sha256(
                        submission
                    ),
                },
            }
            original_write = publication._atomic_write_json

            def write_or_fail(path, value):
                if Path(path) == receipt:
                    raise OSError("simulated receipt failure")
                return original_write(path, value)

            keyword = {
                "payload_path": payload,
                "host_telemetry_path": host,
                "gpu_samples_path": gpu,
                "allocation_tests_log": tests_log,
                "source_contract_path": source,
                "expected_source_contract_sha256": "a" * 64,
                "submission_path": submission,
                "source_job_id": "1",
                "source_job_state": "COMPLETED",
                "source_exit_code": "0:0",
                "publisher_job_id": "2",
                "result_path": result,
                "receipt_path": receipt,
                "repo_root": root,
            }
            with (
                mock.patch.object(
                    publication,
                    "build_sampled_current_envelope",
                    return_value=envelope,
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
                mock.patch.object(publication, "_atomic_write_json", side_effect=write_or_fail),
                self.assertRaisesRegex(OSError, "simulated receipt failure"),
            ):
                publication.publish_sampled_current_envelope(**keyword)
            self.assertFalse(result.exists())
            self.assertFalse(receipt.exists())
            self.assertFalse((case_dir / ".results.candidate.json").exists())

    def test_candidate_swap_at_final_rename_cannot_publish_or_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_root = root / "run"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            candidate = case_dir / ".results.candidate.json"
            paths = {
                name: (
                    run_root / name
                    if name in {"source.json", "submission.json"}
                    else case_dir / name
                )
                for name in (
                    "payload.json",
                    "tests.log",
                    "host.tsv",
                    "gpu.csv",
                    "source.json",
                    "submission.json",
                )
            }
            for path in paths.values():
                path.write_text(path.name + "\n", encoding="utf-8")
            envelope = {
                "scientific_payload": {
                    "path": str(paths["payload.json"]),
                    "sha256": publication.file_sha256(paths["payload.json"]),
                },
                "allocation_tests": {
                    "log_path": str(paths["tests.log"]),
                    "log_sha256": publication.file_sha256(paths["tests.log"]),
                },
                "telemetry": {
                    "host": {
                        "raw_trace_path": str(paths["host.tsv"]),
                        "raw_trace_sha256": publication.file_sha256(paths["host.tsv"]),
                    },
                    "gpu": {
                        "raw_samples_path": str(paths["gpu.csv"]),
                        "raw_samples_sha256": publication.file_sha256(paths["gpu.csv"]),
                    },
                },
                "publication": {
                    "source_contract_receipt_path": str(paths["source.json"]),
                    "source_contract_receipt_sha256": publication.file_sha256(
                        paths["source.json"]
                    ),
                    "atomic_submission_receipt_path": str(paths["submission.json"]),
                    "atomic_submission_receipt_sha256": publication.file_sha256(
                        paths["submission.json"]
                    ),
                },
            }
            real_replace = os.replace

            def swap_candidate_then_replace(source, destination):
                if Path(source) == candidate and Path(destination) == result:
                    candidate.write_text('{"tampered":true}\n', encoding="utf-8")
                return real_replace(source, destination)

            keyword = {
                "payload_path": paths["payload.json"],
                "host_telemetry_path": paths["host.tsv"],
                "gpu_samples_path": paths["gpu.csv"],
                "allocation_tests_log": paths["tests.log"],
                "source_contract_path": paths["source.json"],
                "expected_source_contract_sha256": "a" * 64,
                "submission_path": paths["submission.json"],
                "source_job_id": "1",
                "source_job_state": "COMPLETED",
                "source_exit_code": "0:0",
                "publisher_job_id": "2",
                "result_path": result,
                "receipt_path": receipt,
                "repo_root": root,
            }
            with (
                mock.patch.object(
                    publication,
                    "build_sampled_current_envelope",
                    return_value=envelope,
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
                mock.patch.object(
                    publication.os, "replace", side_effect=swap_candidate_then_replace
                ),
                self.assertRaisesRegex(
                    ValueError, "published result (identity|bytes) differs"
                ),
            ):
                publication.publish_sampled_current_envelope(**keyword)
            self.assertFalse(result.exists())
            self.assertFalse(receipt.exists())
            self.assertFalse(candidate.exists())

    def test_case_directory_swap_at_final_rename_cannot_escape_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_root = root / "run"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            moved_case = root / "moved-case"
            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            candidate = case_dir / ".results.candidate.json"
            paths = {
                name: (
                    run_root / name
                    if name in {"source.json", "submission.json"}
                    else case_dir / name
                )
                for name in (
                    "payload.json",
                    "tests.log",
                    "host.tsv",
                    "gpu.csv",
                    "source.json",
                    "submission.json",
                )
            }
            for path in paths.values():
                path.write_text(path.name + "\n", encoding="utf-8")
            envelope = {
                "scientific_payload": {
                    "path": str(paths["payload.json"]),
                    "sha256": publication.file_sha256(paths["payload.json"]),
                },
                "allocation_tests": {
                    "log_path": str(paths["tests.log"]),
                    "log_sha256": publication.file_sha256(paths["tests.log"]),
                },
                "telemetry": {
                    "host": {
                        "raw_trace_path": str(paths["host.tsv"]),
                        "raw_trace_sha256": publication.file_sha256(paths["host.tsv"]),
                    },
                    "gpu": {
                        "raw_samples_path": str(paths["gpu.csv"]),
                        "raw_samples_sha256": publication.file_sha256(paths["gpu.csv"]),
                    },
                },
                "publication": {
                    "source_contract_receipt_path": str(paths["source.json"]),
                    "source_contract_receipt_sha256": publication.file_sha256(
                        paths["source.json"]
                    ),
                    "atomic_submission_receipt_path": str(paths["submission.json"]),
                    "atomic_submission_receipt_sha256": publication.file_sha256(
                        paths["submission.json"]
                    ),
                },
            }
            real_replace = os.replace

            def swap_case_then_replace(source, destination):
                if Path(source) == candidate and Path(destination) == result:
                    case_dir.rename(moved_case)
                    case_dir.symlink_to(moved_case, target_is_directory=True)
                return real_replace(source, destination)

            keyword = {
                "payload_path": paths["payload.json"],
                "host_telemetry_path": paths["host.tsv"],
                "gpu_samples_path": paths["gpu.csv"],
                "allocation_tests_log": paths["tests.log"],
                "source_contract_path": paths["source.json"],
                "expected_source_contract_sha256": "a" * 64,
                "submission_path": paths["submission.json"],
                "source_job_id": "1",
                "source_job_state": "COMPLETED",
                "source_exit_code": "0:0",
                "publisher_job_id": "2",
                "result_path": result,
                "receipt_path": receipt,
                "repo_root": root,
            }
            with (
                mock.patch.object(
                    publication,
                    "build_sampled_current_envelope",
                    return_value=envelope,
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
                mock.patch.object(
                    publication.os, "replace", side_effect=swap_case_then_replace
                ),
                self.assertRaisesRegex(
                    ValueError, "publication case directory must be a non-symlink"
                ),
            ):
                publication.publish_sampled_current_envelope(**keyword)
            self.assertTrue(case_dir.is_symlink())
            self.assertFalse((moved_case / "results.json").exists())
            self.assertFalse(receipt.exists())

    def test_host_trace_is_rehashed_immediately_after_same_byte_parse(self) -> None:
        source = PUBLISHER.read_text(encoding="utf-8")
        parse = source.index("host_summary = parse_full_lifetime_telemetry(")
        rehash = source.index(
            'file_sha256(resolved["host telemetry"]) '
            '!= host_summary["raw_trace_sha256"]',
            parse,
        )
        payload = source.index("payload, payload_sha256 = _load_hashed_object(", rehash)
        self.assertLess(parse, rehash)
        self.assertLess(rehash, payload)

    def test_final_fingerprint_check_rejects_post_candidate_submission_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_root = root / "run"
            case_dir = run_root / publication.CASE_ID
            case_dir.mkdir(parents=True)
            paths = {
                name: (
                    run_root / name
                    if name in {"source.json", "submission.json"}
                    else case_dir / name
                )
                for name in (
                    "payload.json",
                    "tests.log",
                    "host.tsv",
                    "gpu.csv",
                    "source.json",
                    "submission.json",
                )
            }
            for path in paths.values():
                path.write_text(path.name + "\n", encoding="utf-8")
            envelope = {
                "scientific_payload": {
                    "path": str(paths["payload.json"]),
                    "sha256": publication.file_sha256(paths["payload.json"]),
                },
                "allocation_tests": {
                    "log_path": str(paths["tests.log"]),
                    "log_sha256": publication.file_sha256(paths["tests.log"]),
                },
                "telemetry": {
                    "host": {
                        "raw_trace_path": str(paths["host.tsv"]),
                        "raw_trace_sha256": publication.file_sha256(paths["host.tsv"]),
                    },
                    "gpu": {
                        "raw_samples_path": str(paths["gpu.csv"]),
                        "raw_samples_sha256": publication.file_sha256(paths["gpu.csv"]),
                    },
                },
                "publication": {
                    "source_contract_receipt_path": str(paths["source.json"]),
                    "source_contract_receipt_sha256": publication.file_sha256(
                        paths["source.json"]
                    ),
                    "atomic_submission_receipt_path": str(paths["submission.json"]),
                    "atomic_submission_receipt_sha256": publication.file_sha256(
                        paths["submission.json"]
                    ),
                },
            }
            calls = 0

            def build_then_mutate(**_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    paths["submission.json"].write_text(
                        "mutated after candidate\n", encoding="utf-8"
                    )
                return envelope

            result = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            with (
                mock.patch.object(
                    publication,
                    "build_sampled_current_envelope",
                    side_effect=build_then_mutate,
                ),
                mock.patch.object(publication, "_validate_schema", return_value=[]),
                self.assertRaisesRegex(
                    ValueError, "atomic submission receipt changed after candidate creation"
                ),
            ):
                publication.publish_sampled_current_envelope(
                    payload_path=paths["payload.json"],
                    host_telemetry_path=paths["host.tsv"],
                    gpu_samples_path=paths["gpu.csv"],
                    allocation_tests_log=paths["tests.log"],
                    source_contract_path=paths["source.json"],
                    expected_source_contract_sha256="a" * 64,
                    submission_path=paths["submission.json"],
                    source_job_id="1",
                    source_job_state="COMPLETED",
                    source_exit_code="0:0",
                    publisher_job_id="2",
                    result_path=result,
                    receipt_path=receipt,
                    repo_root=root,
                )
            self.assertFalse(result.exists())
            self.assertFalse(receipt.exists())
            self.assertFalse((case_dir / ".results.candidate.json").exists())


if __name__ == "__main__":
    unittest.main()
