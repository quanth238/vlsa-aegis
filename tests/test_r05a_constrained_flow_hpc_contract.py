from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "scripts/hpc/run_r05a_constrained_flow_workload.sh"
RUNTIME_IDENTITY_HELPER = ROOT / "scripts/hpc/lib/r05a_runtime_identity.sh"
GPU_WRAPPER = ROOT / "scripts/hpc/run_r05a_constrained_flow_canary.sh"
CPU_WRAPPER = ROOT / "scripts/hpc/validate_r05a_constrained_flow_canary.sh"
SUBMITTER = ROOT / "scripts/hpc/submit_r05a_constrained_flow_canary.sh"
PREFLIGHT = ROOT / "scripts/hpc/preflight.sh"
TRANSFORMERS_OVERLAY_HELPER = ROOT / "scripts/hpc/prepare_transformers_overlay.sh"
GPU_SBATCH = ROOT / "slurm/r05a_constrained_flow_canary_h100.sbatch"
CPU_SBATCH = ROOT / "slurm/r05a_constrained_flow_canary_validate_cpu.sbatch"
APPARATUS = ROOT / "configs/experiments/r05a_constrained_flow_canary_apparatus.json"
SCIENTIFIC = ROOT / "configs/experiments/r05a_constrained_flow_canary.json"

RESOURCE = {
    "partition": "main",
    "account": "normal",
    "qos": "normal",
    "source_host": "worker-1",
    "gpus": 1,
    "cpus_per_task": 8,
    "host_memory_mib": 65536,
    "time_limit": "02:00:00",
    "array": "0-0%1",
    "requeue": False,
    "validator_partition": "main",
    "validator_account": "normal",
    "validator_qos": "normal",
    "validator_cpus": 2,
    "validator_host_memory_mib": 8192,
    "validator_time_limit": "00:15:00",
    "validator_gpus": 0,
    "validator_dependency": "afterany",
}
RELEASE_PATHS = [
    "configs/experiments/r05a_constrained_flow_canary.json",
    "configs/experiments/r05a_constrained_flow_canary_apparatus.json",
    "docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md",
]
SCIENTIFIC_PROJECTION_SHA256 = (
    "7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915"
)
VINUNI_GUIDE_SHA256 = (
    "acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConstrainedFlowHPCContractTest(unittest.TestCase):
    def test_new_shell_entrypoints_are_executable_and_parse(self) -> None:
        paths = [
            TRANSFORMERS_OVERLAY_HELPER,
            RUNTIME_IDENTITY_HELPER,
            WORKLOAD,
            GPU_WRAPPER,
            CPU_WRAPPER,
            SUBMITTER,
            PREFLIGHT,
            GPU_SBATCH,
            CPU_SBATCH,
        ]
        for path in paths:
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_gpu_and_cpu_resources_are_exact_and_source_pinned(self) -> None:
        gpu = _text(GPU_SBATCH)
        for directive in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=02:00:00",
            "#SBATCH --array=0-0%1",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(directive, gpu)
        cpu = _text(CPU_SBATCH)
        for directive in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=8G",
            "#SBATCH --time=00:15:00",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(directive, cpu)
        self.assertNotIn("--gres", cpu)
        submit = _text(SUBMITTER)
        self.assertIn("--nodelist=worker-1", submit)
        self.assertIn("--dependency=\"$dependency\"", submit)
        self.assertIn("dependency=afterany:$gpu_job_id", submit)
        self.assertNotIn("worker-2", submit + _text(GPU_WRAPPER) + _text(WORKLOAD))

    def test_scientific_projection_and_core_method_bytes_are_frozen(self) -> None:
        scientific = json.loads(_text(SCIENTIFIC))
        projection = json.loads(json.dumps(scientific))
        for key in ("config_status", "ready_to_run", "blocked_on", "execution_release"):
            projection.pop(key, None)
        projection["preregistration"].pop("h100_submission_authorized", None)
        canonical = json.dumps(
            projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), SCIENTIFIC_PROJECTION_SHA256)

        expected = {
            "main/crfs_oracle/r05a_constrained_flow_canary.py": "a2365d5d97210779a4fbc6d69c97dd27629981d4f6caceffc71d95c5395f427a",
            "openpi/src/openpi/models_pytorch/crfs_linearized_control.py": "4c93bc55bfc266f40fb05d21f5610e8988418bcb0e2c1a472afdec9d50d01b59",
            "openpi/src/openpi/policies/crfs_constrained_flow_adapter.py": "ee7ff8e52757336438823214809b53cc2b9c6c66355f83d5e860666586d9590f",
            "openpi/scripts/serve_cfs_policy.py": "8a919e61e49415bec6e5383be251c6a8b26f4299e1372972ccfc6e06d17011c5",
            "main/run_crfs_r05a_constrained_flow_canary.py": "d11f0b7893bb8fd27a269a772d04df2b71c6e323549117382b602ea4009c162a",
        }
        for relative, digest in expected.items():
            self.assertEqual(_sha(ROOT / relative), digest, relative)

        self.assertEqual(scientific["frozen_case"]["case_id"], "crfs-1069f29a8d76463a")
        self.assertEqual(scientific["target_contract"]["source_budget_float32"], 3.6398398876190186)
        self.assertEqual(scientific["flow_contract"]["active_steps"], [5, 6, 7, 8, 9])
        self.assertEqual(scientific["flow_contract"]["independent_variable_count"], 75)
        self.assertEqual(scientific["arms"]["A_historical_run_b"]["max_updates"], 128)
        self.assertEqual(scientific["linearized_solver"]["fixed_update_count"], 4096)
        self.assertEqual(scientific["execution_boundary"]["policy_generated_action_steps_executed"], 0)
        self.assertEqual(scientific["execution_boundary"]["teacher_generated_action_steps_executed"], 0)

    def test_full_vinuni_guide_controls_are_bound_and_fail_closed(self) -> None:
        apparatus = json.loads(_text(APPARATUS))
        guide = apparatus["vinuni_h100_guide_contract"]
        self.assertEqual(guide["sha256"], VINUNI_GUIDE_SHA256)
        self.assertEqual(guide["line_count"], 1298)
        self.assertEqual(guide["login_node_role"], "control_plane_only")
        self.assertTrue(guide["allocation_compute_only"])
        self.assertFalse(guide["free_h100_required_before_submission"])
        self.assertTrue(guide["pending_submission_allowed"])
        self.assertFalse(guide["reroute_when_worker_1_busy"])
        self.assertEqual(guide["shared_storage_stop_percent"], 90)

        preflight = _text(PREFLIGHT)
        for token in (
            "[partition-main-detail]",
            "scontrol show partition main -o",
            "[worker-1-detail]",
            "scontrol show node worker-1 -o",
            "sacctmgr -n -P show assoc",
            "df -h /mnt/data",
            "tmux|screen|python|train\\.py|render\\.py|metrics\\.py|torch|cuda|serve",
            "jupyter|vllm|ollama",
        ):
            self.assertIn(token, preflight)
        self.assertIn('if [ "$storage_percent" -ge 90 ]', preflight)

        submit = _text(SUBMITTER)
        for token in (
            VINUNI_GUIDE_SHA256,
            "wc -l",
            "MaxTRESPerJob,MaxTRESPU",
            "show assoc",
            "require_two_hour_capacity",
            "scontrol show partition main -o",
            "CfgTRES=",
            "AllocTRES=",
            "free_h100",
            "immediate_h100_capacity_available",
            "pending_submission_allowed:true",
            "/mnt/data/quanth/slurm_logs/crfs-oracle",
            "--export=ALL",
            "inherited SBATCH options could change the exact transaction",
            "ArrayTaskThrottle=1",
            "CPU publisher unexpectedly became an array",
        ):
            self.assertIn(token, submit)
        self.assertNotIn('test "$free_h100" -ge 1', submit)
        self.assertIn('test "$allocated_gpus" -le "$configured_gpus"', submit)
        remote = submit.split("<<'REMOTE'", 1)[1]
        self.assertNotIn('sha256sum "$checkpoint"', remote)
        self.assertIn('sha256sum "$MODEL"', _text(WORKLOAD))

    def test_live_limit_parsers_accept_unlimited_and_reject_small_caps(self) -> None:
        remote = _text(SUBMITTER).split("<<'REMOTE'", 1)[1]
        helpers = remote.split("remote_repo=$1", 1)[0]
        accepted = subprocess.run(
            [
                "bash",
                "-c",
                helpers
                + "\n"
                + "test \"$(slurm_duration_seconds 02:00:00)\" = 7200\n"
                + "test \"$(slurm_duration_seconds 1-08:09:10)\" = 115750\n"
                + "require_two_hour_capacity qos ''\n"
                + "require_two_hour_capacity partition UNLIMITED\n"
                + "require_tres_capacity qos 'cpu=16,gres/gpu=2,mem=256G' cpu 8 optional\n"
                + "require_tres_capacity qos 'cpu=16,gres/gpu=2,mem=256G' gres/gpu 1 optional\n"
                + "require_tres_capacity qos 'cpu=16,gres/gpu=2,mem=256G' mem 65536 optional\n"
                + "require_tres_capacity assoc '' mem 65536 optional\n",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        for command in (
            "require_two_hour_capacity qos 01:59:59",
            "require_tres_capacity qos 'cpu=7' cpu 8 optional",
            "require_tres_capacity qos 'gres/gpu=0' gres/gpu 1 optional",
            "require_tres_capacity qos 'mem=63G' mem 65536 optional",
        ):
            rejected = subprocess.run(
                ["bash", "-c", helpers + "\n" + command],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected.returncode, 0, command)

    def test_workload_uses_only_separate_cfs_server_and_new_client(self) -> None:
        value = _text(WORKLOAD)
        self.assertIn("scripts/serve_cfs_policy.py", value)
        self.assertNotIn("scripts/serve_policy.py", value)
        self.assertIn("main/run_crfs_r05a_constrained_flow_canary.py", value)
        self.assertIn('--config "$CFS_CONFIG"', value)
        self.assertIn('--legacy-config "$LEGACY_CONFIG"', value)
        self.assertIn("complete_comparison)", value)
        self.assertIn("terminal_apparatus_failure)", value)
        self.assertIn('test ! -e "$HIDDEN_CANDIDATE" && test ! -e "$RESULT"', value)

    def test_server_death_and_oom_cannot_be_clean_terminal_evidence(self) -> None:
        value = _text(WORKLOAD)
        client_end = value.index('>"$CLIENT_LOG" 2>&1')
        alive_check = value.index('kill -0 "$SERVER_PID"', client_end)
        variant_check = value.index("payload_variant=$(jq", alive_check)
        reviewed_kill = value.index(
            'crfs_stop_policy_server_exact_sigterm "$SERVER_PID"', variant_check
        )
        self.assertLess(client_end, alive_check)
        self.assertLess(alive_check, variant_check)
        self.assertLess(variant_check, reviewed_kill)
        self.assertIn("terminal CFS payload indicates an out-of-memory failure", value)
        self.assertIn("CFS policy server log indicates an out-of-memory failure", value)
        self.assertIn("out[ -]?of[ -]?memory", value)
        self.assertNotIn(
            'kill "$SERVER_PID" 2>/dev/null || true', value[client_end:]
        )
        self.assertNotIn('wait "$SERVER_PID" 2>/dev/null || true', value[client_end:])
        self.assertIn('if [ "$wait_status" -ne 143 ]', value)
        self.assertLess(
            value.index('crfs_stop_policy_server_exact_sigterm "$SERVER_PID"'),
            value.index('crfs_write_cgroup_v2_full_lifetime_marker "$POLICY_SERVER_CLEANUP_COMPLETE"'),
        )

        function = re.search(
            r"(?ms)^crfs_stop_policy_server_exact_sigterm\(\) \{.*?^\}", value
        )
        self.assertIsNotNone(function)
        body = function.group(0)
        accepted = subprocess.run(
            ["bash", "-c", body + "; sleep 30 & pid=$!; crfs_stop_policy_server_exact_sigterm \"$pid\""],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        for exit_status in (0, 137):
            rejected = subprocess.run(
                [
                    "bash",
                    "-c",
                    body
                    + f"; bash -c 'trap \"exit {exit_status}\" TERM; while :; do :; done' & pid=$!; "
                    + "sleep 0.1; crfs_stop_policy_server_exact_sigterm \"$pid\"",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("expected reviewed SIGTERM status 143", rejected.stderr)

    def test_transformers_overlay_rehashes_exact_cache_free_tree(self) -> None:
        helper = _text(TRANSFORMERS_OVERLAY_HELPER)
        for digest in (
            "430b00a688e12ff457cdd65929bd164fd001ffa1716dc83589d5388806d2bb33",
            "2e1b546bdf42e9872c84734b2d5baf52bd411664685922458e734732c8434098",
            "24be8ac6749a4cf7e19c261b14b39e951a499ec61b0602d0badcc4354171d261",
        ):
            self.assertIn(digest, helper)
        self.assertIn("validate_overlay", helper)
        self.assertIn("! -path '*/__pycache__/*' ! -name '*.pyc'", helper)
        self.assertIn("transformers-openpi-4.53.2-exact-24be8ac6749a", helper)
        self.assertIn('chmod -R a-w "$TMP"', helper)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "overlay"
            package = root / "transformers"
            package.mkdir(parents=True)
            source = package / "reviewed.py"
            source.write_text("reviewed = True\n", encoding="utf-8")

            manifest = (
                hashlib.sha256(source.read_bytes()).hexdigest()
                + "  transformers/reviewed.py\n"
            ).encode("utf-8")
            overlay_sha = hashlib.sha256(manifest).hexdigest()
            source_sha = "1" * 64
            replacement_sha = "2" * 64
            (root / ".source-bundle-sha256").write_text(source_sha + "\n", encoding="utf-8")
            (root / ".replacement-bundle-sha256").write_text(
                replacement_sha + "\n", encoding="utf-8"
            )
            (root / ".overlay-bundle-sha256").write_text(
                overlay_sha + "\n", encoding="utf-8"
            )

            command = [
                "bash",
                str(TRANSFORMERS_OVERLAY_HELPER),
                "--validate-overlay-fixture",
                str(root),
                source_sha,
                replacement_sha,
                overlay_sha,
            ]
            subprocess.run(command, check=True)
            source.write_text("mutated = True\n", encoding="utf-8")
            self.assertNotEqual(subprocess.run(command).returncode, 0)
            source.write_text("reviewed = True\n", encoding="utf-8")
            cache = package / "__pycache__"
            cache.mkdir()
            (cache / "reviewed.pyc").write_bytes(b"unreviewed bytecode")
            self.assertNotEqual(subprocess.run(command).returncode, 0)
            (cache / "reviewed.pyc").unlink()
            cache.rmdir()
            link = package / "redirect.py"
            link.symlink_to(package / "missing.py")
            self.assertNotEqual(subprocess.run(command).returncode, 0)

    def test_runtime_environment_cannot_redirect_cfs_allocations(self) -> None:
        workload = _text(WORKLOAD)
        publisher = _text(CPU_WRAPPER)
        workload_function = re.search(
            r"(?ms)^require_canonical_runtime_path\(\) \{.*?^\}", workload
        )
        publisher_function = re.search(
            r"(?ms)^require_canonical_runtime_path\(\) \{.*?^\}", publisher
        )
        self.assertIsNotNone(workload_function)
        self.assertIsNotNone(publisher_function)
        for function, variable, expected, error in (
            (
                workload_function.group(0),
                "OPENPI_PYTHON",
                "/mnt/data/quanth/venvs/openpi/bin/python",
                "noncanonical inherited CFS-00A runtime path",
            ),
            (
                workload_function.group(0),
                "LIBERO_PYTHON",
                "/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
                "noncanonical inherited CFS-00A runtime path",
            ),
            (
                publisher_function.group(0),
                "LIBERO_PYTHON",
                "/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
                "noncanonical inherited CFS-00A publisher runtime path",
            ),
        ):
            rejected = subprocess.run(
                [
                    "bash",
                    "-c",
                    function
                    + f"; {variable}=/tmp/unreviewed-python; "
                    + f"require_canonical_runtime_path {variable} {expected}",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(error, rejected.stderr)
        for variable, expected in (
            ("OPENPI_PYTHON", '"$CRFS_R05A_OPENPI_PYTHON"'),
            ("LIBERO_PYTHON", '"$CRFS_R05A_LIBERO_PYTHON"'),
            ("OPENPI_DATA_HOME", "/mnt/data/quanth/cache/openpi"),
            ("TRANSFORMERS_SITE_PACKAGES", "/mnt/data/quanth/venvs/openpi/lib/python3.11/site-packages"),
            ("TRANSFORMERS_OVERLAY", "/mnt/data/quanth/cache/crfs/transformers-openpi-4.53.2-exact-24be8ac6749a"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
        ):
            self.assertIn(
                f"require_canonical_runtime_path {variable} {expected}", workload
            )
        for variable, expected in (
            ("LIBERO_PYTHON", "/mnt/data/quanth/venvs/openpi-libero-client/bin/python"),
            ("JSONSCHEMA_SOURCE_SITE", "/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages"),
            ("JSONSCHEMA_OVERLAY", "/mnt/data/quanth/cache/crfs/jsonschema-4.23.0-py38"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
        ):
            self.assertIn(
                f"require_canonical_runtime_path {variable} {expected}", publisher
            )
        self.assertIn('test "$prepared_transformers_overlay" = "$TRANSFORMERS_OVERLAY"', workload)
        self.assertIn('test "$prepared_jsonschema_overlay" = "$JSONSCHEMA_OVERLAY"', publisher)

    def test_exact_interpreter_identity_accepts_only_reviewed_nonexecuting_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "interpreter-was-invoked"
            binary_bytes = (
                "#!/usr/bin/env bash\n"
                f": > {shlex.quote(str(marker))}\n"
            ).encode("utf-8")
            versioned_a = root / "versioned-a" / "bin" / "python"
            versioned_b = root / "versioned-b" / "bin" / "python"
            for binary in (versioned_a, versioned_b):
                binary.parent.mkdir(parents=True)
                binary.write_bytes(binary_bytes)
                binary.chmod(0o755)
            alias = root / "version-alias"
            alias.symlink_to(versioned_a.parent.parent, target_is_directory=True)
            alternate_alias = root / "alternate-alias"
            alternate_alias.symlink_to(versioned_a.parent.parent, target_is_directory=True)
            public = root / "launcher"
            direct = alias / "bin" / "python"
            public.symlink_to(direct)
            digest = hashlib.sha256(binary_bytes).hexdigest()

            def validate(
                *,
                expected_direct: Path = direct,
                expected_resolved: Path = versioned_a,
                expected_digest: str = digest,
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        '. "$1"; crfs_require_exact_interpreter_identity Fixture "$2" "$2" "$3" "$4" "$5"',
                        "bash",
                        str(RUNTIME_IDENTITY_HELPER),
                        str(public),
                        str(expected_direct),
                        str(expected_resolved.resolve()),
                        expected_digest,
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            accepted = validate()
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertFalse(marker.exists(), "identity validation invoked the interpreter")

            public.unlink()
            public.symlink_to(alternate_alias / "bin" / "python")
            self.assertNotEqual(validate().returncode, 0)

            public.unlink()
            public.symlink_to(direct)
            alias.unlink()
            alias.symlink_to(versioned_b.parent.parent, target_is_directory=True)
            self.assertNotEqual(validate().returncode, 0)

            alias.unlink()
            alias.symlink_to(versioned_a.parent.parent, target_is_directory=True)
            versioned_a.write_bytes(binary_bytes + b"# changed\n")
            versioned_a.chmod(0o755)
            self.assertNotEqual(validate().returncode, 0)

            versioned_a.write_bytes(binary_bytes)
            versioned_a.chmod(0o755)
            versioned_a.unlink()
            self.assertNotEqual(validate().returncode, 0)

            versioned_a.write_bytes(binary_bytes)
            versioned_a.chmod(0o644)
            self.assertNotEqual(validate().returncode, 0)

            versioned_a.chmod(0o755)
            public.unlink()
            public.write_bytes(binary_bytes)
            public.chmod(0o755)
            self.assertNotEqual(validate().returncode, 0)

            public.unlink()
            nonregular = root / "nonregular"
            nonregular.mkdir()
            public.symlink_to(nonregular)
            self.assertNotEqual(
                validate(expected_direct=nonregular, expected_resolved=nonregular).returncode,
                0,
            )
            self.assertFalse(marker.exists(), "a rejected identity invoked the interpreter")

    def test_runtime_identity_constants_match_terminal_evidence_and_consumers(self) -> None:
        expected = {
            "openpi_python": {
                "public_path": "/mnt/data/quanth/venvs/openpi/bin/python",
                "direct_link_target": "/mnt/data/quanth/anaconda3/bin/python",
                "resolved_executable": "/mnt/data/quanth/anaconda3/bin/python3.11",
                "resolved_sha256": "c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9",
            },
            "libero_python": {
                "public_path": "/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
                "direct_link_target": "/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8",
                "resolved_executable": "/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8",
                "resolved_sha256": "c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62",
            },
        }
        apparatus = json.loads(_text(APPARATUS))
        self.assertEqual(
            apparatus["runtime_identity_contract"],
            {
                "validation_helper": "scripts/hpc/lib/r05a_runtime_identity.sh",
                "validation_is_shell_only": True,
                "interpreter_invocation_during_validation_allowed": False,
                **expected,
            },
        )
        evidence_binding = {
            "decision_path": "docs/decisions/0045-accept-runtime-identity-regression.md",
            "decision_sha256": "3672cfac46d8ffbd5a224e837e871bdc7010884907f367b6e418471a68c1d98e",
            "evidence_path": "evidence/r05a/runtime-identity-regression-20260716a.json",
            "evidence_sha256": "ba83d7d696310456b696ddd0a846e5a6b0554b994204ecf7c568e2b565508237",
            "preflight_path": "evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt",
            "preflight_sha256": "c351ec194cf838829e82105f1343de242199e93a855b9ce45862b64a6b955221",
            "release_commit": "8415b659a46699757de1e99558713e56b95255b5",
            "job_id": "28043",
            "job_state": "COMPLETED",
            "job_exit_code": "0:0",
            "source_host": "worker-1",
            "result_sha256": "3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de",
            "shell_only": True,
            "gpus_allocated": 0,
            "cfs_runtime_integration_evaluated": False,
            "h100_submission_authorized_by_evidence": False,
        }
        self.assertEqual(apparatus["runtime_identity_evidence_binding"], evidence_binding)
        self.assertEqual(
            _sha(ROOT / evidence_binding["decision_path"]),
            evidence_binding["decision_sha256"],
        )
        self.assertEqual(
            _sha(ROOT / evidence_binding["evidence_path"]),
            evidence_binding["evidence_sha256"],
        )
        self.assertEqual(
            _sha(ROOT / evidence_binding["preflight_path"]),
            evidence_binding["preflight_sha256"],
        )
        evidence = json.loads(
            _text(ROOT / "evidence/r05a/cfs00a-same-budget-launch-a.json")
        )["failure"]
        for key, evidence_key in (
            ("openpi_python", "openpi_python"),
            ("libero_python", "libero_python"),
        ):
            self.assertEqual(evidence[evidence_key]["path"], expected[key]["public_path"])
            self.assertEqual(
                evidence[evidence_key]["link_target"], expected[key]["direct_link_target"]
            )
            self.assertEqual(
                evidence[evidence_key]["resolved_target"], expected[key]["resolved_executable"]
            )
            self.assertEqual(
                evidence[evidence_key]["resolved_target_sha256"], expected[key]["resolved_sha256"]
            )
        helper = _text(RUNTIME_IDENTITY_HELPER)
        for identity in expected.values():
            for value in identity.values():
                self.assertIn(value, helper)
        workload = _text(WORKLOAD)
        publisher = _text(CPU_WRAPPER)
        self.assertIn('crfs_validate_r05a_openpi_python "$OPENPI_PYTHON"', workload)
        self.assertIn('crfs_validate_r05a_libero_python "$LIBERO_PYTHON"', workload)
        self.assertIn('crfs_validate_r05a_libero_python "$LIBERO_PYTHON"', publisher)
        self.assertLess(
            workload.index('crfs_validate_r05a_openpi_python "$OPENPI_PYTHON"'),
            workload.index('mkdir "$CASE_DIR" "$FAILURE_DIR"'),
        )
        self.assertLess(
            publisher.index('crfs_validate_r05a_libero_python "$LIBERO_PYTHON"'),
            publisher.index('"$LIBERO_PYTHON" "$REMOTE_REPO/main/publish_crfs'),
        )
        for consumer in (
            _text(SUBMITTER),
            _text(GPU_WRAPPER),
            publisher,
            _text(ROOT / "main/crfs_oracle/r05a_constrained_flow_publication.py"),
        ):
            self.assertIn("28043", consumer)
            self.assertIn(evidence_binding["evidence_sha256"], consumer)
            self.assertIn(evidence_binding["preflight_sha256"], consumer)

    def test_interpreter_symlink_exception_does_not_weaken_immutable_inputs(self) -> None:
        workload = _text(WORKLOAD)
        match = re.search(
            r'(?ms)^for path in \\\n(?P<body>.*?)^done$', workload
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertNotIn('"$OPENPI_PYTHON"', body)
        self.assertNotIn('"$LIBERO_PYTHON"', body)
        for token in (
            '"$MANIFEST"',
            '"$CFS_CONFIG"',
            '"$LEGACY_CONFIG"',
            '"$R05A_SOURCE_CONTRACT"',
            '"$ALLOCATION_TEST_REGISTRY"',
            '"$SOURCE_R02"',
            '"$MODEL"',
            '"$RUNTIME_IDENTITY_HELPER"',
            '"$REMOTE_REPO/scripts/hpc/lib/r05a_allocation_tests.sh"',
            '"$REMOTE_REPO/scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh"',
            '"$REMOTE_REPO/main/run_crfs_r05a_constrained_flow_canary.py"',
            '"$REMOTE_REPO/openpi/scripts/serve_cfs_policy.py"',
        ):
            self.assertIn(token, body)
        self.assertIn(
            'test -f "$path" && test ! -L "$path"',
            body,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "regular"
            regular.write_text("bound\n", encoding="utf-8")
            link = root / "link"
            link.symlink_to(regular)
            check = 'test -f "$1" && test ! -L "$1"'
            self.assertEqual(subprocess.run(["bash", "-c", check, "bash", str(regular)]).returncode, 0)
            self.assertNotEqual(subprocess.run(["bash", "-c", check, "bash", str(link)]).returncode, 0)

    def test_raw_paths_and_cpu_only_publication_are_explicit(self) -> None:
        source = _text(SUBMITTER) + _text(GPU_WRAPPER)
        for token in (
            "legacy_payload",
            "constrained_flow_payload",
            "canary-payload.json",
            "constrained-flow-payload.json",
            "host-cgroup-sampled-current.tsv",
            "gpu-memory-samples.csv",
            "allocation-focused-tests.log",
            ".results.candidate.json",
            "results.json",
        ):
            self.assertIn(token, source)
        cpu = _text(CPU_WRAPPER)
        for flag in (
            "--legacy-payload",
            "--constrained-flow-payload",
            "--host-telemetry",
            "--gpu-samples",
            "--allocation-tests-log",
            "--source-contract",
            "--expected-source-contract-sha256",
            "--submission",
            "--source-job-id",
            "--source-job-state",
            "--source-exit-code",
            "--publisher-job-id",
            "--expected-git-commit",
            "--output",
            "--validation-receipt",
        ):
            self.assertIn(flag, cpu)
        self.assertIn('"${CUDA_VISIBLE_DEVICES:-NoDevFiles}" = NoDevFiles', cpu)

    def test_full_lifetime_and_allocation_helpers_are_reused_unchanged(self) -> None:
        value = _text(WORKLOAD)
        self.assertIn("scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh", value)
        self.assertIn("crfs_monitor_cgroup_v2_full_lifetime", value)
        self.assertIn("crfs_write_cgroup_v2_full_lifetime_marker", value)
        self.assertIn("scripts/hpc/lib/r05a_allocation_tests.sh", value)
        self.assertIn("crfs_run_r05a_allocation_tests", value)
        self.assertIn("r05a_constrained_flow_allocation_tests.json", value)
        self.assertNotIn("expected_tests=", value)
        self.assertNotRegex(value, r"(?:TOTAL|COUNT|TESTS)[A-Z_]*=(?:85|89|90)\b")

    def test_transaction_is_held_receipted_cpu_registered_then_released_once(self) -> None:
        value = _text(SUBMITTER)
        ordered = [
            "sbatch --parsable --hold",
            'mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"',
            'mv "$held_tmp" "$run_root/held-gpu-submission.json"',
            'mv "$source_tmp" "$run_root/source-contract.json"',
            "dependency=afterany:$gpu_job_id",
            'mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"',
            'mv "$submission_tmp" "$run_root/submission.json"',
            "# Final fingerprint immediately before release.",
            'scontrol release "$gpu_job_id"',
        ]
        offsets = [value.index(item) for item in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(value.count("scontrol release"), 1)
        self.assertNotIn("scancel", value)
        self.assertNotIn("--requeue", value)
        self.assertGreaterEqual(value.count("simulator_efficacy_claim_allowed:false"), 3)
        self.assertGreaterEqual(value.count("infeasibility_claim_allowed:false"), 4)
        self.assertLess(
            value.index('mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"'),
            value.index('gpu_record=$(scontrol show job "$gpu_job_id" -o)'),
        )
        self.assertLess(
            value.index('mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"'),
            value.index('cpu_record=$(scontrol show job "$cpu_job_id" -o)'),
        )
        self.assertIn("automatic_cancellation_allowed:false", value)

    def test_login_submitter_is_control_plane_only(self) -> None:
        local = _text(SUBMITTER).split("<<'REMOTE'", 1)[0]
        self.assertIn("scripts/hpc/preflight.sh", local)
        self.assertIn('ssh "$HOST" bash -s', local)
        self.assertNotRegex(local, r"(^|\n)\s*(python|python3|uv run)\b")
        self.assertNotIn("nvidia-smi", local)

    def test_local_and_remote_source_binding_sets_are_identical(self) -> None:
        value = _text(SUBMITTER)
        matches = re.findall(
            r"(?:BOUND_REPOSITORY_PATHS|bound_paths)=\(\n(.*?)\n\)",
            value,
            flags=re.DOTALL,
        )
        self.assertEqual(len(matches), 2)
        local = shlex.split(matches[0], comments=True)
        remote = shlex.split(matches[1], comments=True)
        self.assertEqual(local, remote)
        self.assertEqual(len(local), len(set(local)))
        publication_source = _text(
            ROOT / "main/crfs_oracle/r05a_constrained_flow_publication.py"
        )
        tree = ast.parse(publication_source)
        constants: dict[str, object] = {}
        publication_paths: set[str] | None = None
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name)
                and target.id == "BOUND_REPOSITORY_PATHS"
                for target in node.targets
            ):
                continue
            self.assertIsInstance(node.value, ast.Call)
            elements = node.value.args[0].elts
            publication_paths = {
                str(
                    element.value
                    if isinstance(element, ast.Constant)
                    else constants[element.id]
                )
                for element in elements
            }
        self.assertIsNotNone(publication_paths)
        self.assertEqual(set(local), publication_paths)
        for required in (
            "main/crfs_oracle/r05a_constrained_flow_publication.py",
            "main/publish_crfs_r05a_constrained_flow_canary.py",
            "openpi/scripts/serve_policy.py",
            "scripts/hpc/prepare_jsonschema_overlay.sh",
            "scripts/hpc/prepare_transformers_overlay.sh",
            "scripts/hpc/lib/r05a_runtime_identity.sh",
            "scripts/hpc/run_r05a_constrained_flow_workload.sh",
            "scripts/hpc/run_r05a_constrained_flow_canary.sh",
            "scripts/hpc/validate_r05a_constrained_flow_canary.sh",
            "scripts/hpc/submit_r05a_constrained_flow_canary.sh",
            "slurm/r05a_constrained_flow_canary_h100.sbatch",
            "slurm/r05a_constrained_flow_canary_validate_cpu.sbatch",
            "tests/test_r05a_constrained_flow_hpc_contract.py",
            "tests/test_r05a_constrained_flow_publication.py",
            "openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py",
            "openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py",
            "openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
            "openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py",
            "openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py",
        ):
            self.assertIn(required, local)

    def test_implementation_is_fail_closed_or_exactly_released(self) -> None:
        apparatus = json.loads(_text(APPARATUS))
        scientific = json.loads(_text(SCIENTIFIC))
        self.assertEqual(apparatus["resource_contract"], RESOURCE)
        self.assertEqual(apparatus["scientific_config"]["sha256"], _sha(SCIENTIFIC))
        self.assertEqual(
            apparatus["submission_recovery_contract"]["gpu_numeric_id_receipt"],
            "provisional-gpu-job-id.json",
        )
        self.assertEqual(
            apparatus["submission_recovery_contract"]["cpu_numeric_id_receipt"],
            "provisional-cpu-job-id.json",
        )
        self.assertTrue(
            apparatus["submission_recovery_contract"][
                "receipt_written_immediately_after_numeric_sbatch_return_before_scontrol_validation"
            ]
        )
        self.assertFalse(
            apparatus["submission_recovery_contract"][
                "automatic_cancellation_allowed"
            ]
        )
        if apparatus["ready_to_run"] is False:
            self.assertTrue(apparatus["blocked_on"])
            self.assertIsNone(apparatus.get("execution_release"))
            self.assertFalse(scientific["ready_to_run"])
            self.assertTrue(scientific["blocked_on"])
        else:
            self.assertEqual(apparatus["blocked_on"], [])
            self.assertTrue(scientific["ready_to_run"])
            self.assertEqual(scientific["blocked_on"], [])
            self.assertEqual(scientific["config_status"], "released_exact_single_canary")
            self.assertTrue(scientific["preregistration"]["h100_submission_authorized"])
            release = apparatus["execution_release"]
            self.assertEqual(scientific["execution_release"], release)
            self.assertEqual(release["schema_version"], "1.0")
            self.assertEqual(
                release["artifact_role"],
                "r05a_constrained_flow_canary_execution_release",
            )
            self.assertEqual(release["source_host"], "worker-1")
            self.assertEqual(release["resources"], {k: v for k, v in RESOURCE.items() if k != "source_host"})
            self.assertEqual(release["allowed_release_diff_paths"], RELEASE_PATHS)
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])

    def test_release_identity_checks_bind_direct_parent_and_three_paths(self) -> None:
        value = _text(SUBMITTER)
        self.assertIn('rev-list --parents -n 1 "$EXPECTED_RELEASE_COMMIT"', value)
        self.assertIn("PARENT_CFS_CONFIG=$(git show", value)
        self.assertIn("PARENT_APPARATUS_CONFIG=$(git show", value)
        for path in RELEASE_PATHS:
            self.assertIn(path, value)
        self.assertIn("release-only path set changed", value)
        self.assertIn("one preregistered CFS-00A canary submission only", value)
        self.assertIn("Automatic resubmission: `false`", value)
        self.assertIn("Automatic next experiment: `false`", value)

    def test_exact_array_task_status_is_not_parent_jobidraw(self) -> None:
        value = _text(CPU_WRAPPER)
        self.assertIn("scripts/hpc/lib/slurm_exact_array_task_status.sh", value)
        self.assertIn("crfs_wait_for_exact_completed_array_task", value)
        self.assertIn("SOURCE_TASK_ID=${SOURCE_JOB_ID}_0", value)
        self.assertNotIn("JobIDRaw", value)

    def test_historical_r05a_shell_apparatus_is_byte_preserved(self) -> None:
        expected = {
            "scripts/hpc/run_r05a_sampled_current_canary.sh": "5dfe1c52d4af27ce1eb4945c845bc60e067a45ab3b6ac882a7281edafa23646b",
            "scripts/hpc/validate_r05a_sampled_current_canary.sh": "30071d73c7250f15190e505b20a8f159d1accd20243cfc0e1238a55b3701cc14",
            "scripts/hpc/submit_r05a_sampled_current_canary.sh": "004a12cf899469d7199262c4e88a25e5a74d6b7398a9f67f404a3e3a15cadd32",
            "slurm/r05a_sampled_current_canary_h100.sbatch": "2b5ad690efddac31b292c3b0b3f40233e0f3ea75557c656c417c4a0cb481696f",
            "slurm/r05a_sampled_current_canary_validate_cpu.sbatch": "26857cf114a5d71302496a192bdeec304468db1c614b029b859f7421cd4a06c1",
            "scripts/hpc/run_r05a_canary.sh": "a41636d74fc59ae34c9d60776f14995dcaea274b3eb2f7c2f41fcd6ae0666248",
            "openpi/scripts/serve_policy.py": "eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809",
        }
        for relative, digest in expected.items():
            self.assertEqual(_sha(ROOT / relative), digest, relative)
        self.assertEqual(
            _sha(
                ROOT
                / "docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"
            ),
            "f1906b21d0fc79b44b01d7e7a4bd693835a6489f014a1cb4ea07379d31013f17",
        )
        self.assertIn(
            "eccc0448b4873fd30a1fff3355c5de7a7c227138e6db04b7dcaa5bb38a6a5809",
            _text(SUBMITTER),
        )


if __name__ == "__main__":
    unittest.main()
