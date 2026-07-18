import collections
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
EVALUATOR_PATH = MAIN / "evaluate_safelibero_aegis.py"
DIAGNOSTICS_PATH = MAIN / "aegis_failure_diagnostics.py"
VALIDATOR_PATH = ROOT / "analysis" / "validate_aegis_failure_diagnostics.py"
REFERENCE_PATH = (
    ROOT / "fixtures" / "vlsa_table1_canary_action_reference.json"
)
MANIFEST_PATH = ROOT / "manifests" / "vlsa_table1_population.jsonl"


def _load(name, path, *, main_path=False):
    if main_path:
        sys.path.insert(0, str(MAIN))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        spec.loader.exec_module(module)
        return module
    finally:
        if main_path:
            sys.path.remove(str(MAIN))


class AegisFailureDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostics = _load(
            "aegis_failure_diagnostics_test", DIAGNOSTICS_PATH
        )
        cls.evaluator = _load(
            "aegis_failure_evaluator_test",
            EVALUATOR_PATH,
            main_path=True,
        )
        cls.validator = _load(
            "aegis_failure_validator_test", VALIDATOR_PATH
        )

    def _synthetic_result(self, *, enabled):
        actions = [
            {
                "step": 0,
                "nominal_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                "nominal_translational": [
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                "executed": [0.0, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
                "env_step_input": [
                    0.0,
                    0.2,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                "control_path": "aegis_qp",
                "qp": {
                    "z_before": [1.0, 0.0, 0.0],
                    "z_after": [0.9, 0.1, 0.0],
                },
            },
            {
                "step": 1,
                "nominal_raw": [0.2, 0.1, 0.0, 0.4, 0.5, 0.6, -1.0],
                "nominal_translational": [
                    0.2,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                "executed": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, -1.0],
                "env_step_input": [
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                "control_path": "aegis_qp",
                "qp": {
                    "z_before": [0.9, 0.1, 0.0],
                    "z_after": [0.8, 0.2, 0.0],
                },
            },
        ]
        queries = [
            {
                "query_index": 0,
                "rng_seed": 17,
                "returned_action_shape": [10, 7],
                "returned_actions_sha256": "a" * 64,
            }
        ]
        ledger = self.diagnostics.action_invariance_ledger(
            actions=actions, policy_queries=queries
        )
        settled_contract = {
            "schema_version": "vlsa_table1_settled_input.v1",
            "prompt": "pick the bowl",
        }
        pairing = {
            "manifest_row_sha256": "1" * 64,
            "initial_state_sha256": "2" * 64,
            "initial_observation_sha256": (
                self.diagnostics.sha256_bytes(
                    self.diagnostics.canonical_json_bytes(
                        settled_contract
                    )
                )
            ),
            "initial_observation_contract": settled_contract,
            "settled_simulator_state_sha256": "4" * 64,
            "settled_active_obstacle_position_sha256": "5" * 64,
            "policy_noise_schedule_id": "schedule",
            "policy_noise_schedule_sha256": "6" * 64,
            "initial_policy_action_chunk_sha256": "a" * 64,
            "semantic_label_record_sha256": "7" * 64,
            "semantic_label_settled_agentview_sha256": "8" * 64,
            "semantic_obstacle_label": "blue moka pot",
            "max_steps": 300,
            "model_action_horizon": 10,
            "replan_steps": 5,
        }
        result = {
            "case_id": "case",
            "arm": "pi05_plus_aegis_translational",
            "mode": "aegis",
            "protocol_id": "protocol",
            "pairing": pairing,
            "actions": actions,
            "policy_queries": queries,
            "action_invariance_ledger": ledger,
        }
        if enabled:
            result["failure_diagnostics"] = {"enabled": True}
        return result

    def _terminal_qp_failure_result(self, *, failure_type="no_solution"):
        import numpy as np

        context = {
            "p1": [0.0, 0.0, 0.0],
            "R1": np.eye(3).tolist(),
            "q1_diag": [0.06, 0.12, 0.11],
            "p2": [1.0, 0.0, 0.0],
            "R2": np.eye(3).tolist(),
            "Q2_diag": [0.1, 0.1, 0.1],
            "z_before": [1.0, 0.0, 0.0],
            "nominal_translational": [
                0.02,
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            "u_solution": [0.0] * 6,
        }
        reconstructed = self.validator._recompute_qp_context(context)
        context.update(
            {
                "status": "failure",
                "failure_type": failure_type,
                "solver": "OSQP",
                "solver_status": "infeasible",
                "solver_stats": {"available": True},
                "solution_observation": {
                    "variable_value_is_none": True,
                    "variable_shape": [6],
                    "problem_value": None,
                },
                "v_ref": reconstructed["v_ref"].tolist(),
                "u_v_reference": reconstructed[
                    "u_v_reference"
                ].tolist(),
                "u_z_reference": reconstructed[
                    "u_z_reference"
                ].tolist(),
                "reference": reconstructed["reference"].tolist(),
                "weights_diagonal": reconstructed[
                    "weights_diagonal"
                ].tolist(),
                "cbf": {
                    "a_v": reconstructed["a_v"].tolist(),
                    "a_omega": reconstructed["a_omega"].tolist(),
                    "a_u_v": reconstructed["a_u_v"].tolist(),
                    "a_u_z": reconstructed["a_u_z"].tolist(),
                    "mu_row": reconstructed["mu_row"].tolist(),
                    "h": reconstructed["h"],
                    "alpha_gain": 10.0,
                    "constant": reconstructed["constant"],
                    "reference_lhs": reconstructed["reference_lhs"],
                    "reference_slack": reconstructed["reference_lhs"],
                    "reference_violation": max(
                        0.0, -reconstructed["reference_lhs"]
                    ),
                },
            }
        )
        context.pop("u_solution")
        method_failure = {
            "status": "method_failure",
            "component": "aegis_qp",
            "phase": "control",
            "step": 0,
            "type": "MethodFailure",
            "message": "synthetic terminal QP failure",
            "safety_by_no_execution": True,
            "nominal_raw": [0.02, 0.01, 0.0, 0.4, 0.5, 0.6, -1.0],
            "nominal_translational": list(
                context["nominal_translational"]
            ),
            "diagnostics": context,
            "diagnostics_payload_sha256": (
                self.diagnostics.sha256_bytes(
                    self.diagnostics.canonical_json_bytes(context)
                )
            ),
        }
        return {
            "mode": "aegis",
            "status": "method_failure",
            "terminal_reason": "method_failure",
            "actions": [],
            "method_failure": method_failure,
        }

    def _rehash_terminal_qp_failure(self, result):
        context = result["method_failure"]["diagnostics"]
        result["method_failure"]["diagnostics_payload_sha256"] = (
            self.diagnostics.sha256_bytes(
                self.diagnostics.canonical_json_bytes(context)
            )
        )

    def _rehash_result_payload(self, result):
        result.pop("result_payload_sha256", None)
        result["result_payload_sha256"] = self.diagnostics.sha256_bytes(
            self.diagnostics.canonical_json_bytes(result)
        )

    def _goal_terminal_result(self, root):
        import numpy as np

        result = self._synthetic_result(enabled=True)
        case = json.loads(
            next(
                line
                for line in MANIFEST_PATH.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        )
        result.update(
            {
                "case_id": case["case_id"],
                "case": case,
                "protocol_id": case["protocol_id"],
                "mode": "pi05",
                "arm": "pi05_translational",
                "status": "complete",
                "scientific_result": True,
                "terminal_reason": "task_success",
                "task_success": True,
            }
        )
        result["pairing"]["manifest_row_sha256"] = (
            self.diagnostics.sha256_bytes(
                self.diagnostics.canonical_json_bytes(case)
            )
        )
        atoms = [
            {
                "index": 0,
                "predicate": "on",
                "arguments": ["akita_black_bowl_1", "plate_1"],
            }
        ]

        def snapshot(step, value, state_hash, previous):
            return {
                "step": step,
                "values": [value],
                "satisfied_count": int(value),
                "fraction": float(value),
                "all_satisfied": value,
                "newly_satisfied_indices": (
                    [0] if value and not previous else []
                ),
                "regressed_indices": (
                    [0] if previous and not value else []
                ),
                "argument_poses": [
                    {
                        "atom_index": 0,
                        "arguments": [
                            {
                                "name": "akita_black_bowl_1",
                                "object_state_type": "object",
                                "position": [0.1, 0.2, 0.3],
                                "quaternion": [1.0, 0.0, 0.0, 0.0],
                            },
                            {
                                "name": "plate_1",
                                "object_state_type": "site",
                                "position": [0.4, 0.5, 0.6],
                                "quaternion": [1.0, 0.0, 0.0, 0.0],
                            },
                        ],
                    }
                ],
                "simulator_state_sha256_before": state_hash,
                "simulator_state_sha256_after": state_hash,
                "inert": True,
            }

        initial = snapshot(-1, False, "4" * 64, False)
        first = snapshot(0, False, "9" * 64, False)
        final = snapshot(1, True, "a" * 64, False)
        result["actions"][0]["done"] = False
        result["actions"][1]["done"] = True
        result["actions"][0]["goal_progress"] = first
        result["actions"][1]["goal_progress"] = final
        definition = {
            "schema_version": "safelibero_goal_progress.v1",
            "source": "native_bddl_goal_predicates",
            "logic": "conjunction",
            "goal_atoms": atoms,
        }
        result["goal_progress"] = {
            **definition,
            "goal_definition_sha256": self.diagnostics.sha256_bytes(
                self.diagnostics.canonical_json_bytes(definition)
            ),
            "initial": initial,
            "final": final,
            "summary": self.evaluator._goal_progress_summary(
                initial,
                result["actions"],
            ),
        }
        case_dir = root / "pi05" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
        terminal_frame = self.diagnostics.publish_terminal_frame_artifact(
            frame=frame,
            case_dir=case_dir,
            output_root=root,
        )
        video_path = case_dir / "episode.mp4"
        video_path.write_bytes(b"synthetic-complete-video")
        result["terminal_observation"] = {
            "agentview_array_sha256": terminal_frame["array"][
                "array_sha256"
            ],
            "simulator_state_sha256": final[
                "simulator_state_sha256_after"
            ],
            "frame_index": 2,
            "after_executed_action_count": 2,
        }
        result["video"] = {
            "path": str(video_path.relative_to(root)),
            "sha256": self.diagnostics.sha256_path(video_path),
            "frames": 3,
            "fps": 30,
            "terminal_source_array_sha256": terminal_frame["array"][
                "array_sha256"
            ],
            "complete_episode": True,
        }
        result["metrics"] = {
            "executed_action_count": 2,
            "termination_reason": "task_success",
            "task_success": True,
        }
        result["failure_diagnostics"] = {
            "schema_version": self.diagnostics.DIAGNOSTICS_SCHEMA,
            "enabled": True,
            "status": "published",
            "control_effect": "read_only_observation",
            "terminal_frame": terminal_frame,
            "contacts": {},
        }
        self._rehash_result_payload(result)
        return result

    def test_released_utils_remain_byte_for_byte_pristine(self):
        self.assertEqual(
            self.diagnostics.sha256_path(ROOT / "main" / "utils.py"),
            "fa394d4d3c01de4b6864674cb7e0072b0d850d841986ddd2dc83963ee88208ab",
        )

    def test_prior_perception_directory_is_attempt_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "aegis" / "case"
            perception = case_dir / "perception"
            perception.mkdir(parents=True)
            stale = perception / "annotated_ agentview_image.jpg"
            stale.write_bytes(b"stale-prior-attempt")
            archived = self.evaluator._archive_prior_case_artifacts(
                case_dir
            )
            self.assertFalse(perception.exists())
            archived_path = case_dir / archived["perception/"]
            self.assertTrue(archived_path.is_dir())
            self.assertEqual(
                (archived_path / stale.name).read_bytes(),
                b"stale-prior-attempt",
            )

    def test_resume_requires_exact_diagnostics_mode(self):
        self.assertTrue(
            self.evaluator._failure_diagnostics_mode_matches(
                {},
                required=False,
            )
        )
        self.assertFalse(
            self.evaluator._failure_diagnostics_mode_matches(
                {"failure_diagnostics": {"enabled": True}},
                required=False,
            )
        )
        self.assertFalse(
            self.evaluator._failure_diagnostics_mode_matches(
                {"failure_diagnostics": {"enabled": False}},
                required=False,
            )
        )
        self.assertTrue(
            self.evaluator._failure_diagnostics_mode_matches(
                {"failure_diagnostics": {"enabled": True}},
                required=True,
            )
        )
        self.assertFalse(
            self.evaluator._failure_diagnostics_mode_matches(
                {},
                required=True,
            )
        )

    def test_action_ledgers_are_equal_and_tampering_is_rejected(self):
        off = self._synthetic_result(enabled=False)
        on = self._synthetic_result(enabled=True)
        receipt = self.validator.validate_action_invariance_pair(off, on)
        self.assertEqual(receipt["status"], "action_invariant")
        tamper_paths = (
            ("nominal_raw", lambda result: result["actions"][0][
                "nominal_raw"
            ].__setitem__(0, 9.0)),
            ("executed", lambda result: result["actions"][0][
                "executed"
            ].__setitem__(0, 9.0)),
            ("z", lambda result: result["actions"][0]["qp"][
                "z_after"
            ].__setitem__(0, 9.0)),
            ("schedule", lambda result: result["policy_queries"][0].__setitem__(
                "rng_seed", 99
            )),
        )
        for name, mutate in tamper_paths:
            changed = copy.deepcopy(on)
            mutate(changed)
            with self.subTest(name=name), self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator.validate_action_invariance_pair(off, changed)
        changed_contract = copy.deepcopy(on)
        changed_contract["pairing"]["initial_observation_contract"][
            "prompt"
        ] = "tampered prompt"
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator.validate_action_invariance_pair(
                off,
                changed_contract,
            )
        terminal_attempt = {
            "component": "aegis_qp",
            "phase": "control",
            "step": 2,
            "nominal_raw": [0.3, 0.2, 0.1, 0.4, 0.5, 0.6, -1.0],
            "nominal_translational": [
                0.3,
                0.2,
                0.1,
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
        }
        off["method_failure"] = copy.deepcopy(terminal_attempt)
        on["method_failure"] = copy.deepcopy(terminal_attempt)
        self.validator.validate_action_invariance_pair(off, on)
        on["method_failure"]["nominal_raw"][0] = 9.0
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator.validate_action_invariance_pair(off, on)

    def test_diagnostic_acceptance_binds_goal_terminal_frame_and_video(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._goal_terminal_result(root)

            def validate(value):
                with mock.patch.object(
                    self.validator, "_validate_contacts"
                ), mock.patch.object(
                    self.validator,
                    "_probe_episode_video",
                    return_value={
                        "status": "fully_decoded",
                        "decoded_frame_count": 3,
                    },
                ):
                    return self.validator.validate_diagnostic_result(
                        value,
                        output_root=root,
                    )

            self.assertEqual(validate(result)["status"], "valid")
            mutations = (
                (
                    "scientific_result",
                    lambda value: value.__setitem__(
                        "scientific_result", False
                    ),
                ),
                (
                    "goal_summary",
                    lambda value: value["goal_progress"]["summary"].__setitem__(
                        "final_satisfied_count", 0
                    ),
                ),
                (
                    "bool_action_step",
                    lambda value: value["actions"][1].__setitem__(
                        "step", True
                    ),
                ),
                (
                    "bool_satisfied_count",
                    lambda value: value["actions"][1][
                        "goal_progress"
                    ].__setitem__("satisfied_count", True),
                ),
                (
                    "string_fraction",
                    lambda value: value["actions"][1][
                        "goal_progress"
                    ].__setitem__("fraction", "1.0"),
                ),
                (
                    "goal_initial_state",
                    lambda value: value["goal_progress"]["initial"].__setitem__(
                        "simulator_state_sha256_before", "0" * 64
                    ),
                ),
                (
                    "goal_terminal_state",
                    lambda value: value["terminal_observation"].__setitem__(
                        "simulator_state_sha256", "0" * 64
                    ),
                ),
                (
                    "terminal_pixels",
                    lambda value: value["failure_diagnostics"][
                        "terminal_frame"
                    ]["array"].__setitem__("array_sha256", "0" * 64),
                ),
                (
                    "terminal_observation_pixels",
                    lambda value: value["terminal_observation"].__setitem__(
                        "agentview_array_sha256", "0" * 64
                    ),
                ),
                (
                    "video_terminal_pixels",
                    lambda value: value["video"].__setitem__(
                        "terminal_source_array_sha256", "0" * 64
                    ),
                ),
                (
                    "video_frame_count",
                    lambda value: value["video"].__setitem__("frames", 2),
                ),
                (
                    "video_fps",
                    lambda value: value["video"].__setitem__("fps", 29),
                ),
            )
            for name, mutate in mutations:
                changed = copy.deepcopy(result)
                mutate(changed)
                self._rehash_result_payload(changed)
                with self.subTest(name=name), self.assertRaises(
                    self.validator.DiagnosticValidationError
                ):
                    validate(changed)

            contradictory_success = copy.deepcopy(result)
            final_action = contradictory_success["actions"][1]
            final_action["done"] = False
            final_snapshot = final_action["goal_progress"]
            final_snapshot.update(
                {
                    "values": [False],
                    "satisfied_count": 0,
                    "fraction": 0.0,
                    "all_satisfied": False,
                    "newly_satisfied_indices": [],
                    "regressed_indices": [],
                }
            )
            contradictory_success["goal_progress"]["final"] = copy.deepcopy(
                final_snapshot
            )
            contradictory_success["goal_progress"]["summary"] = (
                self.evaluator._goal_progress_summary(
                    contradictory_success["goal_progress"]["initial"],
                    contradictory_success["actions"],
                )
            )
            self._rehash_result_payload(contradictory_success)
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                validate(contradictory_success)

            changed_atoms = copy.deepcopy(result)
            changed_atoms["goal_progress"]["goal_atoms"][0][
                "arguments"
            ][0] = "wrong_bowl_1"
            for snapshot in [
                changed_atoms["goal_progress"]["initial"],
                *[
                    action["goal_progress"]
                    for action in changed_atoms["actions"]
                ],
            ]:
                snapshot["argument_poses"][0]["arguments"][0][
                    "name"
                ] = "wrong_bowl_1"
            definition = {
                key: changed_atoms["goal_progress"][key]
                for key in (
                    "schema_version",
                    "source",
                    "logic",
                    "goal_atoms",
                )
            }
            changed_atoms["goal_progress"][
                "goal_definition_sha256"
            ] = self.diagnostics.sha256_bytes(
                self.diagnostics.canonical_json_bytes(definition)
            )
            self._rehash_result_payload(changed_atoms)
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                validate(changed_atoms)

            video_path = root / result["video"]["path"]
            video_path.write_bytes(b"tampered-video")
            self._rehash_result_payload(result)
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                validate(result)
            video_path.write_bytes(b"synthetic-complete-video")
            terminal_path = (
                root
                / result["failure_diagnostics"]["terminal_frame"]["path"]
            )
            terminal_path.unlink()
            self._rehash_result_payload(result)
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                validate(result)

    def test_terminal_frame_artifact_requires_lossless_uint8_rgb(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "pi05" / "case"
            case_dir.mkdir(parents=True)
            descriptor = (
                self.diagnostics.publish_terminal_frame_artifact(
                    frame=np.zeros((4, 5), dtype=np.uint8),
                    case_dir=case_dir,
                    output_root=root,
                )
            )
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_terminal_frame_artifact(
                    descriptor,
                    output_root=root,
                )

    def test_episode_video_probe_decodes_every_frame_and_terminal_source(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        class Reader:
            def __init__(self, frames, *, fps=30.0, decode_error=None):
                self.frames = frames
                self.fps = fps
                self.decode_error = decode_error
                self.closed = False

            def get_meta_data(self):
                return {"fps": self.fps}

            def __iter__(self):
                for frame in self.frames:
                    yield frame
                if self.decode_error is not None:
                    raise self.decode_error

            def close(self):
                self.closed = True

        imageio_package = types.ModuleType("imageio")
        imageio_package.__path__ = []
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio_package.v2 = imageio_v2
        terminal = np.zeros((1024, 1024, 3), dtype=np.uint8)

        def run(reader, *, count=2, source=terminal):
            imageio_v2.get_reader = lambda _path: reader
            with mock.patch.dict(
                sys.modules,
                {
                    "imageio": imageio_package,
                    "imageio.v2": imageio_v2,
                },
            ):
                return self.validator._probe_episode_video(
                    Path("episode.mp4"),
                    expected_frame_count=count,
                    terminal_source=source,
                )

        accepted = run(
            Reader(
                [
                    np.ones((1024, 1024, 3), dtype=np.uint8),
                    terminal,
                ]
            )
        )
        self.assertEqual(accepted["status"], "fully_decoded")
        self.assertEqual(accepted["decoded_frame_count"], 2)
        for name, reader in (
            (
                "truncated",
                Reader(
                    [terminal],
                    decode_error=RuntimeError("truncated h264 stream"),
                ),
            ),
            ("wrong_fps", Reader([terminal, terminal], fps=29.0)),
            (
                "wrong_resolution",
                Reader(
                    [
                        terminal,
                        np.zeros((32, 32, 3), dtype=np.uint8),
                    ]
                ),
            ),
            (
                "wrong_terminal",
                Reader(
                    [
                        terminal,
                        np.full(
                            (1024, 1024, 3),
                            255,
                            dtype=np.uint8,
                        ),
                    ]
                ),
            ),
            (
                "extra_frame",
                Reader([terminal, terminal, terminal]),
            ),
        ):
            with self.subTest(name=name), self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                run(reader)

    def test_canary_reference_is_frozen_and_matches_local_validated_results(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            reference["schema_version"],
            self.diagnostics.REFERENCE_SCHEMA,
        )
        self.assertEqual(set(reference["arms"]), {
            "pi05_translational",
            "pi05_plus_aegis_translational",
        })
        self.assertEqual(
            reference["arms"]["pi05_translational"][
                "nominal_raw_sequence_sha256"
            ],
            "639f764d4cfbeafd9b1e287027de5c876dbe22034066277b7e00e39e400bc2b7",
        )
        local = Path(
            "/Users/quanth238/personal/Research/probe_vla/output/"
            "vlsa_aegis_table1/"
            "paired_canary_vlsa-t1-spatial-i-t2-e00"
        )
        if not local.is_dir():
            self.skipTest("validated allocation canary is not mirrored locally")
        for name in ("pi05", "aegis"):
            result = json.loads(
                (local / f"{name}_result.json").read_text(encoding="utf-8")
            )
            self.validator.validate_canary_reference(result, reference)

    def test_grounding_observer_retains_return_order_selection_and_crop(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_boxes_hash = self.diagnostics.array_sha256(empty_boxes)
        self.assertEqual(len(empty_boxes_hash), 64)
        self.assertEqual(
            empty_boxes_hash,
            self.diagnostics.array_sha256(empty_boxes.copy()),
        )

        inference = types.ModuleType("groundingdino.util.inference")
        box_ops = types.ModuleType("groundingdino.util.box_ops")
        util = types.ModuleType("groundingdino.util")
        package = types.ModuleType("groundingdino")

        def predict(**kwargs):
            return (
                np.asarray(
                    [[0.5, 0.5, 0.4, 0.2], [0.2, 0.2, 0.1, 0.1]],
                    dtype=np.float32,
                ),
                np.asarray([0.91, 0.72], dtype=np.float32),
                ["first", "second"],
            )

        def convert(boxes):
            boxes = np.asarray(boxes)
            output = boxes.copy()
            output[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.0
            output[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2.0
            return output

        inference.predict = predict
        box_ops.box_cxcywh_to_xyxy = convert
        util.inference = inference
        util.box_ops = box_ops

        class Released:
            @staticmethod
            def get_real_depth_map(sim, depth):
                return np.asarray(depth, dtype=np.float32) + 1.0

            def get_point_cloud(
                self,
                image,
                depth,
                env,
                view,
                prompt,
                model,
                save_path,
                device,
            ):
                metric = self.get_real_depth_map(env.sim, depth)
                boxes, logits, phrases = inference.predict(
                    model=model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=device,
                )
                box_ops.box_cxcywh_to_xyxy(boxes)
                self.last_metric_hash = self.diagnostic_hash(metric)
                return np.ones((3, 3), dtype=np.float64)

            @staticmethod
            def diagnostic_hash(value):
                return value.shape

        released = Released()
        env = types.SimpleNamespace(sim=object())
        modules = {
            "groundingdino": package,
            "groundingdino.util": util,
            "groundingdino.util.inference": inference,
            "groundingdino.util.box_ops": box_ops,
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            sys.modules, modules
        ):
            points, record = (
                self.diagnostics.run_released_point_cloud_with_diagnostics(
                    released=released,
                    image=np.zeros((100, 200, 3), dtype=np.uint8),
                    depth=np.zeros((100, 200, 1), dtype=np.float32),
                    env=env,
                    view="agentview",
                    label="blue moka pot",
                    grounding_model=object(),
                    perception_dir=Path(temporary),
                    device="cuda",
                )
            )
            inference.predict = lambda **kwargs: (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                [],
            )
            empty_points, empty_record = (
                self.diagnostics.run_released_point_cloud_with_diagnostics(
                    released=released,
                    image=np.zeros((100, 200, 3), dtype=np.uint8),
                    depth=np.zeros((100, 200, 1), dtype=np.float32),
                    env=env,
                    view="backview",
                    label="blue moka pot",
                    grounding_model=object(),
                    perception_dir=Path(temporary),
                    device="cuda",
                )
            )
        self.assertEqual(points.shape, (3, 3))
        detections = record["detections"]
        self.assertEqual(detections["returned_order_phrases"], [
            "first",
            "second",
        ])
        self.assertEqual(detections["selected_index"], 0)
        self.assertEqual(
            detections["selected_crop"][
                "released_rgb_depth_crop_xyxy"
            ],
            [59, 39, 139, 59],
        )
        self.assertEqual(record["status"], "detected")
        self.assertEqual(empty_points.shape, (3, 3))
        self.assertEqual(empty_record["status"], "no_detection")
        self.assertIsNone(
            empty_record["detections"]["selected_index"]
        )
        self.assertTrue(empty_record["no_detection"]["explicit"])

    def test_qp_reconstruction_rejects_derived_value_tampering(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        context = {
            "p1": [0.0, 0.0, 0.0],
            "R1": np.eye(3).tolist(),
            "q1_diag": [0.06, 0.12, 0.11],
            "p2": [1.0, 0.0, 0.0],
            "R2": np.eye(3).tolist(),
            "Q2_diag": [0.1, 0.1, 0.1],
            "z_before": [1.0, 0.0, 0.0],
            "nominal_translational": [
                0.02,
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
            ],
            "u_solution": [0.1, 0.05, 0.0, 0.0, 0.2, 0.0],
        }
        reconstructed = self.validator._recompute_qp_context(context)
        executed = reconstructed["executed"].tolist()
        context.update(
            {
                "status": "solved",
                "solver": "OSQP",
                "solver_status": "optimal",
                "solver_stats": {},
                "v_ref": reconstructed["v_ref"].tolist(),
                "u_v_reference": reconstructed[
                    "u_v_reference"
                ].tolist(),
                "u_z_reference": reconstructed[
                    "u_z_reference"
                ].tolist(),
                "reference": reconstructed["reference"].tolist(),
                "weights_diagonal": reconstructed[
                    "weights_diagonal"
                ].tolist(),
                "cbf": {
                    "a_v": reconstructed["a_v"].tolist(),
                    "a_omega": reconstructed["a_omega"].tolist(),
                    "a_u_v": reconstructed["a_u_v"].tolist(),
                    "a_u_z": reconstructed["a_u_z"].tolist(),
                    "mu_row": reconstructed["mu_row"].tolist(),
                    "h": reconstructed["h"],
                    "alpha_gain": 10.0,
                    "constant": reconstructed["constant"],
                    "reference_lhs": reconstructed["reference_lhs"],
                    "reference_slack": reconstructed["reference_lhs"],
                    "reference_violation": max(
                        0.0, -reconstructed["reference_lhs"]
                    ),
                },
                "solution_lhs": reconstructed["solution_lhs"],
                "solution_slack": reconstructed["solution_lhs"],
                "solution_violation": max(
                    0.0, -reconstructed["solution_lhs"]
                ),
                "constraint_dual": 0.0,
                "objective": reconstructed["objective"],
                "z_after": reconstructed["z_after"].tolist(),
                "executed_action": executed,
                "executed_action_array_sha256": (
                    self.diagnostics.array_sha256(
                        np.asarray(executed, dtype=float)
                    )
                ),
                "executed_action_canonical_sha256": (
                    self.diagnostics.sha256_bytes(
                        self.diagnostics.canonical_json_bytes(executed)
                    )
                ),
            }
        )
        qp = {
            "solver": "OSQP",
            "solver_status": "optimal",
            "barrier_h": reconstructed["h"],
            "constraint_lhs": reconstructed["solution_lhs"],
            "objective": reconstructed["objective"],
            "u_solution": list(context["u_solution"]),
            "z_before": list(context["z_before"]),
            "z_after": list(context["z_after"]),
            "context": context,
        }
        result = {
            "mode": "aegis",
            "actions": [
                {
                    "step": 0,
                    "control_path": "aegis_qp",
                    "executed": executed,
                    "qp": qp,
                    "post_step_controller_proxy": {
                        "eef_position": [0.0, 0.0, 0.08],
                        "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "p1": [0.0, 0.0, 0.0],
                        "R1": np.eye(3).tolist(),
                    },
                }
            ],
        }
        self.validator._validate_qp_contexts(result)
        controller_binding = {
            "p1": list(context["p1"]),
            "R1": copy.deepcopy(context["R1"]),
            "q1_diag": list(context["q1_diag"]),
            "p2": list(context["p2"]),
            "R2": copy.deepcopy(context["R2"]),
            "Q2_diag": list(context["Q2_diag"]),
            "z_initial": list(context["z_before"]),
        }
        self.validator._validate_qp_contexts(
            result,
            controller_binding=controller_binding,
            require_controller_binding=True,
        )
        changed_binding = copy.deepcopy(controller_binding)
        changed_binding["p2"][0] = 2.0
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(
                result,
                controller_binding=changed_binding,
                require_controller_binding=True,
            )
        bypassed_qp = copy.deepcopy(result)
        bypassed_qp["actions"][0]["control_path"] = (
            "pi05_translational_nominal"
        )
        bypassed_qp["actions"][0]["qp"] = None
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(
                bypassed_qp,
                controller_binding=controller_binding,
                require_controller_binding=True,
            )
        for name, mutate in (
            (
                "cbf_h",
                lambda value: value["actions"][0]["qp"]["context"][
                    "cbf"
                ].__setitem__("h", 999.0),
            ),
            (
                "executed_hash",
                lambda value: value["actions"][0]["qp"]["context"].__setitem__(
                    "executed_action_array_sha256", "0" * 64
                ),
            ),
            (
                "z_transition",
                lambda value: value["actions"][0]["qp"]["context"][
                    "z_after"
                ].__setitem__(0, 0.0),
            ),
            (
                "top_level_lhs",
                lambda value: value["actions"][0]["qp"].__setitem__(
                    "constraint_lhs", 999.0
                ),
            ),
        ):
            tampered = copy.deepcopy(result)
            mutate(tampered)
            with self.subTest(name=name), self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_qp_contexts(tampered)

    def test_terminal_qp_failure_is_reconstructed_and_hash_bound(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy unavailable")

        result = self._terminal_qp_failure_result()
        self.validator._validate_qp_contexts(result)
        context = result["method_failure"]["diagnostics"]
        controller_binding = {
            "p1": list(context["p1"]),
            "R1": copy.deepcopy(context["R1"]),
            "q1_diag": list(context["q1_diag"]),
            "p2": list(context["p2"]),
            "R2": copy.deepcopy(context["R2"]),
            "Q2_diag": list(context["Q2_diag"]),
            "z_initial": list(context["z_before"]),
        }
        self.validator._validate_qp_contexts(
            result,
            controller_binding=controller_binding,
            require_controller_binding=True,
        )
        wrong_geometry = copy.deepcopy(controller_binding)
        wrong_geometry["p2"][0] = 2.0
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(
                result,
                controller_binding=wrong_geometry,
                require_controller_binding=True,
            )

        missing = copy.deepcopy(result)
        missing_context = {
            "status": "failure",
            "failure_type": "no_solution",
        }
        missing["method_failure"]["diagnostics"] = missing_context
        self._rehash_terminal_qp_failure(missing)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(missing)

        tampered_math = copy.deepcopy(result)
        tampered_math["method_failure"]["diagnostics"]["cbf"]["h"] = 999.0
        self._rehash_terminal_qp_failure(tampered_math)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(tampered_math)

        tampered_hash = copy.deepcopy(result)
        tampered_hash["method_failure"][
            "diagnostics_payload_sha256"
        ] = "0" * 64
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(tampered_hash)

        tampered_nominal = copy.deepcopy(result)
        tampered_nominal["method_failure"]["nominal_raw"][0] = 9.0
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(tampered_nominal)

        relabeled = copy.deepcopy(result)
        relabeled["method_failure"]["component"] = "other"
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(relabeled)

        relabeled_precontrol = copy.deepcopy(result)
        relabeled_precontrol["method_failure"].update(
            {
                "component": "aegis_geometry",
                "phase": "precontrol",
            }
        )
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(
                relabeled_precontrol,
                controller_binding=controller_binding,
                require_controller_binding=True,
            )

        stray_precontrol = copy.deepcopy(relabeled_precontrol)
        stray_precontrol["status"] = "complete"
        stray_precontrol["terminal_reason"] = "task_success"
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(stray_precontrol)

        deleted = copy.deepcopy(result)
        del deleted["method_failure"]
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(deleted)

        false_no_solution = copy.deepcopy(result)
        false_no_solution["method_failure"]["diagnostics"][
            "solution_observation"
        ]["variable_value_is_none"] = False
        self._rehash_terminal_qp_failure(false_no_solution)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(false_no_solution)

        optimal_no_solution = copy.deepcopy(result)
        optimal_no_solution["method_failure"]["diagnostics"][
            "solver_status"
        ] = "optimal"
        self._rehash_terminal_qp_failure(optimal_no_solution)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(optimal_no_solution)

        arbitrary_nonfinite = copy.deepcopy(result)
        arbitrary_context = arbitrary_nonfinite["method_failure"][
            "diagnostics"
        ]
        arbitrary_context["failure_type"] = "nonfinite_diagnostics"
        arbitrary_context["nonfinite_values"] = {"unrelated": "nan"}
        self._rehash_terminal_qp_failure(arbitrary_nonfinite)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(arbitrary_nonfinite)

    def test_geometry_failure_and_perception_fail_open_remain_valid(self):
        precontrol = {
            "mode": "aegis",
            "status": "method_failure",
            "terminal_reason": "method_failure",
            "actions": [],
            "method_failure": {
                "status": "method_failure",
                "component": "aegis_geometry",
                "phase": "precontrol",
                "step": 0,
                "safety_by_no_execution": True,
            },
        }
        self.validator._validate_qp_contexts(
            precontrol,
            controller_binding=None,
            require_controller_binding=True,
        )
        passthrough = {
            "mode": "aegis",
            "status": "method_failure_passthrough",
            "terminal_reason": "time_limit",
            "actions": [],
            "method_failure": {
                "status": "method_failure_passthrough",
                "component": "aegis_perception",
                "reason": "no_grounded_points",
                "corrected_execution": "corrected_translational_nominal",
                "upstream_released_execution": (
                    "raw_nominal_including_rotation"
                ),
                "executed_steps": 0,
            },
        }
        self.validator._validate_qp_contexts(
            passthrough,
            controller_binding=None,
            require_controller_binding=True,
        )

    def test_terminal_exception_types_require_typed_evidence(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy unavailable")

        for failure_type in (
            "cbf_coefficient_exception",
            "qp_construction_exception",
            "solver_exception",
        ):
            with self.subTest(failure_type=failure_type):
                result = self._terminal_qp_failure_result(
                    failure_type=failure_type
                )
                failure = {
                    "type": "MethodFailure",
                    "message": "synthetic exception",
                }
                if failure_type == "solver_exception":
                    failure["cause"] = {
                        "type": "RuntimeError",
                        "message": "synthetic OSQP error",
                    }
                result["method_failure"]["diagnostics"][
                    "failure"
                ] = failure
                self._rehash_terminal_qp_failure(result)
                self.validator._validate_qp_contexts(result)
                tampered = copy.deepcopy(result)
                tampered["method_failure"]["diagnostics"][
                    "failure"
                ] = {}
                self._rehash_terminal_qp_failure(tampered)
                with self.assertRaises(
                    self.validator.DiagnosticValidationError
                ):
                    self.validator._validate_qp_contexts(tampered)

    def test_invalid_terminal_solution_bytes_are_reconstructed(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")

        result = self._terminal_qp_failure_result(
            failure_type="invalid_solution"
        )
        solution = np.asarray([0.0, np.nan], dtype=np.float64)
        context = result["method_failure"]["diagnostics"]
        context.update(
            {
                "solution_shape": list(solution.shape),
                "solution_descriptor": (
                    self.diagnostics.array_descriptor(solution)
                ),
                "solution_values": self.diagnostics._json_safe(
                    solution.tolist()
                ),
                "solution_raw_bytes_hex": (
                    np.ascontiguousarray(solution).tobytes().hex()
                ),
            }
        )
        self._rehash_terminal_qp_failure(result)
        self.validator._validate_qp_contexts(result)

        tampered = copy.deepcopy(result)
        tampered["method_failure"]["diagnostics"][
            "solution_raw_bytes_hex"
        ] = np.asarray([0.0, 1.0], dtype=np.float64).tobytes().hex()
        self._rehash_terminal_qp_failure(tampered)
        with self.assertRaises(
            self.validator.DiagnosticValidationError
        ):
            self.validator._validate_qp_contexts(tampered)

    def test_qp_observer_does_not_change_executed_action_or_z(self):
        try:
            import cvxpy as cp
            import numpy as np
        except ImportError:
            self.skipTest("NumPy/CVXPY unavailable")

        class Released:
            @staticmethod
            def compute_h_coeffs_3d(*args):
                return (
                    np.asarray([1.0, 0.0, 0.0]),
                    np.asarray([0.0, 0.0, 0.0]),
                    np.asarray([0.0, 1.0, 0.0]),
                    0.5,
                    np.asarray([0.0, 0.2, 0.0]),
                )

        runtime = {"np": np, "cp": cp, "released_utils": Released()}
        proxy = {
            "p1": np.asarray([0.0, 0.0, 0.0]),
            "R1": np.eye(3),
        }
        geometry = {
            "p2": np.asarray([1.0, 0.0, 0.0]),
            "R2": np.eye(3),
            "Q2_diag": np.asarray([0.1, 0.1, 0.1]),
            "z_fixed": np.asarray([1.0, 0.0, 0.0]),
        }
        nominal = [0.02, 0.01, 0.0, 0.0, 0.0, 0.0, -1.0]
        off_geometry = copy.deepcopy(geometry)
        on_geometry = copy.deepcopy(geometry)
        off_action, off_qp = self.evaluator._aegis_action(
            runtime,
            nominal_translational=nominal,
            proxy=proxy,
            geometry=off_geometry,
            q1_diag=np.asarray([0.06, 0.12, 0.11]),
            diagnostics_enabled=False,
        )
        on_action, on_qp = self.evaluator._aegis_action(
            runtime,
            nominal_translational=nominal,
            proxy=proxy,
            geometry=on_geometry,
            q1_diag=np.asarray([0.06, 0.12, 0.11]),
            diagnostics_enabled=True,
        )
        self.assertEqual(off_action, on_action)
        self.assertEqual(off_qp["z_after"], on_qp["z_after"])
        self.assertEqual(
            self.diagnostics.array_sha256(off_geometry["z_fixed"]),
            self.diagnostics.array_sha256(on_geometry["z_fixed"]),
        )
        context = on_qp["context"]
        for key in (
            "p1",
            "R1",
            "q1_diag",
            "z_before",
            "reference",
            "weights_diagonal",
            "cbf",
            "solution_lhs",
            "constraint_dual",
            "solver_stats",
            "z_after",
            "executed_action_array_sha256",
        ):
            self.assertIn(key, context)
        diagnostic_result = {
            "mode": "aegis",
            "actions": [
                {
                    "step": 0,
                    "control_path": "aegis_qp",
                    "executed": on_action,
                    "qp": on_qp,
                }
            ],
        }
        self.validator._validate_qp_contexts(diagnostic_result)
        for name, mutate in (
            (
                "cbf_h",
                lambda value: value["actions"][0]["qp"]["context"][
                    "cbf"
                ].__setitem__("h", 999.0),
            ),
            (
                "executed_hash",
                lambda value: value["actions"][0]["qp"]["context"].__setitem__(
                    "executed_action_array_sha256", "0" * 64
                ),
            ),
            (
                "z_transition",
                lambda value: value["actions"][0]["qp"]["context"][
                    "z_after"
                ].__setitem__(0, 0.0),
            ),
        ):
            tampered = copy.deepcopy(diagnostic_result)
            mutate(tampered)
            with self.subTest(name=name), self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_qp_contexts(tampered)

    def test_detailed_contacts_include_robot_and_nonrobot_canonical_sides(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy unavailable")

        class Contact:
            def __init__(self, geom1, geom2, dist, normal):
                self.geom1 = geom1
                self.geom2 = geom2
                self.dist = dist
                self.pos = [0.1, 0.2, 0.3]
                self.frame = list(normal) + [0.0] * 6

        class Model:
            geom_bodyid = [1, 2, 3]
            body_parentid = [0, 0, 0, 0]
            body_names = [
                "world",
                "robot0_link",
                "moka_pot_obstacle_1",
                "table",
            ]
            geom_names = ["robot_geom", "obstacle_geom", "table_geom"]

            def body_id2name(self, index):
                return self.body_names[index]

            def geom_id2name(self, index):
                return self.geom_names[index]

        model = Model()
        data = types.SimpleNamespace(
            ncon=2,
            contact=[
                Contact(1, 0, -0.001, [1.0, 0.0, 0.0]),
                Contact(2, 1, -0.002, [0.0, 1.0, 0.0]),
            ],
        )
        env = types.SimpleNamespace(
            sim=types.SimpleNamespace(model=model, data=data)
        )
        snapshot = self.evaluator._detailed_active_obstacle_contacts(
            env, "moka_pot_obstacle_1", step=-1
        )
        self.assertEqual(snapshot["status"], "available")
        self.assertEqual(
            [event["other"]["classification"] for event in snapshot["events"]],
            ["robot", "nonrobot"],
        )
        self.assertEqual(
            snapshot["events"][0]["normal_obstacle_to_other"],
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(
            snapshot["events"][1]["normal_obstacle_to_other"],
            [0.0, -1.0, 0.0],
        )
        for event in snapshot["events"]:
            expected = self.diagnostics.sha256_bytes(
                self.diagnostics.canonical_json_bytes(
                    {
                        key: value
                        for key, value in event.items()
                        if key != "event_sha256"
                    }
                )
            )
            self.assertEqual(event["event_sha256"], expected)

    def test_compressed_geometry_and_contact_artifacts_are_hash_bound(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "aegis" / "case"
            case_dir.mkdir(parents=True)
            state = self.diagnostics.new_geometry_state(
                case_id="case",
                suite_name="safelibero_spatial",
                label="blue moka pot",
            )
            self.diagnostics._add_array(
                state, "fused_points", np.arange(12).reshape(4, 3)
            )
            state["status"] = "complete"
            descriptor = self.diagnostics.publish_geometry_artifact(
                state, case_dir=case_dir, output_root=root
            )
            self.validator._validate_npz_artifact(
                descriptor, output_root=root
            )
            contact_descriptor = (
                self.diagnostics.publish_contact_artifact(
                    case_id="case",
                    snapshots=[
                        {
                            "status": "available",
                            "step": -1,
                            "active_obstacle_name": "moka_pot_obstacle_1",
                            "events": [],
                            "robot_pairs": [],
                        },
                        {
                            "status": "available",
                            "step": 0,
                            "active_obstacle_name": "moka_pot_obstacle_1",
                            "events": [],
                            "robot_pairs": [],
                        },
                    ],
                    case_dir=case_dir,
                    output_root=root,
                )
            )
            self.validator._validate_contacts(
                contact_descriptor,
                output_root=root,
                action_count=1,
                expected_case_id="case",
            )
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_contacts(
                    contact_descriptor,
                    output_root=root,
                    action_count=1,
                    expected_case_id="different-case",
                )
            changed_summary = dict(contact_descriptor)
            changed_summary["snapshot_count"] = 999
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_contacts(
                    changed_summary,
                    output_root=root,
                    action_count=1,
                )
            contact_path = root / contact_descriptor["path"]
            contact_path.write_bytes(contact_path.read_bytes() + b"tamper")
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_contacts(
                    contact_descriptor,
                    output_root=root,
                    action_count=1,
                )
            path = root / descriptor["path"]
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_npz_artifact(
                    descriptor, output_root=root
                )

    def test_controller_geometry_binding_uses_exact_npz_arrays(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "aegis" / "case"
            case_dir.mkdir(parents=True)
            state = self.diagnostics.new_geometry_state(
                case_id="case",
                suite_name="safelibero_spatial",
                label="blue moka pot",
            )
            input_arrays = {
                "agentview": (
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.float32),
                ),
                "backview": (
                    np.ones((2, 2, 3), dtype=np.uint8),
                    np.ones((2, 2), dtype=np.float32),
                ),
            }
            settled_contract = {
                "agentview_array_sha256": (
                    self.diagnostics.array_sha256(
                        input_arrays["agentview"][0]
                    )
                ),
                "agentview_depth_array_sha256": (
                    self.diagnostics.array_sha256(
                        input_arrays["agentview"][1]
                    )
                ),
                "backview_array_sha256": (
                    self.diagnostics.array_sha256(
                        input_arrays["backview"][0]
                    )
                ),
                "backview_depth_array_sha256": (
                    self.diagnostics.array_sha256(
                        input_arrays["backview"][1]
                    )
                ),
            }
            state["views"] = {}
            for view in ("agentview", "backview"):
                boxes = np.empty((0, 4), dtype=np.float32)
                logits = np.empty((0,), dtype=np.float32)
                raw_points = np.empty((0, 3), dtype=np.float64)
                raw_descriptor = self.diagnostics._add_array(
                    state,
                    f"{view}_raw_points",
                    raw_points,
                )
                state["views"][view] = {
                    "view": view,
                    "status": "no_detection",
                    "request": {
                        "caption": "blue moka pot",
                        "box_threshold": 0.35,
                        "text_threshold": 0.25,
                        "device": "cuda",
                    },
                    "input_image": self.diagnostics.array_descriptor(
                        input_arrays[view][0]
                    ),
                    "input_depth": self.diagnostics.array_descriptor(
                        input_arrays[view][1]
                    ),
                    "detections": {
                        "count": 0,
                        "selected_index": None,
                        "returned_order_phrases": [],
                        "returned_order_boxes_cxcywh": [],
                        "returned_order_boxes_cxcywh_array": (
                            self.diagnostics.array_descriptor(boxes)
                        ),
                        "returned_order_boxes_xyxy": [],
                        "returned_order_boxes_xyxy_array": (
                            self.diagnostics.array_descriptor(boxes)
                        ),
                        "returned_order_logits": [],
                        "returned_order_logits_array": (
                            self.diagnostics.array_descriptor(logits)
                        ),
                    },
                    "no_detection": {"explicit": True},
                    "raw_points_array": raw_descriptor,
                    "returned_point_cloud": raw_descriptor,
                }
            filtering_arrays = {}
            for key in (
                "fused_points",
                "range_mask",
                "range_filtered_points",
                "centroid_distances",
                "centroid_sorted_indices",
                "nearest80_points",
                "dbscan_labels",
                "shadow_filtered_points",
                "released_filtered_points",
            ):
                filtering_arrays[key] = self.diagnostics._add_array(
                    state,
                    key,
                    np.zeros((1,), dtype=np.float64),
                )
            state["filtering"] = {
                "status": "captured",
                "shadow_matches_released": True,
                "centroid_trim": {"fraction_kept": 0.8},
                "dbscan": {"eps": 0.0001, "min_points": 50},
                "arrays": filtering_arrays,
            }
            center = np.asarray([1.0, 0.0, 0.0])
            rotation = np.eye(3)
            semiaxes = np.asarray([0.1, 0.2, 0.3])
            mvee_arrays = {}
            for key, value in {
                "hull_vertices": np.zeros((1,), dtype=np.int64),
                "hull_simplices": np.zeros((1, 3), dtype=np.int64),
                "hull_equations": np.zeros((1, 4)),
                "mvee_input": np.zeros((1, 3)),
                "center": center,
                "matrix_A": np.eye(3),
                "released_center": center,
                "released_rotation": rotation,
                "released_semiaxes": semiaxes,
                "matrix_A_eigenvalues": np.ones(3),
                "matrix_A_eigenvectors": np.eye(3),
                "hull_quadratic_values": np.zeros(1),
            }.items():
                mvee_arrays[key] = self.diagnostics._add_array(
                    state, key, value
                )
            state["mvee"] = {
                "status": "captured",
                "solver_calls": [{"status": "optimal"}],
                "matrix_A": np.eye(3).tolist(),
                "eigenvalues": [1.0, 1.0, 1.0],
                "semiaxes": semiaxes.tolist(),
                "center": center.tolist(),
                "rotation": rotation.tolist(),
                "arrays": mvee_arrays,
            }
            self.diagnostics.record_initial_geometry_direction(
                state,
                stale_proxy_center=np.zeros(3),
                stale_proxy_rotation=np.eye(3),
                obstacle_center=center,
                direction=np.asarray([1.0, 0.0, 0.0]),
            )
            state["status"] = "complete"
            descriptor = self.diagnostics.publish_geometry_artifact(
                state,
                case_dir=case_dir,
                output_root=root,
            )
            binding = self.validator._validate_geometry(
                descriptor,
                output_root=root,
                require_ready=True,
                expected_case_id="case",
                expected_suite="safelibero_spatial",
                expected_label="blue moka pot",
                settled_contract=settled_contract,
            )
            self.assertEqual(binding["p2"], center.tolist())
            tampered = copy.deepcopy(descriptor)
            tampered["record"]["mvee"]["center"][0] = 2.0
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_geometry(
                    tampered,
                    output_root=root,
                    require_ready=True,
                    expected_case_id="case",
                    expected_suite="safelibero_spatial",
                    expected_label="blue moka pot",
                    settled_contract=settled_contract,
                )
            wrong_caption = copy.deepcopy(descriptor)
            wrong_caption["record"]["views"]["agentview"]["request"][
                "caption"
            ] = "red mug"
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_geometry(
                    wrong_caption,
                    output_root=root,
                    require_ready=True,
                    expected_case_id="case",
                    expected_suite="safelibero_spatial",
                    expected_label="blue moka pot",
                    settled_contract=settled_contract,
                )
            wrong_points = copy.deepcopy(descriptor)
            wrong_points["record"]["views"]["backview"][
                "raw_points_array"
            ]["array_sha256"] = "0" * 64
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_geometry(
                    wrong_points,
                    output_root=root,
                    require_ready=True,
                    expected_case_id="case",
                    expected_suite="safelibero_spatial",
                    expected_label="blue moka pot",
                    settled_contract=settled_contract,
                )
            wrong_state = copy.deepcopy(state)
            self.diagnostics.record_initial_geometry_direction(
                wrong_state,
                stale_proxy_center=np.zeros(3),
                stale_proxy_rotation=np.eye(3),
                obstacle_center=center,
                direction=np.asarray([0.0, 1.0, 0.0]),
            )
            wrong_case_dir = root / "aegis" / "wrong-direction"
            wrong_case_dir.mkdir(parents=True)
            wrong_descriptor = (
                self.diagnostics.publish_geometry_artifact(
                    wrong_state,
                    case_dir=wrong_case_dir,
                    output_root=root,
                )
            )
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_geometry(
                    wrong_descriptor,
                    output_root=root,
                    require_ready=True,
                    expected_case_id="case",
                    expected_suite="safelibero_spatial",
                    expected_label="blue moka pot",
                    settled_contract=settled_contract,
                )

    def test_failure_geometry_requires_explicit_two_view_ledger(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "aegis" / "case"
            case_dir.mkdir(parents=True)
            state = self.diagnostics.new_geometry_state(
                case_id="case",
                suite_name="safelibero_spatial",
                label="blue moka pot",
            )
            state["status"] = "failure"
            state["failure"] = {
                "component": "groundingdino_point_cloud",
                "type": "RuntimeError",
                "message": "synthetic",
            }
            state["views"] = {
                "agentview": {
                    "view": "agentview",
                    "status": "observer_or_runtime_failure",
                    "failure": {
                        "type": "RuntimeError",
                        "message": "synthetic",
                    },
                },
                "backview": {
                    "view": "backview",
                    "status": "not_attempted",
                    "reason": "prior_view_failed:agentview",
                },
            }
            self.diagnostics._add_array(
                state,
                "failure_marker",
                np.asarray([1], dtype=np.int8),
            )
            descriptor = self.diagnostics.publish_geometry_artifact(
                state,
                case_dir=case_dir,
                output_root=root,
            )
            self.validator._validate_geometry(
                descriptor,
                output_root=root,
                require_ready=False,
            )
            missing = copy.deepcopy(descriptor)
            del missing["record"]["views"]["backview"]
            with self.assertRaises(
                self.validator.DiagnosticValidationError
            ):
                self.validator._validate_geometry(
                    missing,
                    output_root=root,
                    require_ready=False,
                )
            collecting = self.diagnostics.new_geometry_state(
                case_id="collecting",
                suite_name="safelibero_spatial",
                label="blue moka pot",
            )
            with self.assertRaises(ValueError):
                self.diagnostics.publish_geometry_artifact(
                    collecting,
                    case_dir=case_dir,
                    output_root=root,
                )

    def test_manifest_bddl_goal_structure_is_exactly_24_one_and_8_two(self):
        rows = [
            json.loads(line)
            for line in MANIFEST_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        groups = {}
        for row in rows:
            groups.setdefault(row["task_level_group_id"], row)
        self.assertEqual(len(groups), 32)
        counts = collections.Counter()
        predicates = set()
        for group_id, row in groups.items():
            text = (ROOT / row["bddl_path"]).read_text(encoding="utf-8")
            start = text.lower().index("(:goal")
            depth = 0
            end = None
            for index in range(start, len(text)):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            self.assertIsNotNone(end, group_id)
            goal = text[start:end]
            atoms = re.findall(
                r"\(\s*(In|On)\s+[^\s()]+\s+[^\s()]+\s*\)",
                goal,
                flags=re.IGNORECASE,
            )
            trusted_atoms = self.validator._trusted_native_goal_atoms(
                {
                    "case_id": row["case_id"],
                    "case": row,
                    "pairing": {
                        "manifest_row_sha256": (
                            self.diagnostics.sha256_bytes(
                                self.diagnostics.canonical_json_bytes(
                                    row
                                )
                            )
                        )
                    },
                }
            )
            self.assertEqual(len(trusted_atoms), len(atoms), group_id)
            self.assertEqual(
                [atom["predicate"] for atom in trusted_atoms],
                [predicate.lower() for predicate in atoms],
                group_id,
            )
            counts[len(atoms)] += 1
            predicates.update(atom.lower() for atom in atoms)
        self.assertEqual(counts, {1: 24, 2: 8})
        self.assertEqual(predicates, {"in", "on"})


if __name__ == "__main__":
    unittest.main()
