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
        pairing = {
            "manifest_row_sha256": "1" * 64,
            "initial_state_sha256": "2" * 64,
            "initial_observation_sha256": "3" * 64,
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
        return {
            "case_id": "case",
            "arm": "pi05_plus_aegis_translational",
            "mode": "aegis",
            "protocol_id": "protocol",
            "pairing": pairing,
            "actions": actions,
            "policy_queries": queries,
            "action_invariance_ledger": ledger,
            "failure_diagnostics": {"enabled": enabled},
        }

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
                }
            ],
        }
        self.validator._validate_qp_contexts(result)
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
            counts[len(atoms)] += 1
            predicates.update(atom.lower() for atom in atoms)
        self.assertEqual(counts, {1: 24, 2: 8})
        self.assertEqual(predicates, {"in", "on"})


if __name__ == "__main__":
    unittest.main()
