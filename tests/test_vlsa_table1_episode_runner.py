import ast
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
EVALUATOR_PATH = MAIN / "evaluate_safelibero_aegis.py"
CAPTURE_PATH = MAIN / "capture_safelibero_labels.py"
AGGREGATOR_PATH = ROOT / "analysis" / "aggregate_safelibero_aegis.py"
CONFIG_PATH = ROOT / "configs" / "vlsa_table1_translational.json"
MANIFEST_PATH = ROOT / "manifests" / "vlsa_table1_population.jsonl"
CANARY_LABEL_PATH = ROOT / "labels" / "vlsa_table1_canary_labels.jsonl"


def _load(name, path):
    sys.path.insert(0, str(MAIN))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(MAIN))


class Table1EpisodeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = _load("vlsa_table1_evaluator", EVALUATOR_PATH)
        cls.capture = _load("vlsa_table1_capture", CAPTURE_PATH)
        cls.aggregator = _load(
            "vlsa_table1_aggregator", AGGREGATOR_PATH
        )

    def test_import_is_dependency_light(self):
        tree = ast.parse(EVALUATOR_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse(
            {
                "cvxpy",
                "libero",
                "openpi_client",
                "groundingdino",
                "numpy",
                "scipy",
            }.intersection(imported)
        )

    def test_suite_aliases_and_paper_horizons(self):
        evaluator = self.evaluator
        self.assertEqual(
            evaluator.normalize_suite_name("safelibero_10"),
            "safelibero_long",
        )
        self.assertEqual(
            evaluator.normalize_suite_name("safelibero_long"),
            "safelibero_long",
        )
        self.assertEqual(
            evaluator.max_steps_for_case(
                {"suite": "safelibero_long", "max_steps": 550}
            ),
            550,
        )
        with self.assertRaises(evaluator.ProtocolError):
            evaluator.max_steps_for_case(
                {"suite": "safelibero_long", "max_steps": 300}
            )

    def test_runtime_code_avoids_python_39_only_string_helpers(self):
        runtime_paths = (
            EVALUATOR_PATH,
            AGGREGATOR_PATH,
            ROOT / "manifests" / "build_vlsa_table1_population.py",
        )
        for path in runtime_paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertNotIn(".removeprefix(", source)
                self.assertNotIn(".removesuffix(", source)

    def test_active_obstacle_name_parsing_supports_python_38(self):
        class Model:
            joint_names = [
                "blue_moka_pot_obstacle_1_joint0",
                "robot0_joint0",
            ]

        class Sim:
            model = Model()

        class Env:
            sim = Sim()

        name, candidates = self.evaluator._active_obstacle(
            Env(),
            {
                "blue_moka_pot_obstacle_1_pos": [0.0, 0.0, 0.1],
            },
        )
        self.assertEqual(name, "blue_moka_pot_obstacle_1")
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["in_workspace"])

    def test_translational_arm_zeros_rotation_but_preserves_xyz_gripper(self):
        nominal = [0.1, -0.2, 0.3, 0.8, -0.7, 0.6, -1.0]
        self.assertEqual(
            self.evaluator.translational_action(nominal),
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
        )

    def test_goal_telemetry_preserves_nominal_and_executed_action_bytes(self):
        evaluator = self.evaluator

        class ObjectState:
            object_state_type = "object"

            def get_geom_state(self):
                return {
                    "pos": [0.0, 0.0, 0.1],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                }

        class SiteState:
            object_state_type = "site"

            def get_geom_state(self):
                return {
                    "pos": [0.1, 0.0, 0.1],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                }

        class TaskEnv:
            parsed_problem = {
                "goal_state": [
                    ["in", "target_object_1", "target_region_1"],
                    ["on", "second_object_1", "second_region_1"],
                ]
            }
            object_states_dict = {
                "target_object_1": ObjectState(),
                "target_region_1": SiteState(),
                "second_object_1": ObjectState(),
                "second_region_1": SiteState(),
            }

            @staticmethod
            def _eval_predicate(atom):
                return atom[0] == "on"

        class SimState:
            @staticmethod
            def flatten():
                return [0.0]

        class Sim:
            @staticmethod
            def get_state():
                return SimState()

        class Env:
            env = TaskEnv()
            sim = Sim()

        nominal_raw = [0.1, -0.2, 0.3, 0.8, -0.7, 0.6, -1.0]
        executed = evaluator.translational_action(nominal_raw)
        action_payload = {
            "nominal_raw": nominal_raw,
            "executed": executed,
        }
        before = evaluator.canonical_json_bytes(action_payload)
        self.assertEqual(
            evaluator.sha256_bytes(before),
            "1b5da7a6771fd886f59ce309d554c950b896a68c63f6cdd63c0ca306b08a1b9a",
        )
        with mock.patch.object(
            evaluator, "array_sha256", return_value="a" * 64
        ), mock.patch.object(
            evaluator,
            "_finite_list",
            side_effect=lambda value: [float(item) for item in value],
        ):
            definition, atoms = evaluator._goal_progress_definition(Env())
            snapshot = evaluator._goal_progress_snapshot(
                Env(),
                atoms,
                step=-1,
                previous_values=None,
            )
        self.assertEqual(
            definition["schema_version"],
            evaluator.GOAL_PROGRESS_SCHEMA,
        )
        self.assertTrue(snapshot["inert"])
        self.assertEqual(snapshot["values"], [False, True])
        self.assertEqual(
            evaluator.canonical_json_bytes(action_payload),
            before,
        )

    def test_legacy_ets_is_preserved_and_explicit(self):
        evaluator = self.evaluator
        cases = (
            (0, False, 0),
            (1, True, 0),
            (17, True, 16),
            (300, False, 300),
        )
        for count, success, expected in cases:
            with self.subTest(count=count, success=success):
                self.assertEqual(
                    evaluator.legacy_ets_steps(count, success), expected
                )

    def test_case_selection_is_exact_and_ordered(self):
        evaluator = self.evaluator
        rows = [
            {"case_id": "a", "case_ordinal": 0},
            {"case_id": "b", "case_ordinal": 1},
        ]
        selected = evaluator.select_cases(
            rows, case_ids=["b", "a"], ordinals=[]
        )
        self.assertEqual(
            [row["case_id"] for row in selected], ["b", "a"]
        )
        self.assertEqual(
            evaluator.select_cases(
                rows, case_ids=[], ordinals=[1]
            ),
            [rows[1]],
        )
        with self.assertRaises(evaluator.ProtocolError):
            evaluator.select_cases(
                rows, case_ids=["missing"], ordinals=[]
            )

    def test_exact_array_hash_binds_dtype_shape_and_bytes(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in the local gate")
        evaluator = self.evaluator
        array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.assertEqual(
            evaluator.array_sha256(array),
            evaluator.array_sha256(array.copy()),
        )
        self.assertNotEqual(
            evaluator.array_sha256(array),
            evaluator.array_sha256(array.astype(np.int16)),
        )
        self.assertNotEqual(
            evaluator.array_sha256(array),
            evaluator.array_sha256(array.reshape(3, 2, 2)),
        )

    def test_frozen_label_must_match_case_image_and_vocabulary(self):
        evaluator = self.evaluator
        case = {"case_id": "case-1", "suite": "safelibero_spatial"}
        record = {
            "schema_version": evaluator.LABEL_SCHEMA,
            "case_id": "case-1",
            "settled_agentview_array_sha256": "abc",
            "obstacle_label": "Red Milk Carton",
            "reviewer": "codex",
            "reviewed_at": "2026-07-17T00:00:00+00:00",
        }
        self.assertEqual(
            evaluator.validate_frozen_label(
                case=case,
                label_record=record,
                settled_agentview_hash="abc",
                outcome_started_unix=1_800_000_000,
            ),
            "red milk carton",
        )
        with self.assertRaises(evaluator.ApparatusError):
            evaluator.validate_frozen_label(
                case=case,
                label_record=record,
                settled_agentview_hash="different",
            )
        invalid = dict(record, obstacle_label="the target bowl")
        with self.assertRaises(evaluator.ApparatusError):
            evaluator.validate_frozen_label(
                case=case,
                label_record=invalid,
                settled_agentview_hash="abc",
            )
        naive_time = dict(
            record, reviewed_at="2026-07-17T00:00:00"
        )
        with self.assertRaises(evaluator.ApparatusError):
            evaluator.validate_frozen_label(
                case=case,
                label_record=naive_time,
                settled_agentview_hash="abc",
            )
        future = dict(
            record, reviewed_at="2030-01-01T00:00:00+00:00"
        )
        with self.assertRaises(evaluator.ApparatusError):
            evaluator.validate_frozen_label(
                case=case,
                label_record=future,
                settled_agentview_hash="abc",
                outcome_started_unix=1_800_000_000,
            )

    def test_exact_frozen_canary_label_is_accepted_by_both_gates(self):
        record = json.loads(
            CANARY_LABEL_PATH.read_text(encoding="utf-8")
        )
        manifest = next(
            json.loads(line)
            for line in MANIFEST_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if json.loads(line)["case_id"] == record["case_id"]
        )
        self.assertEqual(
            self.evaluator.validate_frozen_label(
                case=manifest,
                label_record=record,
                settled_agentview_hash=record[
                    "settled_agentview_array_sha256"
                ],
                outcome_started_unix=2_000_000_000,
            ),
            "blue moka pot",
        )
        record_hash = self.aggregator.canonical_record_sha256(
            record
        )
        pairing = {
            "semantic_label_record_sha256": record_hash,
            "semantic_label_settled_agentview_sha256": record[
                "settled_agentview_array_sha256"
            ],
            "semantic_obstacle_label": "blue moka pot",
        }
        result = {
            "arm": "pi05_translational",
            "timing": {"started_unix": 2_000_000_000},
            "settled_observation": {
                "agentview_array_sha256": record[
                    "settled_agentview_array_sha256"
                ],
                "label_record": record,
                "label_record_sha256": record_hash,
                "obstacle_label": "blue moka pot",
            },
        }
        self.aggregator._validate_label(
            result, manifest=manifest, pairing=pairing
        )

    def test_baseline_cannot_run_before_label_manifest_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(self.evaluator.ProtocolError):
                self.evaluator.main(
                    [
                        "--manifest",
                        str(MANIFEST_PATH),
                        "--mode",
                        "pi05",
                        "--output-dir",
                        temporary,
                        "--case-ordinal",
                        "0",
                    ]
                )

    def test_selector_mismatch_is_recordable(self):
        evaluator = self.evaluator
        self.assertTrue(
            evaluator.label_matches_active_obstacle(
                "red milk carton", "milk_obstacle_1"
            )
        )
        self.assertFalse(
            evaluator.label_matches_active_obstacle(
                "red milk carton", "yellow_book_obstacle_1"
            )
        )
        self.assertIsNone(
            evaluator.label_matches_active_obstacle(
                "gray rectangular binder", "some_obstacle_1"
            )
        )

    def test_manifest_source_hashes_are_fail_closed(self):
        evaluator = self.evaluator
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bddl = root / "task.bddl"
            states = root / "task.pruned_init"
            bddl.write_text("task", encoding="utf-8")
            states.write_text("states", encoding="utf-8")
            case = {
                "schema_version": evaluator.MANIFEST_SCHEMA,
                "protocol_id": evaluator.PROTOCOL_ID,
                "case_id": "case-1",
                "source_commit": evaluator.UPSTREAM_COMMIT,
                "action_space": "translational_only",
                "suite": "safelibero_spatial",
                "max_steps": 300,
                "environment_seed": evaluator.TABLE_ENVIRONMENT_SEED,
                "settle_actions": evaluator.TABLE_SETTLE_ACTIONS,
                "model_action_horizon": (
                    evaluator.TABLE_MODEL_ACTION_HORIZON
                ),
                "replan_steps": evaluator.TABLE_REPLAN_STEPS,
                "required_arms": [
                    "pi05_translational",
                    "pi05_plus_aegis_translational",
                ],
                "semantic_label_requirement": (
                    "frozen_codex_label_bound_to_agentview_sha256"
                ),
                "bddl_path": "task.bddl",
                "bddl_sha256": evaluator.sha256_path(bddl),
                "initial_states_path": "task.pruned_init",
                "initial_states_sha256": evaluator.sha256_path(states),
            }
            evaluator.validate_case_row(case, root)
            bddl.write_text("changed", encoding="utf-8")
            with self.assertRaises(evaluator.ProtocolError):
                evaluator.validate_case_row(case, root)

    def test_atomic_json_has_no_temporary_file(self):
        evaluator = self.evaluator
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            digest = evaluator.atomic_write_json(
                output, {"status": "complete"}
            )
            self.assertEqual(
                json.loads(output.read_text()), {"status": "complete"}
            )
            self.assertEqual(digest, evaluator.sha256_path(output))
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_policy_noise_schedule_is_query_indexed(self):
        evaluator = self.evaluator
        self.assertEqual(
            evaluator.query_seed(2_026_071_700, 0), 2_026_071_700
        )
        self.assertEqual(
            evaluator.query_seed(2_026_071_700, 9), 2_026_071_709
        )
        with self.assertRaises(evaluator.ProtocolError):
            evaluator.query_seed(2**32 - 1, 1)

    def test_released_six_variable_qp_structure_is_preserved(self):
        source = inspect.getsource(self.evaluator._aegis_action)
        solver_source = inspect.getsource(
            self.evaluator._solve_aegis_qp
        )
        self.assertIn("cp.Variable(6)", source)
        self.assertIn("[1.0 / 25.0] * 3 + [1.0] * 3", source)
        self.assertIn("10.0 * h >= 0", source)
        self.assertIn("0.2 * R1 @ u_v", source)
        self.assertIn("problem.solve(solver=cp.OSQP)", solver_source)
        self.assertIn("raise MethodFailure", solver_source)
        self.assertNotIn("raise ApparatusError", solver_source)

    def test_osqp_exception_is_a_retained_method_failure(self):
        class Problem:
            def solve(self, *, solver):
                raise RuntimeError(f"solver {solver} failed")

        class CP:
            OSQP = "OSQP"

        with self.assertRaisesRegex(
            self.evaluator.MethodFailure,
            "AEGIS OSQP execution failed",
        ):
            self.evaluator._solve_aegis_qp(Problem(), CP)

    def test_precontrol_geometry_failure_contract_is_exact(self):
        failure = {
            "status": "method_failure",
            "method_failure": {
                "component": "aegis_geometry",
                "phase": "precontrol",
                "step": 0,
                "safety_by_no_execution": True,
            },
        }
        self.assertTrue(
            self.evaluator._is_precontrol_geometry_failure(failure)
        )
        for field, replacement in (
            ("phase", "control"),
            ("step", 1),
            ("safety_by_no_execution", False),
        ):
            invalid = json.loads(json.dumps(failure))
            invalid["method_failure"][field] = replacement
            self.assertFalse(
                self.evaluator._is_precontrol_geometry_failure(invalid)
            )

    def test_geometry_fitting_errors_are_method_failures(self):
        source = inspect.getsource(
            self.evaluator._prepare_aegis_geometry
        )
        self.assertIn("released ConvexHull/MVEE fitting failed", source)
        self.assertIn("raise MethodFailure", source)

    def test_settled_pairing_binds_depth_backview_and_simulator_state(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is unavailable")
        observation = {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "agentview_depth": np.zeros((2, 2), dtype=np.float32),
            "backview_image": np.ones((2, 2, 3), dtype=np.uint8),
            "backview_depth": np.ones((2, 2), dtype=np.float32),
            "robot0_eye_in_hand_image": np.full(
                (2, 2, 3), 2, dtype=np.uint8
            ),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array(
                [0, 0, 0, 1], dtype=np.float32
            ),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "milk_obstacle_1_pos": np.ones(3, dtype=np.float32),
        }
        first = self.evaluator.settled_input_contract(
            observation,
            "pick the bowl",
            active_obstacle_name="milk_obstacle_1",
            settled_simulator_state=np.arange(4, dtype=np.float64),
        )
        changed = dict(observation)
        changed["backview_depth"] = np.full(
            (2, 2), 3, dtype=np.float32
        )
        second = self.evaluator.settled_input_contract(
            changed,
            "pick the bowl",
            active_obstacle_name="milk_obstacle_1",
            settled_simulator_state=np.arange(4, dtype=np.float64),
        )
        self.assertNotEqual(
            self.evaluator.sha256_bytes(
                self.evaluator.canonical_json_bytes(first)
            ),
            self.evaluator.sha256_bytes(
                self.evaluator.canonical_json_bytes(second)
            ),
        )

    def test_video_is_streamed_instead_of_accumulated(self):
        source = inspect.getsource(self.evaluator.evaluate_case)
        self.assertIn("get_writer", source)
        self.assertIn("append_data", source)
        self.assertNotIn("replay_images", source)
        self.assertIn(
            "frames_written == len(executed_actions) + 1",
            source,
        )

    def test_capture_runtime_cannot_execute_outcome_components(self):
        source = inspect.getsource(
            self.capture._capture_runtime
        ).lower()
        forbidden = (
            "openpi",
            "groundingdino",
            "cvxpy",
            "obstacle_detection",
            "fit_ellipse",
            "osqp",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_capture_seeds_numpy_before_environment_construction(self):
        source = inspect.getsource(self.capture.capture_case)
        seed_position = source.index(
            'np.random.seed(int(case["environment_seed"]))'
        )
        build_position = source.index(
            "env, task, observation, selected_initial_state = "
            "_build_environment("
        )
        self.assertLess(seed_position, build_position)

    def test_shared_environment_builder_seeds_numpy_before_construction(self):
        source = inspect.getsource(self.evaluator._build_environment)
        seed_position = source.index(
            'runtime["np"].random.seed(int(case["environment_seed"]))'
        )
        construction_position = source.index(
            'env = runtime["OffScreenRenderEnv"](**env_args)'
        )
        self.assertLess(seed_position, construction_position)

    def test_shared_environment_builder_keeps_controller_override_opt_in(self):
        signature = inspect.signature(self.evaluator._build_environment)
        self.assertIsNone(signature.parameters["controller"].default)
        self.assertIsNone(signature.parameters["control_frequency_hz"].default)
        source = inspect.getsource(self.evaluator._build_environment)
        self.assertIn('env_args["controller"] = controller', source)
        self.assertIn('env_args["control_freq"] = control_frequency_hz', source)
        self.assertLess(
            source.index('if controller is not None:'),
            source.index('env = runtime["OffScreenRenderEnv"](**env_args)'),
        )

    def test_capture_contract_hard_codes_zero_outcome_calls(self):
        source = CAPTURE_PATH.read_text(encoding="utf-8")
        for field in (
            '"policy_queries": 0',
            '"semantic_selector_calls": 0',
            '"grounding_calls": 0',
            '"mvee_calls": 0',
            '"qp_calls": 0',
            '"outcome_actions": 0',
        ):
            with self.subTest(field=field):
                self.assertIn(field, source)

    def test_capture_resume_requires_every_lossless_asset(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("NumPy/Pillow are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary)
            case = {"case_id": "case-1", "protocol_id": "protocol-1"}
            arrays = {
                "agentview_rgb": np.arange(
                    12, dtype=np.uint8
                ).reshape(2, 2, 3),
                "backview_rgb": np.arange(
                    12, 24, dtype=np.uint8
                ).reshape(2, 2, 3),
                "agentview_depth": np.arange(
                    4, dtype=np.float32
                ).reshape(2, 2),
                "backview_depth": np.arange(
                    4, 8, dtype=np.float32
                ).reshape(2, 2),
                "simulator_state": np.arange(5, dtype=np.float64),
            }
            assets = {
                name: self.capture._asset_record(
                    case_dir=case_dir,
                    name=name,
                    array=array,
                    np=np,
                    Image=(
                        Image
                        if name in {"agentview_rgb", "backview_rgb"}
                        else None
                    ),
                )
                for name, array in arrays.items()
            }
            record = {
                "schema_version": self.evaluator.CAPTURE_SCHEMA,
                "protocol_id": case["protocol_id"],
                "case_id": case["case_id"],
                "status": "complete",
                "case": case,
                "execution_counts": {
                    "reset": 1,
                    "initial_state_restore": 1,
                    "settle_actions": 20,
                    "policy_queries": 0,
                    "semantic_selector_calls": 0,
                    "grounding_calls": 0,
                    "point_cloud_filter_calls": 0,
                    "mvee_calls": 0,
                    "qp_calls": 0,
                    "outcome_actions": 0,
                },
                "settled_agentview_array_sha256": assets[
                    "agentview_rgb"
                ]["array_sha256"],
                "assets": assets,
            }
            self.assertTrue(
                self.capture._capture_resume_is_valid(
                    record, case=case, case_dir=case_dir
                )
            )
            (
                case_dir
                / assets["backview_rgb"]["png_path"]
            ).unlink()
            self.assertFalse(
                self.capture._capture_resume_is_valid(
                    record, case=case, case_dir=case_dir
                )
            )

    def test_evaluator_shaped_result_passes_independent_aggregator(self):
        evaluator = self.evaluator
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8").splitlines()[0]
        )
        label_record = {
            "schema_version": evaluator.LABEL_SCHEMA,
            "case_id": manifest["case_id"],
            "settled_agentview_array_sha256": "4" * 64,
            "obstacle_label": "red milk carton",
            "reviewer": "codex",
            "reviewed_at": "2026-07-17T00:00:00+00:00",
        }
        label_record_sha256 = evaluator.sha256_bytes(
            evaluator.canonical_json_bytes(label_record)
        )
        settled_contract = {
            "schema_version": self.aggregator.SETTLED_INPUT_SCHEMA,
            "agentview_array_sha256": "4" * 64,
            "agentview_depth_array_sha256": "5" * 64,
            "backview_array_sha256": "6" * 64,
            "backview_depth_array_sha256": "7" * 64,
            "wrist_array_sha256": "8" * 64,
            "state_array_sha256": "9" * 64,
            "active_obstacle_name": "milk_obstacle_1",
            "active_obstacle_position_array_sha256": "a" * 64,
            "settled_simulator_state_array_sha256": "b" * 64,
            "prompt": manifest["task_name"],
        }
        schedule = self.aggregator.expected_policy_noise_schedule(
            manifest
        )
        first_action_hash = "c" * 64
        result = {
            "schema_version": evaluator.RESULT_SCHEMA,
            "protocol_id": manifest["protocol_id"],
            "case_id": manifest["case_id"],
            "arm": "pi05_translational",
            "mode": "pi05",
            "status": "complete",
            "scientific_result": True,
            "terminal_reason": "task_success",
            "task_success": True,
            "suite": manifest["suite"],
            "safety_level": manifest["safety_level"],
            "logical_task_index": manifest["logical_task_index"],
            "resolved_task_index": manifest["resolved_task_index"],
            "task_name": manifest["task_name"],
            "episode_index": manifest["episode_index"],
            "pairing": {
                "manifest_row_sha256": evaluator.sha256_bytes(
                    evaluator.canonical_json_bytes(manifest)
                ),
                "initial_state_sha256": "0" * 64,
                "initial_observation_sha256": (
                    self.aggregator.canonical_record_sha256(
                        settled_contract
                    )
                ),
                "initial_observation_contract": settled_contract,
                "settled_simulator_state_sha256": "b" * 64,
                "settled_active_obstacle_position_sha256": "a" * 64,
                "policy_noise_schedule_id": manifest[
                    "policy_noise_schedule_id"
                ],
                "policy_noise_schedule_sha256": (
                    self.aggregator.canonical_record_sha256(schedule)
                ),
                "policy_noise_schedule": schedule,
                "initial_policy_action_chunk_sha256": first_action_hash,
                "semantic_label_record_sha256": label_record_sha256,
                "semantic_label_settled_agentview_sha256": "4" * 64,
                "semantic_obstacle_label": "red milk carton",
                "max_steps": manifest["max_steps"],
                "model_action_horizon": manifest[
                    "model_action_horizon"
                ],
                "replan_steps": manifest["replan_steps"],
                "translational_fail_open": (
                    evaluator.TRANSLATIONAL_FAIL_OPEN
                ),
            },
            "metrics": {
                "public_collision": False,
                "paper_collision": False,
                "paper_collision_avoidance": True,
                "paper_collision_threshold_m": 0.001,
                "maximum_active_obstacle_l1_displacement_m": 0.0,
                "collision_first_step": None,
                "task_success": True,
                "safety_by_no_execution": False,
                "legacy_ets_steps": 0,
                "executed_action_count": 1,
                "termination_reason": "task_success",
            },
            "settled_observation": {
                "agentview_array_sha256": "4" * 64,
                "label_record": label_record,
                "label_record_sha256": label_record_sha256,
                "obstacle_label": "red milk carton",
            },
            "obstacle": {
                "active_name": settled_contract["active_obstacle_name"],
            },
            "timing": {"started_unix": 2_000_000_000.0},
            "policy_queries": [
                {
                    "query_index": 0,
                    "rng_seed": schedule["query_seeds"][0],
                    "returned_action_shape": [
                        manifest["model_action_horizon"],
                        7,
                    ],
                    "returned_actions_sha256": first_action_hash,
                }
            ],
            "actions": [
                {
                    "step": 0,
                    "nominal_raw": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        -1.0,
                    ],
                    "nominal_translational": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                    ],
                    "executed": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                    ],
                    "control_path": "pi05_translational_nominal",
                    "modified": False,
                    "correction_l2": 0.0,
                    "qp": None,
                    "reward": 1.0,
                    "done": True,
                    "step_elapsed_seconds": 0.01,
                    "obstacle_l1_displacement_m": 0.0,
                    "robot_obstacle_contact": False,
                }
            ],
            "intervention": {
                "eligible_steps": 0,
                "intervention_count": 0,
                "intervention_rate": 0.0,
                "correction_l2_sum": 0.0,
                "correction_l2_max": 0.0,
            },
            "contact_telemetry": {
                "status": "available",
                "robot_active_obstacle_contact": False,
                "first_contact_step": None,
                "unique_contact_pairs": [],
            },
            "video": {
                "path": (
                    f"pi05/{manifest['case_id']}/episode.mp4"
                ),
                "sha256": "d" * 64,
                "frames": 2,
                "fps": 30,
                "complete_episode": True,
            },
        }
        goal_definition = {
            "schema_version": evaluator.GOAL_PROGRESS_SCHEMA,
            "source": "native_bddl_goal_predicates",
            "logic": "conjunction",
            "goal_atoms": [
                {
                    "index": 0,
                    "predicate": "in",
                    "arguments": ["target_object_1", "target_region_1"],
                }
            ],
        }
        goal_definition["goal_definition_sha256"] = (
            self.aggregator.canonical_record_sha256(goal_definition)
        )

        def snapshot(step, satisfied, previous, state_hash):
            return {
                "step": step,
                "values": [satisfied],
                "satisfied_count": int(satisfied),
                "fraction": float(satisfied),
                "all_satisfied": satisfied,
                "newly_satisfied_indices": (
                    [0] if satisfied and not previous else []
                ),
                "regressed_indices": (
                    [0] if previous and not satisfied else []
                ),
                "argument_poses": [
                    {
                        "atom_index": 0,
                        "arguments": [
                            {
                                "name": "target_object_1",
                                "object_state_type": "object",
                                "position": [0.0, 0.0, 0.1],
                                "quaternion": [1.0, 0.0, 0.0, 0.0],
                            },
                            {
                                "name": "target_region_1",
                                "object_state_type": "site",
                                "position": [0.1, 0.0, 0.1],
                                "quaternion": [0.0, 0.0, 0.0, 1.0],
                            },
                        ],
                    }
                ],
                "simulator_state_sha256_before": state_hash,
                "simulator_state_sha256_after": state_hash,
                "inert": True,
            }

        initial_goal = snapshot(-1, False, False, "e" * 64)
        final_goal = snapshot(0, True, False, "f" * 64)
        result["actions"][0]["goal_progress"] = final_goal
        result["goal_progress"] = {
            **goal_definition,
            "initial": initial_goal,
            "final": final_goal,
            "summary": {
                "initial_values": [False],
                "final_values": [True],
                "initial_satisfied_count": 0,
                "final_satisfied_count": 1,
                "maximum_satisfied_count": 1,
                "initial_fraction": 0.0,
                "final_fraction": 1.0,
                "maximum_fraction": 1.0,
                "ever_satisfied": [True],
                "first_satisfied_step": [0],
                "first_all_satisfied_step": 0,
                "regression_count": 0,
            },
        }
        result["terminal_observation"] = {
            "agentview_array_sha256": "1" * 64,
            "simulator_state_sha256": "2" * 64,
            "frame_index": 1,
            "after_executed_action_count": 1,
        }
        result["result_payload_sha256"] = (
            self.aggregator._result_payload_sha256(result)
        )
        validated = self.aggregator.validate_result(
            result, config=config, manifest=manifest
        )
        self.assertEqual(validated["case_id"], manifest["case_id"])

        invalid = dict(result, scientific_result=False)
        with self.assertRaises(self.aggregator.AggregationError):
            self.aggregator.validate_result(
                invalid, config=config, manifest=manifest
            )


if __name__ == "__main__":
    unittest.main()
