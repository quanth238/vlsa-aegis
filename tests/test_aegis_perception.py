from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main" / "crfs_oracle" / "aegis_perception.py"
SPEC = importlib.util.spec_from_file_location("crfs_aegis_perception", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load aegis_perception.py")
PERCEPTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PERCEPTION
SPEC.loader.exec_module(PERCEPTION)

AegisPerceptionError = PERCEPTION.AegisPerceptionError
AegisPerceptionMethodError = PERCEPTION.AegisPerceptionMethodError
CodexSemanticLabelLedger = PERCEPTION.CodexSemanticLabelLedger
GroundingDinoOnlyDependencies = PERCEPTION.GroundingDinoOnlyDependencies
agentview_1024_sha256 = PERCEPTION.agentview_1024_sha256
build_public_obstacle_prompt = PERCEPTION.build_public_obstacle_prompt
call_public_glm_obstacle_selector = PERCEPTION.call_public_glm_obstacle_selector
clean_public_obstacle_response = PERCEPTION.clean_public_obstacle_response
load_codex_semantic_label_jsonl = PERCEPTION.load_codex_semantic_label_jsonl
preferred_obstacles = PERCEPTION.preferred_obstacles
resolve_groundingdino_only_dependencies = (
    PERCEPTION.resolve_groundingdino_only_dependencies
)
resolve_original_perception_dependencies = (
    PERCEPTION.resolve_original_perception_dependencies
)
run_codex_frozen_label_groundingdino_backend = (
    PERCEPTION.run_codex_frozen_label_groundingdino_backend
)
run_codex_frozen_label_groundingdino_backend_for_test = (
    PERCEPTION._run_codex_frozen_label_groundingdino_backend_for_test
)
validate_codex_semantic_label_record = PERCEPTION.validate_codex_semantic_label_record


class _Message:
    content = "<|begin_of_box|>white storage box<|end_of_box|>"


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class _Completions:
    def __init__(self, ledger: list[dict]) -> None:
        self.ledger = ledger

    def create(self, **kwargs):
        self.ledger.append(kwargs)
        return _Response()


class _Chat:
    def __init__(self, ledger: list[dict]) -> None:
        self.completions = _Completions(ledger)


class _Client:
    def __init__(self, ledger: list[dict]) -> None:
        self.chat = _Chat(ledger)


class _FakeDType:
    def __init__(self, text: str) -> None:
        self.str = text
        self.hasobject = False

    def __str__(self) -> str:
        return "uint8" if self.str == "|u1" else "float64"


class _FakeArray:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        payload: bytes,
        values,
        dtype: str = "<f8",
    ) -> None:
        self.shape = shape
        self.dtype = _FakeDType(dtype)
        self._payload = payload
        self._values = values
        self.nbytes = len(payload)

    def tobytes(self, *, order: str) -> bytes:
        if order != "C":
            raise AssertionError("test arrays support only C order")
        return self._payload

    def tolist(self):
        return self._values


class _FakeFinite:
    def all(self) -> bool:
        return True


class _FakeNumpy:
    @staticmethod
    def asarray(value):
        if not isinstance(value, _FakeArray):
            raise AssertionError(f"unexpected fake array input: {type(value).__name__}")
        return value

    @staticmethod
    def ascontiguousarray(value):
        return _FakeNumpy.asarray(value)

    @staticmethod
    def isfinite(_value):
        return _FakeFinite()

    @staticmethod
    def vstack(values):
        rows = sum(int(value.shape[0]) for value in values)
        payload = b"".join(value.tobytes(order="C") for value in values)
        nested = []
        for value in values:
            nested.extend(value.tolist())
        return _FakeArray((rows, 3), payload=payload, values=nested)


class _FakePublicUtils:
    def __init__(self, points_by_view: dict[str, _FakeArray]) -> None:
        self.points_by_view = points_by_view
        self.calls: list[tuple] = []
        self.filtered = _FakeArray(
            (3, 3),
            payload=b"filtered-points",
            values=[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
        )
        self.fit = (
            _FakeArray((3,), payload=b"p2", values=[0.1, 0.2, 1.0]),
            _FakeArray(
                (3, 3),
                payload=b"r2",
                values=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            _FakeArray((3,), payload=b"q2", values=[0.1, 0.2, 0.3]),
        )

    def get_point_cloud(
        self,
        image,
        depth,
        environment,
        view,
        text_prompt,
        model,
        save_path,
    ):
        self.calls.append(
            (
                "get_point_cloud",
                view,
                text_prompt,
                environment,
                model,
                str(save_path),
                image,
                depth,
            )
        )
        return self.points_by_view[view]

    def filtering_points(self, points, task_suite):
        self.calls.append(("filtering_points", points, task_suite))
        return self.filtered

    def fit_ellipse(self, points, *, plot, save_path):
        self.calls.append(("fit_ellipse", points, plot, str(save_path)))
        return self.fit


class AegisPerceptionTest(unittest.TestCase):
    def test_prompt_and_preference_list_match_public_source(self) -> None:
        self.assertEqual(
            preferred_obstacles("safelibero_spatial"),
            [
                "yellow rectangular book",
                "blue moka pot",
                "red mug",
                "white storage box",
                "black wine bottle",
                "red milk carton",
            ],
        )
        self.assertEqual(
            preferred_obstacles("safelibero_long")[-1], "gray rectangular binder"
        )
        prompt = build_public_obstacle_prompt("reach the bowl", "safelibero_spatial")
        self.assertIn("The robot must follow this instruction: reach the bowl.", prompt)
        self.assertTrue(
            prompt.endswith("Output only the object name, with no additional words.")
        )

    def test_clean_response_removes_only_public_box_tokens(self) -> None:
        self.assertEqual(
            clean_public_obstacle_response(
                "<|begin_of_box|>white storage box<|end_of_box|>"
            ),
            "white storage box",
        )

    def test_dependency_preflight_fails_before_recording_secret(self) -> None:
        with self.assertRaises(AegisPerceptionError) as captured:
            resolve_original_perception_dependencies(
                {}, require_packages=False, require_cuda=False
            )
        self.assertEqual(captured.exception.code, "missing_glm45v_credential")
        self.assertEqual(captured.exception.details, {})
        self.assertIn("ZHIPUAI_API_KEY", str(captured.exception))

    def test_dependency_preflight_binds_files_and_returns_no_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "GroundingDINO_SwinT_OGC.py"
            checkpoint = root / "groundingdino_swint_ogc.pth"
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"checkpoint")
            environment = {
                "ZHIPUAI_API_KEY": "top-secret-value",
                "AEGIS_GROUNDING_DINO_CONFIG": str(config),
                "AEGIS_GROUNDING_DINO_CHECKPOINT": str(checkpoint),
                "CUDA_VISIBLE_DEVICES": "0",
            }
            resolved = resolve_original_perception_dependencies(
                environment,
                expected_config_sha256=hashlib.sha256(b"config").hexdigest(),
                expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
                require_packages=False,
            ).to_dict()
            self.assertTrue(resolved["api_key_present"])
            self.assertNotIn("top-secret-value", str(resolved))
            self.assertTrue(resolved["cuda_visible"])

    def test_dependency_preflight_rejects_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "GroundingDINO_SwinT_OGC.py"
            checkpoint = root / "groundingdino_swint_ogc.pth"
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"checkpoint")
            environment = {
                "ZHIPUAI_API_KEY": "secret",
                "AEGIS_GROUNDING_DINO_CONFIG": str(config),
                "AEGIS_GROUNDING_DINO_CHECKPOINT": str(checkpoint),
                "CUDA_VISIBLE_DEVICES": "0",
            }
            with self.assertRaises(AegisPerceptionError) as captured:
                resolve_original_perception_dependencies(
                    environment,
                    expected_config_sha256="0" * 64,
                    require_packages=False,
                )
            self.assertEqual(
                captured.exception.code, "groundingdino_config_hash_mismatch"
            )

    def test_glm_call_uses_public_settings_without_returning_secret(self) -> None:
        ledger: list[dict] = []

        def factory(*, api_key: str):
            self.assertEqual(api_key, "secret")
            return _Client(ledger)

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "obstacle_detection.png"
            image.write_bytes(b"fake-png")
            result = call_public_glm_obstacle_selector(
                image,
                instruction="reach the bowl",
                task_suite_name="safelibero_spatial",
                api_key="secret",
                client_factory=factory,
            )
        self.assertEqual(len(ledger), 1)
        request = ledger[0]
        self.assertEqual(request["model"], "glm-4.5v")
        self.assertEqual(request["temperature"], 0.1)
        self.assertEqual(request["top_p"], 0.1)
        self.assertEqual(request["thinking"], {"type": "enabled"})
        self.assertEqual(result["cleaned_obstacle_name"], "white storage box")
        self.assertFalse(result["credential_value_recorded"])
        self.assertNotIn("secret", str(result))

    @staticmethod
    def _label_value(
        image_sha256: str = "a" * 64,
        *,
        case_id: str = "crfs-1069f29a8d76463a",
    ) -> dict:
        return {
            "schema_version": "aegis_codex_semantic_label.v1",
            "case_id": case_id,
            "instruction": "put the bowl on the plate",
            "agentview_image_sha256": image_sha256,
            "obstacle_label": "white storage box",
            "allowed_label_vocabulary": list(
                PERCEPTION.PUBLIC_OBSTACLE_LABEL_VOCABULARY
            ),
            "reviewer": "codex",
            "reviewed_at": "2026-07-17T03:30:00Z",
        }

    def test_codex_label_schema_is_closed_public_and_has_no_simulator_name(
        self,
    ) -> None:
        record = validate_codex_semantic_label_record(self._label_value())
        self.assertEqual(record.reviewer, "codex")
        self.assertIn(
            record.obstacle_label,
            PERCEPTION.PUBLIC_OBSTACLE_LABEL_VOCABULARY,
        )
        self.assertNotIn("simulator", str(record.to_dict()).lower())

        extra = self._label_value()
        extra["simulator_name"] = "obstacle_white_storage_box"
        with self.assertRaises(PERCEPTION.AegisPerceptionApparatusError) as captured:
            validate_codex_semantic_label_record(extra)
        self.assertEqual(captured.exception.code, "invalid_codex_label_record_keys")

        wrong_reviewer = self._label_value()
        wrong_reviewer["reviewer"] = "human"
        with self.assertRaises(PERCEPTION.AegisPerceptionApparatusError):
            validate_codex_semantic_label_record(wrong_reviewer)

    def test_codex_label_rejects_missing_or_tampered_chronology_fields(self) -> None:
        for field in ("reviewed_at", "allowed_label_vocabulary"):
            missing = self._label_value()
            del missing[field]
            with self.subTest(field=field), self.assertRaises(
                PERCEPTION.AegisPerceptionApparatusError
            ) as captured:
                validate_codex_semantic_label_record(missing)
            self.assertEqual(
                captured.exception.code,
                "invalid_codex_label_record_keys",
            )

        tampered_vocabulary = self._label_value()
        tampered_vocabulary["allowed_label_vocabulary"] = list(
            reversed(tampered_vocabulary["allowed_label_vocabulary"])
        )
        with self.assertRaises(PERCEPTION.AegisPerceptionApparatusError) as caught:
            validate_codex_semantic_label_record(tampered_vocabulary)
        self.assertEqual(caught.exception.code, "invalid_codex_label_vocabulary")

        for invalid_timestamp in (
            "",
            "2026-07-17",
            "2026-07-17T03:30:00+00:00",
            "2026-02-30T03:30:00Z",
        ):
            tampered_time = self._label_value()
            tampered_time["reviewed_at"] = invalid_timestamp
            with self.subTest(reviewed_at=invalid_timestamp), self.assertRaises(
                PERCEPTION.AegisPerceptionApparatusError
            ) as captured:
                validate_codex_semantic_label_record(tampered_time)
            self.assertEqual(
                captured.exception.code,
                "invalid_codex_label_reviewed_at",
            )

    def test_codex_label_jsonl_is_hash_bound_unique_and_immutable(self) -> None:
        rows = [
            self._label_value(),
            self._label_value(case_id="crfs-3bd38b2879b8b0a9"),
        ]
        payload = b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
        )
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codex-labels.jsonl"
            path.write_bytes(payload)
            ledger = load_codex_semantic_label_jsonl(
                path,
                expected_sha256=digest,
            )
            self.assertEqual(len(ledger.records), 2)
            self.assertEqual(
                ledger.for_case("crfs-3bd38b2879b8b0a9").reviewer,
                "codex",
            )
            with self.assertRaises(PERCEPTION.AegisPerceptionApparatusError):
                load_codex_semantic_label_jsonl(
                    path,
                    expected_sha256="0" * 64,
                )

            duplicate = payload + json.dumps(rows[0]).encode("utf-8") + b"\n"
            path.write_bytes(duplicate)
            with self.assertRaises(PERCEPTION.AegisPerceptionApparatusError) as caught:
                load_codex_semantic_label_jsonl(
                    path,
                    expected_sha256=hashlib.sha256(duplicate).hexdigest(),
                )
            self.assertEqual(caught.exception.code, "duplicate_codex_label_case")

    def test_groundingdino_only_preflight_requires_hashes_but_no_glm_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "GroundingDINO_SwinT_OGC.py"
            checkpoint = root / "groundingdino_swint_ogc.pth"
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"checkpoint")
            environment = {
                "AEGIS_GROUNDING_DINO_CONFIG": str(config),
                "AEGIS_GROUNDING_DINO_CHECKPOINT": str(checkpoint),
                "CUDA_VISIBLE_DEVICES": "0",
            }
            resolved = resolve_groundingdino_only_dependencies(
                environment,
                expected_config_sha256=hashlib.sha256(b"config").hexdigest(),
                expected_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
                require_packages=False,
            )
        self.assertTrue(resolved.cuda_visible)
        self.assertNotIn("zai", resolved.packages)
        self.assertNotIn("ZHIPUAI_API_KEY", str(resolved.to_dict()))

    def test_frozen_label_backend_calls_only_public_geometry_functions(self) -> None:
        np = _FakeNumpy()
        agent_image = _FakeArray(
            (1024, 1024, 3),
            payload=b"reviewed-agent-image",
            values=[],
            dtype="|u1",
        )
        back_image = _FakeArray(
            (1024, 1024, 3),
            payload=b"back-image",
            values=[],
            dtype="|u1",
        )
        agent_depth = _FakeArray(
            (1024, 1024),
            payload=b"agent-depth",
            values=[],
        )
        back_depth = _FakeArray(
            (1024, 1024),
            payload=b"back-depth",
            values=[],
        )
        image_sha256 = agentview_1024_sha256(
            agent_image,
            numpy_module=np,
        )
        record = validate_codex_semantic_label_record(self._label_value(image_sha256))
        ledger = CodexSemanticLabelLedger(
            path="/immutable/codex-labels.jsonl",
            sha256="b" * 64,
            records=(record,),
        )
        agent_points = _FakeArray(
            (2, 3),
            payload=b"agent-points",
            values=[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        )
        back_points = _FakeArray(
            (1, 3),
            payload=b"back-points",
            values=[[0.0, 0.1, 1.0]],
        )
        public_utils = _FakePublicUtils(
            {"agentview": agent_points, "backview": back_points}
        )
        dependencies = GroundingDinoOnlyDependencies(
            config_path=PERCEPTION.REGISTERED_GROUNDING_DINO_CONFIG_PATH,
            config_sha256=PERCEPTION.REGISTERED_GROUNDING_DINO_CONFIG_SHA256,
            checkpoint_path=PERCEPTION.REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH,
            checkpoint_sha256=(PERCEPTION.REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256),
            packages={},
            cuda_visible=True,
        )
        model = object()

        def point_cloud_runner(**kwargs):
            points = kwargs["public_utils"].get_point_cloud(
                kwargs["image"],
                kwargs["depth"],
                kwargs["environment"],
                kwargs["view"],
                kwargs["obstacle_label"],
                kwargs["model"],
                kwargs["output_dir"],
            )
            return points, {
                "caption": kwargs["obstacle_label"],
                "box_threshold": 0.35,
                "text_threshold": 0.25,
                "device": "cuda",
                "boxes": {
                    "shape": [1, 4],
                    "values": [[0.5, 0.5, 0.2, 0.2]],
                },
                "logits": {"shape": [1], "values": [0.9]},
                "phrases": [kwargs["obstacle_label"]],
                "selected_box_index": 0,
                "selected_box": [0.5, 0.5, 0.2, 0.2],
                "selected_phrase": kwargs["obstacle_label"],
            }

        runtime_environment = object()

        class _Runtime:
            env = runtime_environment

        views = {
            "agentview_image": agent_image,
            "agentview_depth": agent_depth,
            "backview_image": back_image,
            "backview_depth": back_depth,
            "public_double_axis_flip_applied": True,
            "render_advanced_physics": False,
            "camera_orientation": {
                "robosuite_version": "1.4.1",
                "live_image_convention": "opengl",
                "image_convention_mapping_step": 1,
                "observable_mapping_applied_before_public_flip": True,
                "same_state_224_observable_anchor_passed": True,
                "same_state_224_policy_agent_anchor_passed": True,
                "controller_state_fingerprint_sha256": "c" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_codex_frozen_label_groundingdino_backend_for_test(
                label_ledger=ledger,
                case_id=record.case_id,
                views=views,
                instruction=record.instruction,
                task_suite="safelibero_spatial",
                runtime=_Runtime(),
                dependencies=dependencies,
                output_dir=Path(directory) / "audit",
                repo_root=ROOT,
                public_utils=public_utils,
                model_loader=lambda _dependencies: model,
                point_cloud_runner=point_cloud_runner,
                numpy_module=np,
            )
        self.assertEqual(
            result["backend_kind"],
            "injected_test_double_codex_frozen_label_groundingdino",
        )
        self.assertTrue(result["test_injection_used"])
        self.assertFalse(result["release_runtime_backend"])
        self.assertFalse(result["canonical_original_end_to_end"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["original_glm_executed"])
        self.assertFalse(result["oracle_substitution_used"])
        self.assertNotIn("original_end_to_end_only", str(result))
        self.assertNotIn("simulator_name", str(result))
        self.assertEqual(
            [call[0] for call in public_utils.calls],
            [
                "get_point_cloud",
                "get_point_cloud",
                "filtering_points",
                "fit_ellipse",
            ],
        )
        self.assertEqual(
            result["groundingdino"]["views"]["agentview"]["phrases"],
            ["white storage box"],
        )
        self.assertEqual(result["point_cloud"]["combined"]["count"], 3)
        self.assertEqual(result["point_cloud"]["filtered"]["count"], 3)
        self.assertEqual(result["ellipsoid"]["p2"], [0.1, 0.2, 1.0])
        self.assertTrue(
            result["inputs"]["orientation"][
                "physical_equivalence_to_observation_api_claimed"
            ]
        )

    def test_release_backend_has_no_dependency_injection_or_prebuilt_records(
        self,
    ) -> None:
        parameters = inspect.signature(
            run_codex_frozen_label_groundingdino_backend
        ).parameters
        forbidden = {
            "label_ledger",
            "dependencies",
            "_public_utils",
            "_model_loader",
            "_point_cloud_runner",
            "_numpy_module",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))
        self.assertIn("label_jsonl_path", parameters)
        self.assertIn("expected_label_jsonl_sha256", parameters)
        self.assertIn("environment", parameters)

    def test_release_backend_reloads_label_and_rehashes_registered_assets(
        self,
    ) -> None:
        ledger = object()
        dependencies = object()
        with (
            mock.patch.object(
                PERCEPTION,
                "load_codex_semantic_label_jsonl",
                return_value=ledger,
            ) as load_labels,
            mock.patch.object(
                PERCEPTION,
                "resolve_groundingdino_only_dependencies",
                return_value=dependencies,
            ) as resolve_assets,
            mock.patch.object(
                PERCEPTION,
                "_run_codex_frozen_label_groundingdino_core",
                return_value={"status": "sentinel"},
            ) as run_core,
        ):
            result = run_codex_frozen_label_groundingdino_backend(
                label_jsonl_path="/immutable/labels.jsonl",
                expected_label_jsonl_sha256="a" * 64,
                case_id="crfs-1069f29a8d76463a",
                views={},
                instruction="put the bowl on the plate",
                task_suite="safelibero_spatial",
                runtime=object(),
                environment={"CUDA_VISIBLE_DEVICES": "0"},
                output_dir="/audit",
                repo_root=ROOT,
            )
        self.assertEqual(result, {"status": "sentinel"})
        load_labels.assert_called_once_with(
            "/immutable/labels.jsonl",
            expected_sha256="a" * 64,
        )
        resolve_assets.assert_called_once_with(
            {"CUDA_VISIBLE_DEVICES": "0"},
            expected_config_sha256=(
                PERCEPTION.REGISTERED_GROUNDING_DINO_CONFIG_SHA256
            ),
            expected_checkpoint_sha256=(
                PERCEPTION.REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256
            ),
        )
        self.assertIs(run_core.call_args.kwargs["label_ledger"], ledger)
        self.assertIs(run_core.call_args.kwargs["dependencies"], dependencies)
        self.assertFalse(run_core.call_args.kwargs["test_injection_used"])

    def test_prediction_audit_rejects_request_or_selected_box_forgery(self) -> None:
        audit = {
            "caption": "white storage box",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "device": "cuda",
            "boxes": {
                "shape": [1, 4],
                "values": [[0.5, 0.5, 0.2, 0.2]],
            },
            "logits": {"shape": [1], "values": [0.9]},
            "phrases": ["white storage box"],
            "selected_box_index": 0,
            "selected_box": [0.5, 0.5, 0.2, 0.2],
            "selected_phrase": "white storage box",
        }
        validated = PERCEPTION._validate_groundingdino_prediction_audit(
            audit,
            view="agentview",
            obstacle_label="white storage box",
        )
        self.assertEqual(validated, audit)

        for field, value in (
            ("device", "cpu"),
            ("caption", "red mug"),
            ("box_threshold", 0.5),
            ("selected_box", [0.1, 0.1, 0.1, 0.1]),
            ("selected_phrase", "red mug"),
        ):
            tampered = dict(audit)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaises(
                AegisPerceptionMethodError
            ):
                PERCEPTION._validate_groundingdino_prediction_audit(
                    tampered,
                    view="agentview",
                    obstacle_label="white storage box",
                )

    def test_prediction_audit_allows_one_public_camera_to_have_no_box(self) -> None:
        empty = {
            "caption": "white storage box",
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "device": "cuda",
            "boxes": {"shape": [0, 4], "values": []},
            "logits": {"shape": [0], "values": []},
            "phrases": [],
            "selected_box_index": None,
            "selected_box": None,
            "selected_phrase": None,
        }
        self.assertEqual(
            PERCEPTION._validate_groundingdino_prediction_audit(
                empty,
                view="backview",
                obstacle_label="white storage box",
            ),
            empty,
        )
        forged = dict(empty)
        forged["selected_box_index"] = 0
        with self.assertRaises(PERCEPTION.AegisPerceptionMethodError):
            PERCEPTION._validate_groundingdino_prediction_audit(
                forged,
                view="backview",
                obstacle_label="white storage box",
            )

    def test_setup_job_checkpoint_identity_is_frozen_in_backend(self) -> None:
        self.assertEqual(
            PERCEPTION.REGISTERED_GROUNDING_DINO_CONFIG_PATH,
            "/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/"
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        )
        self.assertEqual(
            PERCEPTION.REGISTERED_GROUNDING_DINO_CONFIG_SHA256,
            "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
        )
        self.assertEqual(
            PERCEPTION.REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH,
            "/mnt/data/quanth/cache/aegis/groundingdino/" "groundingdino_swint_ogc.pth",
        )
        self.assertEqual(
            PERCEPTION.REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256,
            "3b3ca2563c77c69f651d7bd133e97139" "c186df06231157a64c507099c52bc799",
        )
        self.assertEqual(
            PERCEPTION.PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256,
            "b21cd3b3c5035e4706cef272d0475b673bd9eb1d31165c9c0833e3ce1da74e0d",
        )


if __name__ == "__main__":
    unittest.main()
