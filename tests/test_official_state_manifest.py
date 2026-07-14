from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - dependency-free init environment
    jsonschema = None

from crfs_harness.manifest import build_cases, build_official_cases, validate_case
from crfs_harness.official_state import (
    canonical_init_state_relative_path,
    file_sha256,
    init_state_row_identity,
    verify_official_state_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_SUITE = "safelibero_spatial"
TASK_NAME = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
RELATIVE_PATH = f"{TASK_SUITE}/{TASK_NAME}_level_II.pruned_init"


class _Dtype:
    str = "<f8"


class _Row:
    dtype = _Dtype()

    def __init__(self, payload: bytes, shape: tuple[int, ...] = (3,)) -> None:
        self._payload = payload
        self.shape = shape

    def tobytes(self, order: str = "C") -> bytes:
        if order != "C":
            raise AssertionError("official identity must use C-order bytes")
        return self._payload


def _build_case(init_states_root: Path, row: _Row | None = None) -> dict:
    path = init_states_root / RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"official-safe-libero-archive")
    row = row or _Row(b"row-seven-bytes")
    return build_official_cases(
        task_suite=TASK_SUITE,
        safety_level="II",
        task_index=1,
        task_name=TASK_NAME,
        init_state_relative_path=RELATIVE_PATH,
        init_state_file_sha256=file_sha256(path),
        row_identities={7: init_state_row_identity(row)},
        seeds_per_episode=1,
        seed_namespace="official-transfer-v1",
    )[0]


class OfficialStateManifestTest(unittest.TestCase):
    def test_schema_v3_is_deterministic_and_content_binds_official_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _build_case(root)
            second = _build_case(root)
            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], "3.0")
            self.assertEqual(first["task_name"], TASK_NAME)
            self.assertEqual(first["init_state_relative_path"], RELATIVE_PATH)
            self.assertEqual(first["init_state_row_dtype"], "<f8")
            self.assertEqual(first["init_state_row_shape"], [3])
            self.assertEqual(
                first["init_state_row_bytes_sha256"],
                hashlib.sha256(b"row-seven-bytes").hexdigest(),
            )
            self.assertIn(":ostate-", first["group_id"])
            self.assertEqual(validate_case(first), [])

            changed = _build_case(root, _Row(b"different-row"))
            self.assertNotEqual(first["case_id"], changed["case_id"])
            self.assertNotEqual(first["group_id"], changed["group_id"])

    def test_schema_v3_validator_rejects_every_identity_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = _build_case(Path(directory))
        mutations = {
            "task_name": "wrong_task",
            "init_state_relative_path": "safelibero_spatial/wrong_level_II.pruned_init",
            "init_state_file_sha256": "0" * 64,
            "init_state_row_dtype": "<f4",
            "init_state_row_shape": [4],
            "init_state_row_bytes_sha256": "1" * 64,
            "episode_index": 8,
            "environment_seed": case["environment_seed"] + 1,
            "random_control_seed": case["random_control_seed"] + 1,
            "group_id": "wrong-group",
            "case_id": "crfs-" + "0" * 16,
            "unexpected_outcome": False,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                mutated = copy.deepcopy(case)
                mutated[key] = value
                self.assertTrue(validate_case(mutated))

    def test_runtime_verifier_recomputes_task_path_file_and_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _Row(b"row-seven-bytes")
            case = _build_case(root, row)
            path = root / RELATIVE_PATH
            self.assertEqual(
                verify_official_state_bindings(
                    case,
                    actual_task_name=TASK_NAME,
                    init_states_root=root,
                    init_state_path=path,
                    row=row,
                ),
                [],
            )

            wrong_task = verify_official_state_bindings(
                case,
                actual_task_name="different_task",
                init_states_root=root,
                init_state_path=path,
                row=row,
            )
            self.assertTrue(any("task_name" in error for error in wrong_task))

            wrong_row = verify_official_state_bindings(
                case,
                actual_task_name=TASK_NAME,
                init_states_root=root,
                init_state_path=path,
                row=_Row(b"different-row"),
            )
            self.assertTrue(any("row_bytes_sha256" in error for error in wrong_row))

            path.write_bytes(b"changed-archive")
            wrong_file = verify_official_state_bindings(
                case,
                actual_task_name=TASK_NAME,
                init_states_root=root,
                init_state_path=path,
                row=row,
            )
            self.assertTrue(any("file_sha256" in error for error in wrong_file))

    def test_relative_path_is_portable_and_cannot_escape_root(self) -> None:
        self.assertEqual(canonical_init_state_relative_path(RELATIVE_PATH), RELATIVE_PATH)
        for value in ("/absolute/state", "../escape", "suite/../escape", "suite\\state"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    canonical_init_state_relative_path(value)

    def test_legacy_builder_remains_schema_v1_byte_for_field_compatible(self) -> None:
        legacy = build_cases(TASK_SUITE, "II", 0, [2], 1, "test")[0]
        self.assertEqual(
            set(legacy),
            {
                "schema_version",
                "case_id",
                "task_suite",
                "safety_level",
                "task_index",
                "episode_index",
                "environment_seed",
                "policy_seed",
                "random_control_seed",
                "group_id",
            },
        )
        self.assertEqual(legacy["schema_version"], "1.0")
        self.assertEqual(validate_case(legacy), [])

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_json_schema_accepts_v1_and_v3_and_rejects_cross_labeling(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/case-manifest.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        legacy = build_cases(TASK_SUITE, "II", 0, [0], 1, "test")[0]
        with tempfile.TemporaryDirectory() as directory:
            official = _build_case(Path(directory))
        jsonschema.validate(legacy, schema)
        jsonschema.validate(official, schema)
        official["schema_version"] = "1.0"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(official, schema)


if __name__ == "__main__":
    unittest.main()
