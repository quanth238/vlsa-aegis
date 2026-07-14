from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - dependency-free init environment
    jsonschema = None

from crfs_harness.manifest import (
    GENERATED_SOURCE_ESTIMAND,
    build_cases,
    build_generated_cases,
    generated_source_state_id,
    validate_case,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(
    index: int,
    state_hex: str = "a",
    bundle_hex: str = "b",
    branch_hex: str | None = None,
) -> dict:
    state_sha256 = state_hex * 64
    branch_sha256 = (branch_hex or state_hex) * 64
    return {
        "source_index": index,
        "source_estimand": GENERATED_SOURCE_ESTIMAND,
        "source_state_id": generated_source_state_id(branch_sha256),
        "source_branch_sha256": branch_sha256,
        "source_state_sha256": state_sha256,
        "source_bundle_sha256": bundle_hex * 64,
        "source_bundle_path": f"/immutable/generated/{index}",
        "environment_seed": 1000 + index,
    }


class GeneratedManifestTest(unittest.TestCase):
    def test_released_schema_v1_schedule_is_unchanged(self) -> None:
        expected = build_cases("safelibero_spatial", "II", 0, [2], 1, "test")
        self.assertEqual(expected[0]["schema_version"], "1.0")
        self.assertEqual(expected[0]["safety_level"], "II")
        self.assertEqual(expected[0]["group_id"], "safelibero_spatial:II:0:2")
        self.assertEqual(validate_case(expected[0]), [])

    def test_generated_schedule_is_deterministic_grouped_and_never_level_ii(self) -> None:
        kwargs = {
            "task_suite": "safelibero_spatial",
            "task_index": 0,
            "sources": [_source(7)],
            "seeds_per_source": 3,
            "seed_namespace": "retired-pilot",
        }
        first = build_generated_cases(**kwargs)
        second = build_generated_cases(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({case["case_id"] for case in first}), 3)
        self.assertEqual(len({case["group_id"] for case in first}), 1)
        self.assertTrue(all(case["schema_version"] == "2.0" for case in first))
        self.assertTrue(all(case["safety_level"] == "generated" for case in first))
        self.assertTrue(all(case["safety_level"] != "II" for case in first))
        self.assertFalse([error for case in first for error in validate_case(case)])

    def test_generated_constructor_refuses_outcome_conditioning_and_duplicates(self) -> None:
        source = _source(0)
        for forbidden in ("collision", "clearance_m", "progress_m", "planner_success", "probe_score"):
            mutated = {**source, forbidden: False}
            with self.assertRaisesRegex(ValueError, "non-identity fields"):
                build_generated_cases(
                    task_suite="safelibero_spatial",
                    task_index=0,
                    sources=[mutated],
                    seeds_per_source=1,
                    seed_namespace="test",
                )
        with self.assertRaisesRegex(ValueError, "duplicate generated source branch"):
            build_generated_cases(
                task_suite="safelibero_spatial",
                task_index=0,
                sources=[source, {**source, "source_index": 1}],
                seeds_per_source=1,
                seed_namespace="test",
            )
        with self.assertRaisesRegex(ValueError, "duplicate generated source_index"):
            build_generated_cases(
                task_suite="safelibero_spatial",
                task_index=0,
                sources=[source, {**_source(1, state_hex="c"), "source_index": 0}],
                seeds_per_source=1,
                seed_namespace="test",
            )
        for task_suite, task_index in (("wrong_suite", 0), ("safelibero_spatial", 1)):
            with self.subTest(task_suite=task_suite, task_index=task_index):
                with self.assertRaisesRegex(ValueError, "bound to safelibero_spatial task index 0"):
                    build_generated_cases(
                        task_suite=task_suite,
                        task_index=task_index,
                        sources=[source],
                        seeds_per_source=1,
                        seed_namespace="test",
                    )

    def test_validator_rejects_identity_path_and_level_tampering(self) -> None:
        case = build_generated_cases(
            task_suite="safelibero_spatial",
            task_index=0,
            sources=[_source(0)],
            seeds_per_source=1,
            seed_namespace="test",
        )[0]
        mutations = {
            "safety_level": "II",
            "source_state_id": "gsrc-" + "0" * 16,
            "source_branch_sha256": "0" * 64,
            "source_state_sha256": "not-a-hash",
            "source_bundle_sha256": "not-a-hash",
            "source_bundle_path": "",
            "group_id": "wrong",
            "case_id": "crfs-" + "0" * 16,
            "episode_index": 1,
            "task_suite": "wrong_suite",
            "task_index": 1,
            "collision": True,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                mutated = copy.deepcopy(case)
                mutated[key] = value
                self.assertTrue(validate_case(mutated))

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_case_schema_accepts_both_versions_and_rejects_cross_labeling(self) -> None:
        schema = json.loads((ROOT / "schemas/case-manifest.schema.json").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        released = build_cases("safelibero_spatial", "II", 0, [0], 1, "test")[0]
        generated = build_generated_cases(
            task_suite="safelibero_spatial",
            task_index=0,
            sources=[_source(0)],
            seeds_per_source=1,
            seed_namespace="test",
        )[0]
        jsonschema.validate(released, schema)
        jsonschema.validate(generated, schema)
        generated["safety_level"] = "II"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(generated, schema)


if __name__ == "__main__":
    unittest.main()
