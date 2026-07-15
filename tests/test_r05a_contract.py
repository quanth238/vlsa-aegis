import json
import hashlib
from pathlib import Path
import unittest
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "feature_list.json"
MANIFEST = ROOT / "manifests" / "r05a_inverse_flow_teacher_smoke.jsonl"
ADR = ROOT / "docs" / "decisions" / "0028-pivot-to-inverse-flow-transport.md"
LEDGER = ROOT / "EXPERIMENTS.md"
PROGRESS = ROOT / "PROGRESS.md"
ARCHIVED_PROGRESS = (
    ROOT / "docs" / "archive" / "progress" / "2026-07-15-pre-inverse-flow-pivot.md"
)
ARCHIVED_HANDOFF = (
    ROOT / "docs" / "archive" / "handoffs" / "2026-07-14-h03-session-handoff.md"
)

SMOKE_CASES = [
    "crfs-1069f29a8d76463a",
    "crfs-7eddaafffb4f9474",
    "crfs-bd7b0adf95145623",
]


class R05AContractTest(unittest.TestCase):
    def test_inverse_flow_is_the_only_active_gate(self):
        value = json.loads(FEATURES.read_text(encoding="utf-8"))
        features = {item["id"]: item for item in value["features"]}
        active = [item["id"] for item in value["features"] if item["status"] == "active"]
        self.assertEqual(active, ["R05A"])
        self.assertEqual(features["R03A"]["status"], "blocked")
        self.assertEqual(features["R04"]["status"], "blocked")
        self.assertEqual(features["R05A"]["dependencies"], ["R03"])
        self.assertIn("No inverse-flow action or outcome", features["R05A"]["evidence"])

    def test_smoke_manifest_is_exact_unique_development_subset(self):
        raw = MANIFEST.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633",
        )
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        self.assertEqual([row["case_id"] for row in rows], SMOKE_CASES)
        self.assertEqual(len({row["case_id"] for row in rows}), 3)
        self.assertEqual(len({row["group_id"] for row in rows}), 3)
        self.assertTrue(all(row["task_suite"] == "safelibero_spatial" for row in rows))
        self.assertTrue(all(row["safety_level"] == "II" for row in rows))
        self.assertTrue(all(row["task_index"] == 0 for row in rows))

        eligible = {
            row["case_id"]: row
            for line in (ROOT / "manifests" / "r03a_analytic_kill_test_eligible.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            for row in [json.loads(line)]
        }
        self.assertEqual(rows, [eligible[case_id] for case_id in SMOKE_CASES])

    def test_decision_freezes_transport_not_action_overwrite(self):
        text = ADR.read_text(encoding="utf-8")
        for case_id in SMOKE_CASES:
            self.assertIn(case_id, text)
        required = [
            "u_0` through `u_4` exactly zero",
            "active steps 5--9",
            "`P <= B`",
            "B/5`",
            "first-five XYZ",
            "may not query simulator outcomes",
            "final-step overwrite",
            "at least 14/17",
            "all nine historical constant-residual successes",
            "five of the eight historical misses",
        ]
        for phrase in required:
            self.assertIn(phrase, text)

    def test_ledger_blocks_training_and_uses_joint_outcome(self):
        text = LEDGER.read_text(encoding="utf-8")
        self.assertIn("IFT-00 — Local inverse-control contract", text)
        self.assertIn("IFT-01 — Three-case real transport smoke", text)
        self.assertIn("IFT-02 — Seventeen-case development population", text)
        self.assertIn("joint Safe-Progress Success", text)
        self.assertIn("Do not train a scalar ECG probe or residual-field MLP", text)
        self.assertIn("These 17 groups remain development", text)

    def test_current_tracker_cannot_launch_retired_population(self):
        current = PROGRESS.read_text(encoding="utf-8")
        self.assertNotIn("submit_r03a_grouped_array.sh", current)
        self.assertIn("Do not launch `r03a-analytic-kill-population-20260715a`", current)
        self.assertIn(
            "RUN_ID=r05a-inverse-flow-canary-20260715a "
            "scripts/hpc/submit_r05a_canary.sh",
            current,
        )
        self.assertIn(
            "c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb",
            current,
        )

        launcher = ROOT / "scripts" / "hpc" / "submit_r03a_grouped_array.sh"
        completed = subprocess.run(
            [str(launcher)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("retired unrun by ADR-0028", completed.stderr)

    def test_stale_records_are_preserved_but_not_authoritative(self):
        self.assertTrue(ARCHIVED_PROGRESS.is_file())
        self.assertTrue(ARCHIVED_HANDOFF.is_file())
        self.assertIn(
            "Current gate: H03 measurement audit",
            ARCHIVED_HANDOFF.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "R03A is active",
            ARCHIVED_PROGRESS.read_text(encoding="utf-8"),
        )
        self.assertNotIn("Current gate: H03", PROGRESS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
