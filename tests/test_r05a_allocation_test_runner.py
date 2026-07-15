from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "hpc" / "lib" / "r05a_allocation_tests.sh"
REGISTRY = ROOT / "main" / "crfs_oracle" / "r05a_allocation_tests.json"


class R05AAllocationTestRunnerTest(unittest.TestCase):
    def _run(
        self,
        *,
        registry: Path,
        fake_python: Path,
        test_dir: Path,
        output_log: Path,
        suite_log_dir: Path,
        offset: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_R05A_REGISTRY": str(registry),
                "FAKE_R05A_COUNT_OFFSET": str(offset),
            }
        )
        return subprocess.run(
            [
                "bash",
                "-c",
                '. "$1"; crfs_run_r05a_allocation_tests "$2" "$3" "$4" "$5" "$6"',
                "r05a-allocation-test-runner",
                str(HELPER),
                str(registry),
                str(fake_python),
                str(test_dir),
                str(output_log),
                str(suite_log_dir),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    def test_shared_runner_uses_registry_and_fails_closed_on_drift(self) -> None:
        subprocess.run(["bash", "-n", str(HELPER)], check=True)
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "pattern = sys.argv[sys.argv.index('-p') + 1]\n"
                "registry = json.load(open(os.environ['FAKE_R05A_REGISTRY'], encoding='utf-8'))\n"
                "counts = {item['pattern']: item['expected_tests'] for item in registry['suites']}\n"
                "count = counts[pattern] + int(os.environ.get('FAKE_R05A_COUNT_OFFSET', '0'))\n"
                "print(f'Ran {count} tests in 0.001s')\n"
                "print()\n"
                "print('OK')\n",
                encoding="utf-8",
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            test_dir = root / "tests"
            test_dir.mkdir()
            output_log = root / "focused.log"

            completed = self._run(
                registry=REGISTRY,
                fake_python=fake_python,
                test_dir=test_dir,
                output_log=output_log,
                suite_log_dir=root,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            lines = output_log.read_text(encoding="utf-8").splitlines()
            markers = [line for line in lines if line.startswith("verified_test_suite=")]
            self.assertEqual(len(markers), len(registry["suites"]))
            for item in registry["suites"]:
                self.assertIn(
                    "verified_test_suite={pattern} expected={count} observed={count} "
                    "skips=0 status=passed".format(
                        pattern=item["pattern"], count=item["expected_tests"]
                    ),
                    markers,
                )

            mismatch = self._run(
                registry=REGISTRY,
                fake_python=fake_python,
                test_dir=test_dir,
                output_log=root / "mismatch.log",
                suite_log_dir=root,
                offset=1,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("tests; expected", mismatch.stderr)

            invalid_registry = root / "invalid.json"
            invalid_registry.write_text(
                json.dumps({**registry, "unexpected": True}), encoding="utf-8"
            )
            invalid = self._run(
                registry=invalid_registry,
                fake_python=fake_python,
                test_dir=test_dir,
                output_log=root / "invalid.log",
                suite_log_dir=root,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("strict contract", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
