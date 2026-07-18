import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "fetch_vlsa_table1_video.py"
SPEC = importlib.util.spec_from_file_location("fetch_vlsa_table1_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FETCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FETCH
SPEC.loader.exec_module(FETCH)


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class FakeTransport(FETCH.VideoTransport):
    def __init__(self, payload, info=None, corrupt=False):
        self.payload = payload
        self.info = info or FETCH.RemoteFileInfo(
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.corrupt = corrupt
        self.probes = []
        self.fetches = []

    def probe(self, run_root, relative_path):
        self.probes.append((run_root, relative_path))
        return self.info

    def fetch(
        self,
        run_root,
        relative_path,
        expected_bytes,
        expected_sha256,
        destination_fd,
    ):
        self.fetches.append(
            (
                run_root,
                relative_path,
                expected_bytes,
                expected_sha256,
                destination_fd,
            )
        )
        os.write(
            destination_fd,
            self.payload + (b"corrupt" if self.corrupt else b""),
        )


class FetchVideoTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.manifest = self.bundle / FETCH.EXPECTED_MANIFEST_NAME
        self.receipt = self.bundle / "video-transfer-receipt.json"
        self.run_id = "vlsa-table1-contact-authority-population-20260718a"
        self.run_root = (
            "/mnt/data/quanth/experiments/vlsa-aegis-table1/" + self.run_id
        )
        self.case_id = "vlsa-t1-spatial-i-t0-e00"
        self.arm = "pi05_translational"
        self.payload = b"fake mp4 bytes"
        self.video_sha = hashlib.sha256(self.payload).hexdigest()
        self.rows = [
            self.make_row(index)
            for index in range(FETCH.EXPECTED_ROWS)
        ]
        self.write_bundle()

    def tearDown(self):
        self.temporary.cleanup()

    def make_row(self, index):
        ordinal = index // 2
        task_index = ordinal // 50
        episode = ordinal % 50
        case_id = (
            self.case_id
            if ordinal == 0
            else "vlsa-case-{:04d}".format(ordinal)
        )
        arm = FETCH.EXPECTED_ARMS[index % 2]
        mode = FETCH.ARM_MODES[arm]
        prefix = "tasks/task-{}/results/{}/{}/".format(
            task_index, mode, case_id
        )
        return {
            "schema_version": FETCH.ROW_SCHEMA,
            "case_ordinal": ordinal,
            "task_index": task_index,
            "task_level_group_id": "group-{:02d}".format(task_index),
            "case_id": case_id,
            "arm": arm,
            "mode": mode,
            "status": "complete",
            "result": {
                "path": prefix + "result.json",
                "bytes": 123,
                "sha256": "1" * 64,
                "result_payload_sha256": "2" * 64,
            },
            "video": {
                "path": prefix + "episode.mp4",
                "bytes": len(self.payload),
                "sha256": self.video_sha,
                "frames": episode + 1,
                "fps": 20,
                "complete_episode": True,
            },
            "publisher_v2_inventory_item": {
                "relative_result_path": prefix + "result.json",
            },
        }

    def write_bundle(self):
        manifest_payload = b"".join(
            canonical_json_bytes(row) + b"\n" for row in self.rows
        )
        self.manifest.write_bytes(manifest_payload)
        receipt = {
            "schema_version": FETCH.RECEIPT_SCHEMA,
            "status": "validated",
            "source": {
                "run_id": self.run_id,
                "run_root": self.run_root,
                "source_git_commit": "1" * 40,
                "publication_receipt": {
                    "path": self.run_root + "/population-publication.json",
                    "sha256": "3" * 64,
                    "receipt_payload_sha256": "4" * 64,
                },
            },
            "transfer_manifest": {
                "path": FETCH.EXPECTED_MANIFEST_NAME,
                "rows": len(self.rows),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "canonical_rows_sha256": hashlib.sha256(
                    canonical_json_bytes(self.rows)
                ).hexdigest(),
            },
        }
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(receipt)
        ).hexdigest()
        self.receipt.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def select(self):
        return FETCH.select_video(
            self.manifest, self.receipt, self.case_id, self.arm
        )

    def test_selects_and_fetches_exactly_one_video(self):
        selected = self.select()
        transport = FakeTransport(self.payload)
        output = self.root / "selected.mp4"
        observed = FETCH.fetch_selected_video(selected, output, transport)
        self.assertEqual(observed, output.absolute())
        self.assertEqual(output.read_bytes(), self.payload)
        self.assertEqual(len(transport.probes), 1)
        self.assertEqual(len(transport.fetches), 1)
        self.assertEqual(
            transport.probes[0],
            (self.run_root, self.rows[0]["video"]["path"]),
        )
        self.assertEqual(
            transport.fetches[0][0:4],
            (
                self.run_root,
                self.rows[0]["video"]["path"],
                len(self.payload),
                self.video_sha,
            ),
        )
        self.assertEqual(output.stat().st_mode & 0o777, 0o444)
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_missing_case_is_rejected_without_transport(self):
        with self.assertRaisesRegex(FETCH.FetchError, "absent"):
            FETCH.select_video(
                self.manifest, self.receipt, "missing-case", self.arm
            )

    def test_duplicate_case_arm_is_rejected(self):
        self.rows[1]["case_id"] = self.case_id
        self.rows[1]["arm"] = self.arm
        self.rows[1]["mode"] = "pi05"
        self.rows[1]["case_ordinal"] = 0
        self.rows[1]["task_index"] = 0
        self.rows[1]["task_level_group_id"] = "group-00"
        self.rows[1]["result"]["path"] = self.rows[0]["result"]["path"]
        self.rows[1]["video"]["path"] = self.rows[0]["video"]["path"]
        self.write_bundle()
        with self.assertRaisesRegex(FETCH.FetchError, "duplicate"):
            self.select()

    def test_unsupported_or_unknown_row_schema_is_rejected(self):
        self.rows[0]["schema_version"] = "vlsa_table1_video_transfer_row.v1"
        self.write_bundle()
        with self.assertRaisesRegex(FETCH.FetchError, "unsupported schema"):
            self.select()
        self.rows[0]["schema_version"] = FETCH.ROW_SCHEMA
        self.rows[0]["unexpected"] = True
        self.write_bundle()
        with self.assertRaisesRegex(FETCH.FetchError, "extra"):
            self.select()

    def test_unsafe_video_paths_are_rejected(self):
        for unsafe in (
            "/tmp/episode.mp4",
            "../episode.mp4",
            "tasks/task-0/results/pi05/other/episode.mp4",
            "tasks/task-0/results/pi05/{}/not-video.mp4".format(self.case_id),
        ):
            with self.subTest(unsafe=unsafe):
                self.rows[0]["video"]["path"] = unsafe
                self.write_bundle()
                with self.assertRaises(FETCH.FetchError):
                    self.select()
        self.rows[0] = self.make_row(0)
        self.write_bundle()

    def test_symlinked_manifest_and_receipt_are_rejected(self):
        manifest_link = self.root / "manifest-link.jsonl"
        receipt_link = self.root / "receipt-link.json"
        manifest_link.symlink_to(self.manifest)
        receipt_link.symlink_to(self.receipt)
        with self.assertRaisesRegex(FETCH.FetchError, "symlinked"):
            FETCH.select_video(
                manifest_link, self.receipt, self.case_id, self.arm
            )
        with self.assertRaisesRegex(FETCH.FetchError, "symlinked"):
            FETCH.select_video(
                self.manifest, receipt_link, self.case_id, self.arm
            )

    def test_manifest_and_receipt_hash_tampering_is_rejected(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaises(FETCH.FetchError):
            self.select()
        self.write_bundle()
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        receipt["source"]["run_root"] += "-changed"
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(FETCH.FetchError, "payload SHA-256"):
            self.select()

    def test_receipt_publication_and_run_root_binding_are_required(self):
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        receipt["source"]["publication_receipt"]["path"] = (
            "/mnt/data/quanth/experiments/vlsa-aegis-table1/other/publication.json"
        )
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_payload_sha256"
                }
            )
        ).hexdigest()
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(FETCH.FetchError, "not bound"):
            self.select()

    def test_remote_identity_and_symlink_are_rejected_before_fetch(self):
        selected = self.select()
        wrong = FakeTransport(
            self.payload,
            info=FETCH.RemoteFileInfo(
                bytes=len(self.payload),
                sha256="f" * 64,
            ),
        )
        with self.assertRaisesRegex(FETCH.FetchError, "manifest identity"):
            FETCH.fetch_selected_video(selected, self.root / "wrong.mp4", wrong)
        self.assertEqual(wrong.fetches, [])

        symlink = FakeTransport(
            self.payload,
            info=FETCH.RemoteFileInfo(
                bytes=len(self.payload),
                sha256=self.video_sha,
                symlink=True,
            ),
        )
        with self.assertRaisesRegex(FETCH.FetchError, "regular file"):
            FETCH.fetch_selected_video(
                selected, self.root / "symlink.mp4", symlink
            )
        self.assertEqual(symlink.fetches, [])

    def test_corrupt_transfer_is_not_published_and_temp_is_removed(self):
        selected = self.select()
        transport = FakeTransport(self.payload, corrupt=True)
        output = self.root / "corrupt.mp4"
        with self.assertRaisesRegex(FETCH.FetchError, "downloaded video"):
            FETCH.fetch_selected_video(selected, output, transport)
        self.assertFalse(os.path.lexists(str(output)))
        self.assertEqual(
            [path for path in self.root.iterdir() if path.name.endswith(".partial")],
            [],
        )

    def test_temp_path_swap_cannot_truncate_symlink_target(self):
        selected = self.select()
        victim = self.root / "victim.txt"
        victim.write_bytes(b"must remain unchanged")

        class SwapTransport(FakeTransport):
            def fetch(
                inner_self,
                run_root,
                relative_path,
                expected_bytes,
                expected_sha256,
                destination_fd,
            ):
                temporary = next(self.root.glob(".swapped.mp4.fetch-*.partial"))
                temporary.unlink()
                temporary.symlink_to(victim)
                os.write(destination_fd, inner_self.payload)

        output = self.root / "swapped.mp4"
        with self.assertRaisesRegex(
            FETCH.FetchError, "pathname no longer names its open file"
        ):
            FETCH.fetch_selected_video(
                selected, output, SwapTransport(self.payload)
            )
        self.assertEqual(victim.read_bytes(), b"must remain unchanged")
        self.assertFalse(os.path.lexists(str(output)))
        self.assertEqual(
            [path for path in self.root.iterdir() if path.name.endswith(".partial")],
            [],
        )

    def test_existing_or_symlink_output_is_never_replaced(self):
        selected = self.select()
        transport = FakeTransport(self.payload)
        existing = self.root / "existing.mp4"
        existing.write_bytes(b"keep")
        with self.assertRaisesRegex(FETCH.FetchError, "already exists"):
            FETCH.fetch_selected_video(selected, existing, transport)
        self.assertEqual(existing.read_bytes(), b"keep")
        self.assertEqual(transport.probes, [])
        target = self.root / "target.mp4"
        target.write_bytes(b"keep-target")
        link = self.root / "link.mp4"
        link.symlink_to(target)
        with self.assertRaisesRegex(FETCH.FetchError, "already exists"):
            FETCH.fetch_selected_video(selected, link, transport)
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_bytes(), b"keep-target")

    def test_output_suffix_and_parent_symlink_are_rejected(self):
        selected = self.select()
        transport = FakeTransport(self.payload)
        with self.assertRaisesRegex(FETCH.FetchError, ".mp4 suffix"):
            FETCH.fetch_selected_video(
                selected, self.root / "not-a-video.bin", transport
            )
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(FETCH.FetchError, "existing real directory"):
            FETCH.fetch_selected_video(
                selected, parent_link / "video.mp4", transport
            )

    def test_intermediate_output_parent_symlink_is_rejected(self):
        selected = self.select()
        transport = FakeTransport(self.payload)
        real_parent = self.root / "real-tree"
        nested = real_parent / "nested"
        nested.mkdir(parents=True)
        ancestor_link = self.root / "ancestor-link"
        ancestor_link.symlink_to(real_parent, target_is_directory=True)
        output = ancestor_link / "nested" / "video.mp4"
        self.assertFalse(output.parent.is_symlink())
        with self.assertRaisesRegex(FETCH.FetchError, "ancestry contains a symlink"):
            FETCH.fetch_selected_video(selected, output, transport)
        self.assertEqual(transport.probes, [])

    def test_scp_transport_rejects_unsafe_host_and_remote_path(self):
        with self.assertRaisesRegex(FETCH.FetchError, "safe configured alias"):
            FETCH.ScpTransport("vinuni;touch-pwned")
        transport = FETCH.ScpTransport("vinuni")
        with self.assertRaisesRegex(FETCH.FetchError, "unsafe"):
            with tempfile.TemporaryFile() as destination:
                transport.fetch(
                    "/tmp",
                    "file name.mp4",
                    1,
                    "1" * 64,
                    destination.fileno(),
                )


if __name__ == "__main__":
    unittest.main()
