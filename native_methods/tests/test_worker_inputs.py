"""Raw ancestry provenance rejects changed bytes or changed source manifests."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from native_methods.worker import lock_raw_source, raw_source_unchanged


class RawSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "Trace/mooncake/raw.jsonl"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b'{"prefix":"original"}\n')
        self.manifest = self.root / "Trace/sources.json"
        self.manifest.write_text(
            json.dumps(
                dict(
                    files=[
                        dict(
                            name="Mooncake",
                            file="mooncake/raw.jsonl",
                            sha256=sha256(self.raw.read_bytes()).hexdigest(),
                            commit="fixed-test-source",
                            download_url="https://example.invalid/raw",
                        )
                    ]
                )
            )
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_lock_records_actual_source_and_detects_mutation(self):
        record = lock_raw_source("mooncake", self.root)
        self.assertEqual(record["sha256"], sha256(self.raw.read_bytes()).hexdigest())
        self.assertTrue(raw_source_unchanged(record, self.root))
        self.raw.write_bytes(b'{"prefix":"changed"}\n')
        self.assertFalse(raw_source_unchanged(record, self.root))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            lock_raw_source("mooncake", self.root)

    def test_manifest_change_and_missing_file_are_not_unchanged(self):
        record = lock_raw_source("mooncake", self.root)
        self.manifest.write_text(self.manifest.read_text() + "\n")
        self.assertFalse(raw_source_unchanged(record, self.root))
        record = lock_raw_source("mooncake", self.root)
        self.raw.unlink()
        self.assertFalse(raw_source_unchanged(record, self.root))
