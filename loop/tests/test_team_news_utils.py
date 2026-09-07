import json
from pathlib import Path
import sys
import tempfile
import unittest


LOOP_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = LOOP_DIR / "scripts"
sys.path.insert(0, str(LOOP_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from loop_lib import config as C
from loop_lib import data as D
from analyze_team_news_copykiller import exact_two_sided_binomial, summarize_scores


class PairedDataTests(unittest.TestCase):
    def test_external_pair_schema_and_per_split_limit(self):
        rows = []
        for split in D.SPLITS:
            for index in range(2):
                rows.append(
                    {
                        "id": f"{split}-{index}",
                        "split": split,
                        "human_text": f"human {index}",
                        "ai_text": f"ai {index}",
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded = D._load_paired_jsonl(path, limit=1)

        self.assertEqual([row["split"] for row in loaded], list(D.SPLITS))
        self.assertEqual([row["doc_id"] for row in loaded], ["train-0", "dev-0", "test-0"])

    def test_external_pair_rejects_duplicate_ids(self):
        duplicate = {
            "doc_id": "same",
            "split": "train",
            "human_text": "human",
            "ai_text": "ai",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "중복 doc_id"):
                D._load_paired_jsonl(path)


class ConfigTests(unittest.TestCase):
    def test_initial_model_paths_override_default_stage_paths(self):
        cfg = C.NS(
            {
                "paths": {
                    "runs": "loop/runs/example",
                    "initial_generator": "private/g1",
                    "initial_detector": "private/d1/model",
                },
                "arm": {"name": "example"},
            }
        )
        self.assertEqual(C.generator_in(cfg, 1), C.REPO_ROOT / "private/g1")
        self.assertEqual(C.detector_in(cfg, 1), C.REPO_ROOT / "private/d1/model")


class CopyKillerStatisticsTests(unittest.TestCase):
    def test_summary_uses_inclusive_ai_threshold(self):
        import numpy as np

        summary = summarize_scores(np.array([0, 49, 50, 100]), tau=50)
        self.assertEqual(summary["detected_ai_count"], 2)
        self.assertEqual(summary["asr_count"], 2)

    def test_exact_binomial_is_two_sided(self):
        self.assertEqual(exact_two_sided_binomial(0, 4), 0.125)
        self.assertEqual(exact_two_sided_binomial(2, 4), 1.0)


if __name__ == "__main__":
    unittest.main()
