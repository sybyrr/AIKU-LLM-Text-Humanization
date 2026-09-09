import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

from scripts.build_humanizer_skill_baseline import generator_engine
from scripts.summarize_zero_shot_matrix import (
    BASELINE_METHODS,
    DETECTORS,
    render_baseline,
    render_cross,
)
from scripts.zero_shot_matrix import prepare


class PromptBaselineTest(unittest.TestCase):
    def test_routes_to_original_generator_family(self):
        self.assertEqual(generator_engine("Qwen3-8B"), "qwen")
        self.assertEqual(generator_engine("EXAONE-3.5-7.8B"), "exaone")
        with self.assertRaises(ValueError):
            generator_engine("unknown")


class ZeroShotMatrixTest(unittest.TestCase):
    def test_prepare_deduplicates_and_balances_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = [
                {"text": "첫 번째 문서입니다."},
                {"text": "두 번째 문서입니다."},
                {"text": "첫 번째 문서입니다."},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                prepare([source], root / "work", shards=2)
            manifest = json.loads(
                (root / "work" / "inputs" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["unique_texts"], 2)
            self.assertEqual(manifest["shard_counts"], [1, 1])

    def test_result_html_has_no_document_fields(self):
        metric = {
            "n": 2, "expected": 2, "coverage_pct": 100.0,
            "asr": 50.0, "tpr": 50.0, "mean_score": 0.25,
        }
        methods = {
            detector: {
                method: dict(metric) for method in ("human", "x_ai", "SFT", "DPO")
            }
            for detector in DETECTORS
        }
        cross = {"smoke_to_smoke": {"methods": methods}}
        baseline = {
            "smoke": {
                "methods": {
                    detector: {method: dict(metric) for method in BASELINE_METHODS}
                    for detector in DETECTORS
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            cross_path = Path(temporary) / "cross.html"
            baseline_path = Path(temporary) / "baseline.html"
            render_cross(cross, ("smoke",), cross_path)
            render_baseline(baseline, ("smoke",), baseline_path)
            for path in (cross_path, baseline_path):
                raw = path.read_text(encoding="utf-8")
                parser = HTMLParser()
                parser.feed(raw)
                self.assertNotIn("doc_id", raw)
                self.assertNotIn('"text"', raw)


if __name__ == "__main__":
    unittest.main()
