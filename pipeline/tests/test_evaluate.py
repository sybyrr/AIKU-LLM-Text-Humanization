import unittest
from collections import Counter

from pipeline.evaluate import acute_collapse, stratified_sample, strip_dialogue_markers


class StratifiedSampleTest(unittest.TestCase):
    def test_preserves_generator_ratio_and_exact_size(self):
        rows = ([{"id": f"q{i}", "generator": "qwen"} for i in range(75)]
                + [{"id": f"e{i}", "generator": "exaone"} for i in range(25)])
        sample = stratified_sample(rows, 10, seed=42)
        self.assertEqual(len(sample), 10)
        self.assertEqual(Counter(row["generator"] for row in sample), {"qwen": 7, "exaone": 3})
        self.assertEqual(sample, stratified_sample(rows, 10, seed=42))

    def test_zero_means_full_split(self):
        rows = [{"id": str(i), "generator": "qwen"} for i in range(4)]
        self.assertEqual(stratified_sample(rows, 0, seed=42), rows)

    def test_rows_without_generator_are_seed_sampled(self):
        rows = [{"id": str(i)} for i in range(10)]
        sample = stratified_sample(rows, 3, seed=42)
        self.assertEqual(sample, stratified_sample(rows, 3, seed=42))
        self.assertNotEqual(sample, rows[:3])


class AcuteCollapseTest(unittest.TestCase):
    def test_detects_repeated_fragment_across_whitespace(self):
        self.assertTrue(acute_collapse(("반복 문장 " * 30)))

    def test_normal_text_does_not_collapse(self):
        self.assertFalse(acute_collapse("서로 다른 문장으로 구성된 짧고 정상적인 글입니다."))

    def test_dialogue_markers_are_removed_only_at_line_start(self):
        text = "A: 첫 발화\nB:: 둘째 발화\n본문 A: 표시는 유지"
        self.assertEqual(
            strip_dialogue_markers(text),
            "첫 발화\n둘째 발화\n본문 A: 표시는 유지",
        )


if __name__ == "__main__":
    unittest.main()
