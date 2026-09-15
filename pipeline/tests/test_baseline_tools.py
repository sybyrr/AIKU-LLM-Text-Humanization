import unittest

from scripts.build_humanizer_skill_baseline import generator_engine


class PromptBaselineTest(unittest.TestCase):
    def test_routes_to_original_generator_family(self):
        self.assertEqual(generator_engine("Qwen3-8B"), "qwen")
        self.assertEqual(generator_engine("EXAONE-3.5-7.8B"), "exaone")
        with self.assertRaises(ValueError):
            generator_engine("unknown")


if __name__ == "__main__":
    unittest.main()
