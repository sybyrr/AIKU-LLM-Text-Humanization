import json
import tempfile
import unittest
from pathlib import Path

from pipeline.run import (
    ConfigError,
    build_commands,
    generation_complete,
    load_config,
    resolve_paths,
    stage_slice,
)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.example = Path(__file__).resolve().parents[1] / "config.example.json"

    def test_example_config_loads_and_resolves(self):
        config = load_config(self.example)
        paths = resolve_paths(config, self.example, None)
        self.assertEqual(config["domain"], "example")
        self.assertTrue(paths.human_pool.is_absolute())
        self.assertEqual(set(paths.generator_outputs), {"qwen", "exaone"})

        commands = build_commands(config, paths, "python")
        gate = commands["gate"][0][1]
        sft = commands["sft"][0][1]
        hard = commands["hard-negative"][0][1]
        dpo = commands["dpo"][0][1]
        evaluate = commands["evaluate"][0][1]
        self.assertTrue(gate[1].endswith("scripts/domain_gate.py"))
        self.assertIn("--resume", sft)
        self.assertEqual(hard[hard.index("--no-repeat-ngram-size") + 1], "3")
        self.assertIn("--dpop", dpo)
        self.assertIn("--length-norm", dpo)
        self.assertEqual(evaluate[evaluate.index("--num-beams") + 1], "4")
        self.assertEqual(evaluate[evaluate.index("--output-max-length") + 1], "1024")

    def test_invalid_stage_order_is_rejected(self):
        with self.assertRaises(ConfigError):
            stage_slice("dpo", "gate")


class ResumeTest(unittest.TestCase):
    def test_generation_requires_every_successful_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            output = root / "output.jsonl"
            prompts.write_text(
                "\n".join(json.dumps({"doc_id": str(i), "cond": "P3b"}) for i in range(3)) + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps({"doc_id": "0", "cond": "P3b", "text": "ok"}) + "\n"
                + json.dumps({"doc_id": "1", "cond": "P3b", "error": "failed"}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(generation_complete(prompts, output))
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"doc_id": "2", "cond": "P3b", "text": "ok"}) + "\n")
            self.assertFalse(generation_complete(prompts, output))
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"doc_id": "1", "cond": "P3b", "text": "ok"}) + "\n")
            self.assertTrue(generation_complete(prompts, output))


if __name__ == "__main__":
    unittest.main()
