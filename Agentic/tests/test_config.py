import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_env, runtime_defaults
from agent import parser


class ConfigurationTests(unittest.TestCase):
    def test_commented_model_switch_and_automatic_folder_name(self):
        with tempfile.TemporaryDirectory() as directory, patch("config.ROOT", Path(directory)):
            for selected, inactive in [("model-a", "model-b"), ("model-b", "model-a")]:
                (Path(directory) / ".env").write_text(
                    f"# LLM_MODEL={inactive}\nLLM_MODEL={selected}\nMODEL_NAME=\n", encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True):
                    load_env()
                    defaults = runtime_defaults()
                    self.assertEqual(defaults["model"], selected)
                    self.assertEqual(defaults["model_name"], "")
                    self.assertEqual(defaults["observation"], "dom")

    def test_active_endpoint_and_shared_key(self):
        values = {"AGENT_BRAIN_MODE": "llm_only", "LLM_ENABLED": "1", "VLM_ENABLED": "0",
                  "LLM_MODEL": "text", "VLM_MODEL": "vision", "LLM_BASE_URL": "http://localhost:1234/v1",
                  "VLM_BASE_URL": "http://localhost:8000/v1", "OPENAI_API_KEY": "fixture-key"}
        with patch.dict(os.environ, values, clear=True):
            defaults = runtime_defaults()
            self.assertEqual(defaults["model"], "text")
            self.assertEqual(defaults["observation"], "dom")
            self.assertEqual(defaults["api_key"], "fixture-key")
            os.environ.update(AGENT_BRAIN_MODE="vlm_only", VLM_ENABLED="1")
            defaults = runtime_defaults()
            self.assertEqual(defaults["model"], "vision")
            self.assertEqual(defaults["observation"], "vision")
            self.assertEqual(defaults["lm_base_url"], "http://localhost:8000/v1")

    def test_disabled_endpoint_and_hybrid_are_rejected(self):
        for values in ({"AGENT_BRAIN_MODE": "hybrid"},
                       {"AGENT_BRAIN_MODE": "vlm_only", "VLM_ENABLED": "0"}):
            with patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                runtime_defaults()

    def test_survey_selection_and_cli_overrides(self):
        values = {"LLM_MODEL": "model-a", "SURVEY_VERSION": "v0", "SURVEY_OCCURRENCE": "o1",
                  "SURVEY_ORDERS": "order02 order03", "SURVEY_THEMES": "work finance", "SURVEY_REPEATS": "5",
                  "SURVEY_TARGET": "http://localhost:3001"}
        with patch.dict(os.environ, values, clear=True), patch("agent.load_env"):
            args = parser().parse_args(["run"])
            self.assertEqual(args.model, "model-a")
            self.assertEqual(args.orders, ["order02", "order03"])
            self.assertEqual(args.themes, ["work", "finance"])
            self.assertEqual(args.repeats, 5)
            self.assertEqual(args.suite_version, "v0")
            args = parser().parse_args(["run", "--model", "override", "--orders", "order01"])
            self.assertEqual(args.model, "override")
            self.assertEqual(args.orders, ["order01"])


if __name__ == "__main__":
    unittest.main()
