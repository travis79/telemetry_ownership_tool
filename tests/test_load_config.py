from pathlib import Path
import tempfile
import unittest

from telemetry_ownership import DEFAULT_CONFIG, HAS_YAML, load_config


@unittest.skipUnless(HAS_YAML, "pyyaml not installed")
class LoadConfigTests(unittest.TestCase):
    def test_missing_config_returns_defaults(self):
        cfg = load_config(Path("does_not_exist.yaml"))
        self.assertEqual(cfg["fallbacks"], DEFAULT_CONFIG["fallbacks"])
        self.assertEqual(
            cfg["mirrors"]["heuristic_phrases"],
            DEFAULT_CONFIG["mirrors"]["heuristic_phrases"],
        )
        self.assertEqual(
            cfg["codeowners"].get("owner_token_map", {}),
            DEFAULT_CONFIG["codeowners"].get("owner_token_map", {}),
        )

    def test_partial_overrides_merge_defaults(self):
        config_text = """
fallbacks:
  ios_general: custom-owner@example.com
mirrors: null
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(config_text, encoding="utf-8")
            cfg = load_config(cfg_path)

        self.assertEqual(cfg["fallbacks"]["ios_general"], "custom-owner@example.com")
        self.assertEqual(cfg["fallbacks"]["ios_sync"], DEFAULT_CONFIG["fallbacks"]["ios_sync"])
        self.assertEqual(cfg["fallbacks"]["ios_nimbus"], DEFAULT_CONFIG["fallbacks"]["ios_nimbus"])
        self.assertEqual(
            cfg["mirrors"]["heuristic_phrases"],
            DEFAULT_CONFIG["mirrors"]["heuristic_phrases"],
        )


if __name__ == "__main__":
    unittest.main()
