from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from telemetry_ownership import (
    DEFAULT_CONFIG,
    add_desktop_mirrors,
    choose_codeowners_owner,
    codeowners_match,
    ios_fallback_owner,
    is_core_metric,
    map_codeowners_tokens,
    metric_has_mirror_heuristic,
    owner_from_tags_emails,
    rows_from_app,
    source_url_to_ios_rel,
    write_markdown_summary,
)


class HelperFunctionTests(unittest.TestCase):
    def test_is_core_metric_matches_core_ping_and_prefix(self):
        core = {"baseline", "metrics"}
        prefixes = ["glean.", "nimbus."]
        self.assertTrue(is_core_metric("baseline.foo", core, prefixes))
        self.assertTrue(is_core_metric("glean.event", core, prefixes))
        self.assertFalse(is_core_metric("custom.metric", core, prefixes))

    def test_owner_from_tags_emails_precedence(self):
        tagged = {"tags": ["Data Team"]}
        self.assertEqual(owner_from_tags_emails(tagged, ["team"]), ("Data Team", "team"))

        team_email = {"notification_emails": ["awesome-team@mozilla.com"]}
        self.assertEqual(
            owner_from_tags_emails(team_email, ["team"]), ("awesome-team@mozilla.com", "team")
        )

        individual = {"notification_emails": ["owner@example.com"]}
        self.assertEqual(
            owner_from_tags_emails(individual, ["team"]), ("owner@example.com", "individual")
        )

    def test_ios_fallback_owner_variants(self):
        fallbacks = DEFAULT_CONFIG["fallbacks"]
        sync_item = {"origin": "sync service"}
        self.assertEqual(ios_fallback_owner(sync_item, fallbacks), (fallbacks["ios_sync"], "team"))

        nimbus_item = {"name": "NimbusFoo"}
        self.assertEqual(
            ios_fallback_owner(nimbus_item, fallbacks), (fallbacks["ios_nimbus"], "team")
        )

        general_item = {"source_url": "https://github.com/mozilla-mobile/some"}
        self.assertEqual(
            ios_fallback_owner(general_item, fallbacks), (fallbacks["ios_general"], "team")
        )

    def test_codeowners_helpers(self):
        rules = [
            ("*.swift", ["@old/team"]),
            ("Sources/*", ["@new/team", "owner@example.com"]),
        ]
        match = codeowners_match("Sources/View.swift", rules)
        mapped = map_codeowners_tokens(match, {"@new/team": "team@example.com"})
        owner = choose_codeowners_owner(mapped)
        self.assertEqual(owner, ("team@example.com", "team"))

    def test_source_url_to_ios_rel(self):
        repo, rel = source_url_to_ios_rel(
            "https://github.com/mozilla-mobile/firefox-ios/Sources/Foo.swift#L10"
        )
        self.assertEqual(repo, "firefox_ios")
        self.assertEqual(rel, "Sources/Foo.swift")
        repo, rel = source_url_to_ios_rel(
            "https://github.com/mozilla-mobile/focus-ios/App/Main.swift"
        )
        self.assertEqual(repo, "focus_ios")
        self.assertEqual(rel, "App/Main.swift")
        self.assertEqual(source_url_to_ios_rel("https://github.com/other/repo/file"), (None, None))

    def test_metric_has_mirror_heuristic(self):
        metric = {"name": "foo", "telemetry_mirror": True}
        self.assertTrue(metric_has_mirror_heuristic(metric, []))
        metric = {"name": "foo", "description": "Generated to correspond with legacy telemetry"}
        self.assertTrue(metric_has_mirror_heuristic(metric, []))

    def test_add_desktop_mirrors_modes(self):
        metrics = [
            {"name": "metric_a", "telemetry_mirror": True},
            {"name": "metric_b", "description": "legacy telemetry"},
        ]
        rows = [
            {"application": "firefox_desktop", "type": "metric", "name": "metric_a", "mirrored": 0},
            {"application": "firefox_desktop", "type": "metric", "name": "metric_b", "mirrored": 0},
        ]
        cfg = deepcopy(DEFAULT_CONFIG)
        heuristic = add_desktop_mirrors(deepcopy(rows), {"metrics": metrics}, "heuristic", cfg)
        self.assertEqual([r["mirrored"] for r in heuristic], [1, 1])
        strict = add_desktop_mirrors(deepcopy(rows), {"metrics": metrics}, "strict", cfg)
        self.assertEqual([r["mirrored"] for r in strict], [1, 0])

    def test_rows_from_app_applies_codeowners_and_fallbacks(self):
        cfg = deepcopy(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as tmpdir:
            codeowners_path = Path(tmpdir) / "CODEOWNERS"
            codeowners_path.write_text("Sources/* @foo/team\n", encoding="utf-8")
            cfg["codeowners"]["firefox_ios"] = str(codeowners_path)
            cfg["codeowners"]["owner_token_map"] = {"@foo/team": "team@example.com"}

            index_obj = {
                "metrics": [
                    {"name": "glean.internal", "tags": [], "notification_emails": []},
                    {
                        "name": "usable_metric",
                        "tags": [],
                        "notification_emails": [],
                        "source_url": "https://github.com/mozilla-mobile/firefox-ios/Sources/View.swift#L10",
                    },
                ],
                "pings": [
                    {"name": "baseline"},
                    {"name": "custom_ping", "tags": [], "notification_emails": []},
                ],
            }

            rows = rows_from_app(
                "firefox_ios",
                index_obj,
                cfg,
                exclude_core=True,
                ios_codeowners=True,
            )

        metric_rows = [r for r in rows if r["type"] == "metric"]
        ping_rows = [r for r in rows if r["type"] == "ping"]

        # glean.internal metric is excluded, so only one metric remains
        self.assertEqual(len(metric_rows), 1)
        self.assertEqual(metric_rows[0]["owner"], "team@example.com")
        self.assertEqual(metric_rows[0]["owner_type"], "team")

        # baseline ping excluded, fallback applies to remaining ping
        self.assertEqual(len(ping_rows), 1)
        self.assertEqual(
            ping_rows[0]["owner"],
            DEFAULT_CONFIG["fallbacks"]["ios_general"],
        )

    def test_write_markdown_summary_outputs_expected_sections(self):
        detail_df = pd.DataFrame(
            [
                {"application": "firefox_desktop", "type": "metric", "mirrored": 1},
                {"application": "firefox_desktop", "type": "ping", "mirrored": 0},
            ]
        )
        summary_df = pd.DataFrame(
            [
                {
                    "application": "firefox_desktop",
                    "owner": "team@example.com",
                    "owner_type": "team",
                    "glean_metric_count": 1,
                    "mirrored_metric_count": 1,
                    "owned_pings_count": 1,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            write_markdown_summary(summary_df, detail_df, outdir, title="Test Report")
            output = (outdir / "SUMMARY.md").read_text(encoding="utf-8")

        self.assertIn("# Test Report", output)
        self.assertIn("firefox_desktop", output)
        self.assertIn("team@example.com", output)


if __name__ == "__main__":
    unittest.main()
