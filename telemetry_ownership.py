#!/usr/bin/env python3
from __future__ import annotations

# Telemetry Ownership Tool (v3.3)
# Default behavior:
#   - Always include Firefox Desktop (fetch index.json from Glean Dictionary if no local file)
#   - Mobile apps optional via --apps (each fetched from Glean Dictionary if no local file)
#   - Local overrides via --inputs still work (e.g., ./inputs/fenix_index.json)
# Features:
#   - Desktop mirrored counts (heuristic|strict)
#   - iOS fallbacks + CODEOWNERS enrichment
#   - CSV + Markdown outputs
import argparse
from copy import deepcopy
import fnmatch
import json
from pathlib import Path
import sys

import pandas as pd


try:
    import yaml
except ImportError:  # pragma: no cover - missing only in constrained envs
    yaml = None  # type: ignore[assignment]

try:
    import requests
except ImportError:  # pragma: no cover - exercised via runtime checks
    requests = None  # type: ignore[assignment]

HAS_YAML = yaml is not None
HAS_REQUESTS = requests is not None


DEFAULT_CONFIG = {
    "exclusions": {
        "core_pings": [
            "baseline",
            "metrics",
            "events",
            "health",
            "deletion-request",
            "fog-validation",
            "glean_validation",
            "glean_internal",
        ],
        "core_metric_prefixes": [
            "glean.",
            "fog.",
            "glean_internal.",
            "nimbus.validation.",
        ],
    },
    "ownership_rules": {"team_email_heuristics": ["team", "-", "telemetry", "fx", "nimbus"]},
    "fallbacks": {
        "ios_general": "fx-ios-data-stewards@mozilla.com",
        "ios_sync": "sync-team@mozilla.com",
        "ios_nimbus": "project-nimbus@mozilla.com",
    },
    "mirrors": {
        "heuristic_phrases": [
            "legacy telemetry",
            "generated to correspond",
            "telemetry_mirror",
            "gifft",
        ]
    },
    "codeowners": {},
}

DEFAULT_FALLBACKS = DEFAULT_CONFIG["fallbacks"].copy()


def _deep_update(base: dict, overrides: dict | None) -> dict:
    if not overrides:
        return base
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


# -------------------- Config helpers --------------------


def load_config(path: Path) -> dict:
    if not HAS_YAML:
        raise RuntimeError("The 'pyyaml' package is required; run pip install -r requirements.txt.")
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path or not path.exists():
        return cfg
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            print(f"[warn] Config at {path} is not a mapping; using defaults.", file=sys.stderr)
            raw = {}
    cfg = _deep_update(cfg, raw)
    for section in ("exclusions", "ownership_rules", "fallbacks", "mirrors", "codeowners"):
        if not isinstance(cfg.get(section), dict):
            cfg[section] = deepcopy(DEFAULT_CONFIG[section])
    heuristics = cfg["mirrors"].get("heuristic_phrases")
    if not heuristics:
        cfg["mirrors"]["heuristic_phrases"] = deepcopy(
            DEFAULT_CONFIG["mirrors"]["heuristic_phrases"]
        )
    cfg["codeowners"].setdefault("owner_token_map", {})
    return cfg


# -------------------- Network fetching --------------------

# Default Glean Dictionary endpoints per app
APP_TO_INDEX_URL = {
    "firefox_desktop": "https://dictionary.telemetry.mozilla.org/data/firefox_desktop/index.json",
    "fenix": "https://dictionary.telemetry.mozilla.org/data/fenix/index.json",
    "focus_android": "https://dictionary.telemetry.mozilla.org/data/focus_android/index.json",
    "firefox_ios": "https://dictionary.telemetry.mozilla.org/data/firefox_ios/index.json",
    "focus_ios": "https://dictionary.telemetry.mozilla.org/data/focus_ios/index.json",
}


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    if not HAS_REQUESTS:
        raise RuntimeError(
            "The 'requests' package is required; run pip install -r requirements.txt."
        )
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_index_for_app(
    app_key: str, inputs_dir: Path | None, desktop_filename: str | None = None
) -> dict | None:
    """
    Load index.json for an app.
    Priority:
      1) Local file from inputs_dir:
         - Desktop: inputs_dir/<desktop_filename or firefox_desktop_index.json>
         - Mobile:  inputs_dir/<app_key>_index.json
      2) Fetch from Glean Dictionary URL
    """
    # Try local
    if inputs_dir:
        if app_key == "firefox_desktop":
            local_name = desktop_filename or "firefox_desktop_index.json"
            local_path = inputs_dir / local_name
        else:
            local_path = inputs_dir / f"{app_key}_index.json"

        if local_path.exists():
            try:
                with open(local_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[warn] Failed to read local {local_path}: {e}", file=sys.stderr)

    # Fallback: fetch from Glean Dictionary
    url = APP_TO_INDEX_URL.get(app_key)
    if not url:
        print(f"[warn] No URL known for app '{app_key}'", file=sys.stderr)
        return None
    try:
        return fetch_json(url)
    except Exception as e:
        print(f"[warn] Failed to fetch {app_key} index from {url}: {e}", file=sys.stderr)
        return None


# -------------------- Exclusions --------------------


def is_core_ping(name: str, core_pings: set) -> bool:
    return bool(name) and name.lower() in core_pings


def is_core_metric(name: str, core_pings: set, prefixes: list[str]) -> bool:
    if not name:
        return False
    low = name.lower()
    first = low.split(".")[0]
    if first in core_pings:
        return True
    return any(low.startswith(pfx) for pfx in prefixes)


# -------------------- Ownership rules --------------------


def owner_from_tags_emails(item: dict, team_email_heur: list[str]) -> tuple[str, str]:
    tags = item.get("tags") or []
    emails = item.get("notification_emails") or []
    if tags:
        return tags[0], "team"
    for e in emails:
        local = e.split("@")[0].lower()
        if any(key in local for key in team_email_heur):
            return e, "team"
    if emails:
        return emails[0], "individual"
    return "Unknown", "unknown"


def ios_fallback_owner(item: dict, fallbacks: dict) -> tuple[str, str]:
    src = (item.get("source_url") or "").lower()
    origin = (item.get("origin") or "").lower()
    name = (item.get("name") or "").lower()
    if "sync" in origin or "sync" in name or "application-services" in src:
        return fallbacks["ios_sync"], "team"
    if "nimbus" in origin or "nimbus" in name or "nimbus" in src:
        return fallbacks["ios_nimbus"], "team"
    return fallbacks["ios_general"], "team"


# -------------------- CODEOWNERS (iOS) --------------------


def parse_codeowners(path: Path) -> list[tuple[str, list[str]]]:
    rules = []
    if not path.exists():
        return rules
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rules.append((parts[0], parts[1:]))
    return rules


def codeowners_match(relpath: str, rules: list[tuple[str, list[str]]]) -> list[str] | None:
    matched = None
    for pattern, owners in rules:
        if fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch("/" + relpath, pattern):
            matched = owners  # last matching wins
    return matched


def map_codeowners_tokens(tokens: list[str], token_map: dict[str, str]) -> list[str]:
    return [token_map.get(t, t) for t in (tokens or [])]


def choose_codeowners_owner(mapped_tokens: list[str]) -> tuple[str, str]:
    # prefer emails, else first token as a team label
    for t in mapped_tokens:
        if "@" in t and "." in t:
            return t, "team"
    if mapped_tokens:
        return mapped_tokens[0], "team"
    return "Unknown", "unknown"


def source_url_to_ios_rel(src: str) -> tuple[str | None, str | None]:
    s = (src or "").lower()
    if "github.com/mozilla-mobile/firefox-ios" in s:
        try:
            rel = src.split("firefox-ios/")[1].split("#", 1)[0]
            return "firefox_ios", rel
        except Exception:
            return "firefox_ios", ""
    if "github.com/mozilla-mobile/focus-ios" in s:
        try:
            rel = src.split("focus-ios/")[1].split("#", 1)[0]
            return "focus_ios", rel
        except Exception:
            return "focus_ios", ""
    return None, None


# -------------------- Mirrored (Desktop) --------------------


def metric_has_mirror_heuristic(metric_obj: dict, phrases: list[str]) -> bool:
    # literal key presence
    if "telemetry_mirror" in metric_obj and metric_obj.get("telemetry_mirror"):
        return True
    # scan entire object as text
    try:
        blob = json.dumps(metric_obj, ensure_ascii=False).lower()
    except Exception:
        blob = " ".join(str(v) for v in metric_obj.values()).lower()
    base_phrases = {"telemetry_mirror", "legacy telemetry", "generated to correspond", "gifft"}
    targets = {p.lower() for p in (phrases or [])} | base_phrases
    return any(p in blob for p in targets)


def add_desktop_mirrors(rows: list[dict], index_obj: dict, mode: str, cfg: dict) -> list[dict]:
    if mode == "none":
        return rows
    metrics_by_name: dict[str, dict] = {m.get("name"): m for m in (index_obj.get("metrics") or [])}
    if mode == "heuristic":
        phrases = [p.lower() for p in (cfg.get("mirrors", {}).get("heuristic_phrases") or [])]
        if not phrases:
            phrases = ["legacy telemetry", "generated to correspond", "telemetry_mirror", "gifft"]
        for r in rows:
            if r["type"] != "metric":
                continue
            src = metrics_by_name.get(r["name"], {})
            if metric_has_mirror_heuristic(src, phrases):
                r["mirrored"] = 1
    elif mode == "strict":
        for r in rows:
            if r["type"] != "metric":
                continue
            src = metrics_by_name.get(r["name"], {})
            if "telemetry_mirror" in src and src.get("telemetry_mirror"):
                r["mirrored"] = 1
    return rows


# -------------------- App ingestion --------------------


def rows_from_app(
    app_key: str, index_obj: dict, cfg: dict, exclude_core=True, ios_codeowners: bool = False
) -> list[dict]:
    core_pings = {n.lower() for n in (cfg.get("exclusions", {}).get("core_pings") or []) if n}
    prefixes = [p.lower() for p in (cfg.get("exclusions", {}).get("core_metric_prefixes") or [])]
    team_email_heur = (cfg.get("ownership_rules", {}) or {}).get(
        "team_email_heuristics", ["team", "-", "telemetry", "fx", "nimbus"]
    )
    user_fallbacks = cfg.get("fallbacks") or {}
    fallbacks = {
        "ios_general": user_fallbacks.get("ios_general", DEFAULT_FALLBACKS["ios_general"]),
        "ios_sync": user_fallbacks.get("ios_sync", DEFAULT_FALLBACKS["ios_sync"]),
        "ios_nimbus": user_fallbacks.get("ios_nimbus", DEFAULT_FALLBACKS["ios_nimbus"]),
    }

    # CODEOWNERS (iOS only)
    co_cfg = cfg.get("codeowners", {})
    co_path = co_cfg.get(app_key) if ios_codeowners else None
    co_rules = parse_codeowners(Path(co_path)) if co_path else []
    token_map = co_cfg.get("owner_token_map", {})

    metrics = index_obj.get("metrics", []) or []
    pings = index_obj.get("pings", []) or []
    rows = []

    # metrics
    for m in metrics:
        name = m.get("name")
        if exclude_core and is_core_metric(name, core_pings, prefixes):
            continue
        owner, owner_type = owner_from_tags_emails(m, team_email_heur)
        if owner_type == "unknown" and app_key in ("firefox_ios", "focus_ios"):
            owner, owner_type = ios_fallback_owner(m, fallbacks)
        if ios_codeowners and co_rules:
            repo_key, rel = source_url_to_ios_rel(m.get("source_url") or "")
            if repo_key == app_key and rel:
                tokens = codeowners_match(rel, co_rules)
                if tokens:
                    mapped = map_codeowners_tokens(tokens, token_map)
                    owner, owner_type = choose_codeowners_owner(mapped)
        rows.append(
            {
                "application": app_key,
                "type": "metric",
                "name": name,
                "owner": owner,
                "owner_type": owner_type,
                "mirrored": 0,
                "origin": m.get("origin"),
                "source_url": m.get("source_url"),
            }
        )

    # pings
    for p in pings:
        name = p.get("name")
        if exclude_core and is_core_ping(name, core_pings):
            continue
        owner, owner_type = owner_from_tags_emails(p, team_email_heur)
        if owner_type == "unknown" and app_key in ("firefox_ios", "focus_ios"):
            owner, owner_type = ios_fallback_owner(p, fallbacks)
        if ios_codeowners and co_rules:
            repo_key, rel = source_url_to_ios_rel(p.get("source_url") or "")
            if repo_key == app_key and rel:
                tokens = codeowners_match(rel, co_rules)
                if tokens:
                    mapped = map_codeowners_tokens(tokens, token_map)
                    owner, owner_type = choose_codeowners_owner(mapped)
        rows.append(
            {
                "application": app_key,
                "type": "ping",
                "name": name,
                "owner": owner,
                "owner_type": owner_type,
                "mirrored": 0,
                "origin": p.get("origin"),
                "source_url": p.get("source_url"),
            }
        )

    return rows


# -------------------- Markdown summary --------------------


def write_markdown_summary(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    outdir: Path,
    title="Telemetry Ownership — Report",
):
    out = Path(outdir) / "SUMMARY.md"
    total_metrics = int((detail_df['type'] == "metric").sum())
    total_pings = int((detail_df['type'] == "ping").sum())
    total_mirrors = int(detail_df.loc[detail_df['type'] == "metric", "mirrored"].sum())
    lines = [
        f"# {title}\n",
        f"- Applications: **{summary_df['application'].nunique()}**",
        f"- Owners: **{summary_df[['owner', 'application']].drop_duplicates().shape[0]}**",
        f"- Metrics: **{total_metrics}**  |  Pings: **{total_pings}**  |  Mirrored (metrics): **{total_mirrors}**\n",
    ]
    for app in sorted(summary_df['application'].unique()):
        sub = summary_df[summary_df['application'] == app].copy()
        sub['total'] = sub['glean_metric_count'] + sub['owned_pings_count']
        top = sub.sort_values(
            ['total', 'glean_metric_count', 'owned_pings_count'], ascending=False
        ).head(12)
        lines.append(f"## {app}\n| owner | owner_type | metrics | mirrored | pings |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['owner']} | {r['owner_type']} | {int(r['glean_metric_count'])} | {int(r['mirrored_metric_count'])} | {int(r['owned_pings_count'])} |"
            )
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


# -------------------- CLI --------------------


def main():
    ap = argparse.ArgumentParser(
        description="Telemetry ownership summarizer (Desktop default via Glean Dictionary; mobile optional; local overrides via --inputs)"
    )
    ap.add_argument(
        "--inputs",
        default="",
        help="Optional folder with local *index.json overrides (e.g., fenix_index.json). If absent, fetch from Glean Dictionary.",
    )
    ap.add_argument("--outdir", required=True, help="Output folder for CSVs/Markdown")
    ap.add_argument(
        "--apps",
        default="",
        help="Comma-separated mobile apps (e.g. fenix,focus_android,firefox_ios,focus_ios)",
    )
    ap.add_argument(
        "--desktop",
        default="firefox_desktop_index.json",
        help="Desktop index filename to look for locally (default). If not found, we fetch from Glean Dictionary.",
    )
    ap.add_argument(
        "--mirrors",
        default="heuristic",
        choices=["none", "heuristic", "strict"],
        help="Mirror mode for Desktop (default=heuristic)",
    )
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml (optional)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    inputs_dir = Path(args.inputs) if args.inputs else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    app_keys = [a.strip() for a in args.apps.split(",") if a.strip()]

    all_rows: list[dict] = []

    # Always include Desktop (try local, else fetch)
    dindex = load_index_for_app("firefox_desktop", inputs_dir, desktop_filename=args.desktop)
    if dindex:
        drows = rows_from_app(
            "firefox_desktop", dindex, cfg, exclude_core=False, ios_codeowners=False
        )
        drows = add_desktop_mirrors(drows, dindex, args.mirrors, cfg)
        all_rows.extend(drows)
    else:
        print(
            "[warn] Could not load Firefox Desktop index.json (local or remote).", file=sys.stderr
        )

    # Optional mobile apps
    for app in app_keys:
        idx = load_index_for_app(app, inputs_dir)
        if not idx:
            print(f"[warn] Skipping app '{app}' (no local file and fetch failed).", file=sys.stderr)
            continue
        ios_co = app in ("firefox_ios", "focus_ios")
        rows = rows_from_app(app, idx, cfg, exclude_core=True, ios_codeowners=ios_co)
        all_rows.extend(rows)

    detail = pd.DataFrame(all_rows)
    if detail.empty:
        print("[error] no data produced — check connectivity and inputs.", file=sys.stderr)
        sys.exit(2)

    summary = (
        detail.groupby(["owner", "owner_type", "application"], dropna=False)
        .agg(
            glean_metric_count=("type", lambda s: int((s == "metric").sum())),
            mirrored_metric_count=("mirrored", "sum"),
            owned_pings_count=("type", lambda s: int((s == "ping").sum())),
        )
        .reset_index()
        .sort_values(["application", "owner_type", "owner"])
    )

    # Outputs
    detail_path = outdir / "ownership_detail.csv"
    summary_path = outdir / "ownership_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown_summary(
        summary, detail, outdir, title="Telemetry Ownership — Desktop (default) + Mobile (optional)"
    )

    # Console summary
    print(f"Wrote:\n  {summary_path}\n  {detail_path}\n  {outdir / 'SUMMARY.md'}")
    print(
        f"[apps]={summary['application'].nunique()} "
        f"[owners]={summary[['owner', 'application']].drop_duplicates().shape[0]} "
        f"[rows]={len(detail)}"
    )
    d_mir = int(
        detail[(detail['application'] == "firefox_desktop") & (detail['type'] == "metric")][
            "mirrored"
        ].sum()
    )
    print(f"[desktop mirrored metrics]={d_mir} (mode={args.mirrors})")


if __name__ == "__main__":
    main()
