# Telemetry Ownership Tool

Generate **ownership summaries** for Firefox Telemetry:

- **Firefox Desktop** (default; always included)
- **Mobile apps** (optional): Fenix (Firefox Android), Focus Android, Firefox iOS, Focus iOS
- Outputs **CSV + Markdown** summary
- **Desktop mirrored metrics** (GIFFT) supported via `--mirrors {heuristic|strict|none}`
- **iOS ownership** improves via **fallbacks** and optional **CODEOWNERS** enrichment

---

## Quick Start

```bash
pip install -r requirements.txt

# Desktop only (downloaded from the Glean Dictionary)
python telemetry_ownership.py \
  --outdir ./outputs
```

This produces:

```
outputs/ownership_detail.csv
outputs/ownership_summary.csv
outputs/SUMMARY.md
```

## Include Mobile Apps

```bash
python telemetry_ownership.py \
  --outdir ./outputs \
  --apps fenix,focus_android,firefox_ios,focus_ios
```

## Local Overrides (optional)

If you have local index.json files, you can override any app by passing --inputs:

```bash
# Uses local files where available (others will be fetched)
# Expected local names:
#   firefox_desktop_index.json
#   fenix_index.json
#   focus_android_index.json
#   firefox_ios_index.json
#   focus_ios_index.json
python telemetry_ownership.py \
  --outdir ./outputs \
  --inputs ./inputs \
  --apps fenix,focus_android,firefox_ios,focus_ios
```

You can override the Desktop filename (default is firefox_desktop_index.json):

```bash
python telemetry_ownership.py \
  --outdir ./outputs \
  --desktop my_firefox_desktop_index.json
```

## Desktop Mirrored Metrics

Choose how Desktop mirrors are detected:

- `--mirrors heuristic` (default): searches the Desktop index for GIFFT cues (e.g., telemetry_mirror, “legacy telemetry”, “generated to correspond”, “gifft”, etc.).
- `--mirrors strict`: only counts mirrors if a literal telemetry_mirror field exists in the Desktop index object.
- `--mirrors none`: disables mirror counting.

Examples:

```bash
# Desktop only, heuristic mirroring (default)
python telemetry_ownership.py --outdir ./outputs
```

```bash
# Desktop + Mobile, strict mirroring
python telemetry_ownership.py \
  --outdir ./outputs \
  --apps fenix,firefox_ios \
  --mirrors strict
```

## iOS Ownership: Fallbacks & CODEOWNERS

When iOS metrics/pings lack tags and team emails:

- Default to <fx-ios-data-stewards@mozilla.com>
- If the metric/ping looks like Sync/App Services → <sync-team@mozilla.com>
- If it looks like Nimbus → <project-nimbus@mozilla.com>

You can refine iOS ownership using CODEOWNERS:

- Put the CODEOWNERS files in any path and point to them in config.yaml:

```yaml
    codeowners:
      firefox_ios: "./inputs/firefox_ios_CODEOWNERS"
      focus_ios: "./inputs/focus_ios_CODEOWNERS"
```

- Optionally map CODEOWNERS tokens (e.g. @org/team) to emails via owner_token_map.

## How Ownership is Determined (all apps)

Precedence per metric/ping:

- tags → treated as a team owner label

- A team-like list email in notification_emails (contains “team”, “-”, “telemetry”, “fx”, or “nimbus”)

- Otherwise the first email in notification_emails → individual

- For iOS only: if still unresolved, apply fallbacks (see above)

Core Glean pings/metrics (baseline, events, metrics, health, deletion-request, etc.) are excluded on mobile by default.

## Configuration

All rules & knobs live in config.yaml. A ready-to-use example is in this repo. You can customize exclusions, iOS fallbacks, CODEOWNERS paths, and token mappings.

## Outputs

- CSV

  - ownership_detail.csv — one row per metric/ping with owner, type, application, mirrored flag

  - ownership_summary.csv — grouped by owner, owner_type, application with counts

- Markdown

  - SUMMARY.md — human-readable overview, totals, and top owners per app

## Troubleshooting

- Mirrored counts are zero on Desktop
    Use --mirrors heuristic (default). If still zero, your Desktop index may not include GIFFT phrases. Try a newer export or enable strict if the index contains a literal telemetry_mirror key.

- iOS owners show as “Unknown”
    Ensure you’re on the latest script. It applies iOS fallbacks. Add CODEOWNERS + token mappings to improve grouping.

- Fetch failures
    The tool fetches from the Glean Dictionary by default. If your environment blocks outbound requests, provide local files via --inputs.