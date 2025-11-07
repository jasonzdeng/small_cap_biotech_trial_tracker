# Biotech Catalyst Tracker

A lean Python workflow that narrows in on upcoming Phase 2/3 small-cap biotech catalysts using the curated list in `small_cap_biotech_phase2_3_readouts_2025_2026.json`. Each run:

- Pulls ClinicalTrials.gov data for the next 90 days.
- Matches trials back to the tickers in the JSON list.
- Uses the Perplexity API to confirm announcement timing and surface price-moving coverage.
- Writes a timestamped Markdown report under `output/` with concise verification, news, and key takeaways for each catalyst.

## Prerequisites

- Python 3.10+
- Perplexity API access and key (https://docs.perplexity.ai)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # optional convenience
$env:PERPLEXITY_API_KEY = "ptx-..."
```

Optional environment tweaks:

- `PERPLEXITY_MODEL` – override the default `sonar-pro` model.
- `LOOKAHEAD_DAYS` – change the 90-day window.
- `OPENAI_API_KEY` / `OPENAI_MODEL` – enable OpenAI edits and the closing **AI Quality Review** (defaults to `gpt-4o-mini`).
- `INSTRUCTIONS_PROMPT` – optional path to `instructions_prompt.md` if you plan to regenerate the universe via Claude.

## Usage

```powershell
python biotech_ai_watchlist.py
```

The script prints the generated filename, e.g. `output\biotech_readout_report_20251107_064321.md`. Open the Markdown in your editor or GitHub to review catalysts, announcement windows, and cited coverage.

## File Map

| File | Description |
| --- | --- |
| `biotech_ai_watchlist.py` | Main script that assembles the report. |
| `small_cap_biotech_phase2_3_readouts_2025_2026.json` | Company/ticker universe generated with Claude (`claude.ai`) using `instructions_prompt.md` and `search_method.md`. |
| `search_method.md` *(optional)* | Describes the research approach used when compiling the JSON. |
| `instructions_prompt.md` *(optional)* | Prompt text fed to Claude to produce/refresh the JSON list. |
| `output/` | Timestamped Markdown reports (plus optional AI review). |
| `.env.example` | Example of the minimum environment variables. |
| `requirements.txt` | Dependency list (`openai`, `requests`). |

## Customisation Tips

- **Adjust horizon**: update `LOOKAHEAD_DAYS` inside the script or via env var.
- **Control Perplexity model**: export `PERPLEXITY_MODEL` before running.
- **Swap data set**: replace the JSON file with your own list (same keys: `company`, `ticker`). The current JSON was produced by Claude following `search_method.md` and `instructions_prompt.md`.
- **Refresh the universe**: tweak the prompt files and rerun your Claude workflow to emit an updated JSON.

## Disclaimer

This report is for research purposes only. It highlights upcoming biotech catalysts and related commentary but is not investment advice. Always validate timelines and risks independently before trading.
