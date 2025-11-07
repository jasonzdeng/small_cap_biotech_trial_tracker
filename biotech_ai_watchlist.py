#!/usr/bin/env python3
"""Biotech catalyst tracker that enriches matches with Perplexity research."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DATA_FILE = Path(__file__).with_name("small_cap_biotech_phase2_3_readouts_2025_2026.json")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "company_scan_cache.json"


def _resolve_lookahead_days(default: int = 90) -> int:
    raw = os.getenv("LOOKAHEAD_DAYS")
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


LOOKAHEAD_DAYS = _resolve_lookahead_days()
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"
PERPLEXITY_FALLBACK_MODELS = ["sonar-medium", "sonar-small", "sonar"]

GENERIC_TOKENS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "ltd",
    "limited",
    "plc",
    "llc",
    "lp",
    "sa",
    "ag",
    "nv",
    "holdings",
    "holding",
    "company",
    "co",
    "group",
    "biosciences",
    "bioscience",
    "biopharma",
    "biopharmaceutical",
    "biopharmaceuticals",
    "biotech",
    "bio",
    "therapeutics",
    "therapeutic",
    "pharma",
    "pharmaceutical",
    "pharmaceuticals",
    "sciences",
    "science",
    "medical",
    "medicine",
    "medicines",
    "technologies",
    "technology",
}

TOKEN_REPLACEMENTS = {
    "pharmaceutical": "pharma",
    "pharmaceuticals": "pharma",
    "biopharmaceutical": "biopharma",
    "biopharmaceuticals": "biopharma",
    "biosciences": "bioscience",
    "therapeutic": "therapeutics",
}


@dataclass
class Company:
    name: str
    ticker: str


@dataclass
class Trial:
    company: Company
    sponsor: str
    nct_id: str
    title: str
    condition: str
    phase: str
    url: str
    date: datetime


def load_companies(data_file: Path) -> List[Company]:
    with data_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [Company(item["company"], item["ticker"]) for item in payload]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def extract_tokens(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens: Set[str] = set()
    for token in raw_tokens:
        token = TOKEN_REPLACEMENTS.get(token, token)
        if token in GENERIC_TOKENS:
            continue
        if len(token) <= 2:
            continue
        tokens.add(token)
    return tokens


def parse_date(struct: Optional[Dict[str, str]]) -> Optional[datetime]:
    if not struct:
        return None
    value = struct.get("date")
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_iso_utc_date(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def extract_json_object(payload: str) -> Optional[Any]:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def load_company_scan_cache() -> Optional[Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return None
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def save_company_scan_cache(lookahead_days: int, entries: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "lookahead_days": lookahead_days,
        "entries": entries,
    }
    CACHE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prompt_use_cache(
    cached_window: int,
    current_window: int,
    cache_freshness: Optional[str] = None,
) -> bool:
    if not sys.stdin.isatty():
        return False
    freshness_note = f" Cache last updated: {cache_freshness}." if cache_freshness else ""
    prompt = (
        f"Cached supplemental scan covers {cached_window} days.{freshness_note} Reuse for current "
        f"{current_window}-day run? (Note: a fresh scan may surface newer catalysts.) [y/N]: "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"", "n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def fetch_trials(days_ahead: int) -> List[Dict]:
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    now_utc = datetime.now(UTC)
    start = now_utc.strftime("%Y-%m-%d")
    end = (now_utc + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    query = (
        f"(AREA[PrimaryCompletionDate]RANGE[{start},{end}] OR "
        f"AREA[CompletionDate]RANGE[{start},{end}]) "
        "AND AREA[StudyType]Interventional AND AREA[Phase](Phase 2 OR Phase 3)"
    )

    studies: List[Dict] = []
    page_token: Optional[str] = None

    while True:
        params = {
            "filter.advanced": query,
            "fields": (
                "NCTId,BriefTitle,Phase,Condition,CompletionDate,PrimaryCompletionDate," \
                "LeadSponsorName,LeadSponsorClass"
            ),
            "pageSize": 200,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return studies


def match_trials_to_companies(companies: Sequence[Company], studies: Sequence[Dict]) -> List[Trial]:
    company_profiles: List[Tuple[Company, str, Set[str]]] = []
    for company in companies:
        normalized_name = normalize(company.name)
        tokens = extract_tokens(company.name)
        if not tokens:
            tokens = {normalized_name}
        company_profiles.append((company, normalized_name, tokens))

    trials: List[Trial] = []

    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        conditions_module = protocol.get("conditionsModule", {})

        sponsor_name = sponsor_module.get("leadSponsor", {}).get("name")
        if not sponsor_name:
            continue

        sponsor_norm = normalize(sponsor_name)
        sponsor_tokens = extract_tokens(sponsor_name)
        matched_company: Optional[Company] = None

        for company, normalized_name, tokens in company_profiles:
            if normalized_name and (normalized_name in sponsor_norm or sponsor_norm in normalized_name):
                matched_company = company
                break

        if not matched_company and sponsor_tokens:
            best_company: Optional[Company] = None
            best_score = 0.0
            best_overlap = 0
            for company, normalized_name, tokens in company_profiles:
                overlap = tokens & sponsor_tokens
                if not overlap:
                    continue
                score_company = len(overlap) / max(len(tokens), 1)
                score_sponsor = len(overlap) / max(len(sponsor_tokens), 1)
                score = max(score_company, score_sponsor)
                if score > best_score:
                    best_score = score
                    best_company = company
                    best_overlap = len(overlap)
            if best_company and (best_score >= 0.6 or best_overlap >= 2):
                matched_company = best_company

        if not matched_company:
            continue
        completion_dt = parse_date(status.get("primaryCompletionDateStruct"))
        if not completion_dt:
            completion_dt = parse_date(status.get("completionDateStruct"))
        if not completion_dt:
            continue

        trial = Trial(
            company=matched_company,
            sponsor=sponsor_name,
            nct_id=identification.get("nctId", "N/A"),
            title=identification.get("briefTitle", "N/A"),
            condition=", ".join(conditions_module.get("conditions", [])[:2]) or "N/A",
            phase=" / ".join(design.get("phases", [])) or "N/A",
            url=f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}",
            date=completion_dt,
        )
        trials.append(trial)

    unique: Dict[Tuple[str, str], Trial] = {}
    for trial in trials:
        key = (trial.company.ticker, trial.nct_id)
        if key not in unique or trial.date < unique[key].date:
            unique[key] = trial

    return sorted(unique.values(), key=lambda t: t.date)


def filter_upcoming(trials: Sequence[Trial], window_days: int) -> List[Trial]:
    now = datetime.now(UTC)
    upper = now + timedelta(days=window_days)
    eligible = [trial for trial in trials if now <= trial.date <= upper]
    return sorted(eligible, key=lambda t: t.date)


def perplexity_research(api_key: Optional[str], trial: Trial) -> str:
    if not api_key:
        return "Perplexity API key not provided; skipping enriched research."

    endpoint = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    user_prompt = (
        f"Today is {today}. Verify the {trial.phase} catalyst for {trial.company.name} ({trial.company.ticker}) titled "
        f"'{trial.title}' targeting {trial.condition}. "
        f"Confirm when management or regulators have indicated topline data or announcement will occur near {trial.date.strftime('%Y-%m-%d')}. "
        "Highlight public communications (press releases, earnings calls, SEC filings, reputable biotech media) that mention timing or could move the stock. "
        "If timing is unclear, state the best estimate and why."
    )

    candidate_models = []
    env_model = os.getenv("PERPLEXITY_MODEL")
    if env_model:
        candidate_models.append(env_model)
    candidate_models.append(DEFAULT_PERPLEXITY_MODEL)
    candidate_models.extend(PERPLEXITY_FALLBACK_MODELS)

    seen_models = set()
    errors: List[str] = []

    for model in candidate_models:
        if model in seen_models:
            continue
        seen_models.add(model)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You verify biotech catalyst timing and surface price-moving news. "
                        "Respond in tight markdown (≤3 bullets per section, ≤18 words each). Format:\n\n"
                        "### Verification\n- [concise fact with citation]\n\n"
                        "### News Coverage\n- [Outlet](URL): headline or takeaway\n\n"
                        "### Key Takeaways\n- Expected announcement window (or 'unknown')\n- Risk/uncertainty\n- Actionable watchpoint\n\n"
                        "Always cite with '- [Outlet](URL)'."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 700,
            "return_citations": True,
            "return_images": False,
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
            if response.status_code == 401:
                return "Perplexity authentication failed; check PERPLEXITY_API_KEY."

            if response.status_code >= 400:
                detail = response.text[:400]
                errors.append(f"model '{model}' -> {detail}")
                error_type = ""
                try:
                    error_type = response.json().get("error", {}).get("type", "")
                except ValueError:
                    error_type = ""

                if error_type == "invalid_model":
                    continue
                return f"Perplexity request failed ({response.status_code}): {detail}"

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                return content.strip()
            return "Perplexity returned no content for this trial."
        except requests.RequestException as exc:
            errors.append(f"model '{model}' -> {exc}")
            continue

    if errors:
        return "Perplexity request error: " + "; ".join(errors)
    return "Perplexity request error: Unable to reach service."


def extract_confirmed_window(research: str) -> Optional[str]:
    for raw_line in research.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.lstrip("-*").strip()
        if normalized.lower().startswith("expected announcement window"):
            parts = normalized.split(":", 1)
            if len(parts) == 2:
                window = parts[1].strip()
                if window and window.lower() != "unknown":
                    return window
    return None


def generate_price_impact_comment(
    api_key: Optional[str],
    trial: Trial,
    research: str,
    confirmed_window: Optional[str],
) -> str:
    if not confirmed_window:
        return "No confirmed announcement window; impact commentary deferred."
    if not api_key:
        return "OpenAI API key not provided; cannot assess likely stock impact."

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an equity analyst at a biotech-focused trading desk."
        " Review the supplied research summary, which already synthesized recent news searches."
        " If the announcement timing is confirmed, provide a concise 1-2 sentence view of the likely near-term stock price impact."
        " Anchor your view in the cited news catalysts (mention sources inline) and reference the confirmed window."
        " Be specific about direction (upside/downside/muted) and catalyst sensitivity."
        " Do not add disclaimers or advice; keep it purely informational."
    )

    user_prompt = (
        f"Company: {trial.company.name} ({trial.company.ticker})\n"
        f"Trial: {trial.title}\n"
        f"Confirmed announcement window: {confirmed_window}\n"
        "\nResearch summary (with citations):\n"
        f"{research}"
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.25,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else "Impact assessment unavailable from OpenAI."
    except Exception as exc:
        return f"OpenAI impact assessment failed: {exc}"


def generate_executive_summary(
    entries: Sequence[Tuple[Trial, str, str, Optional[str]]]
) -> Optional[str]:
    if not entries:
        return None

    header = (
        "| Name | Ticker | Trial Completion Date | Announcement Date | Trial Type | Trial Name | Therapeutic Area |"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    rows: List[str] = [header, separator]

    for trial, _research, _impact_comment, confirmed_window in entries:
        completion_date = trial.date.strftime("%Y-%m-%d")
        announcement_date = confirmed_window or "Unconfirmed"
        trial_type = trial.phase or "N/A"
        trial_name = trial.title or "N/A"
        therapeutic_area = trial.condition or "N/A"

        def sanitize(value: str) -> str:
            return value.replace("|", "/") if value else "N/A"

        row = (
            f"| {sanitize(trial.company.name)} | {sanitize(trial.company.ticker)} | {completion_date} | "
            f"{sanitize(announcement_date)} | {sanitize(trial_type)} | {sanitize(trial_name)} | "
            f"{sanitize(therapeutic_area)} |"
        )
        rows.append(row)

    return "\n".join(rows)


def build_markdown(
    entries: Sequence[Tuple[Trial, str, str, Optional[str]]],
    executive_summary: Optional[str],
) -> str:
    lines: List[str] = []
    lines.append("# Biotech Catalyst Research Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Lookahead Window: {LOOKAHEAD_DAYS} days")
    lines.append(f"Opportunities Covered: {len(entries)}")
    lines.append("")
    if executive_summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(executive_summary.strip())
        lines.append("")

    if not entries:
        lines.append(
            f"No qualifying trial readouts within the next {LOOKAHEAD_DAYS} days for the provided company list."
        )
        return "\n".join(lines)

    for trial, research, impact_comment, confirmed_window in entries:
        days_out = (trial.date - datetime.now(UTC)).days
        lines.append(f"## {trial.company.name} ({trial.company.ticker})")
        lines.append("")
        lines.append("**Catalyst Overview**")
        lines.append(f"- Trial: {trial.title}")
        lines.append(f"- Phase: {trial.phase}")
        lines.append(f"- Indication: {trial.condition}")
        lines.append(f"- Expected Readout: {trial.date.strftime('%Y-%m-%d')} (in {days_out} days)")
        lines.append(f"- NCT ID: [{trial.nct_id}]({trial.url})")
        lines.append(f"- Sponsor: {trial.sponsor}")
        lines.append("")
        lines.append("**Research Highlights**")
        window_text = confirmed_window or f"{trial.date.strftime('%Y-%m-%d')} (subject to latest disclosures)"
        lines.append(f"- **Expected announcement window:** {window_text}")
        lines.append("")
        lines.append(research)
        lines.append("")
        lines.append("**Stock Impact Commentary**")
        lines.append(f"- {impact_comment}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def enforce_opportunity_count(report: str, count: int) -> str:
    patterns = [
        (r"Opportunities Covered:\s*\d+", f"Opportunities Covered: {count}"),
        (r"\*\*Opportunities Covered:\*\*\s*\d+", f"**Opportunities Covered:** {count}"),
    ]
    for pattern, replacement in patterns:
        report, subs = re.subn(pattern, replacement, report, count=1)
        if subs:
            return report
    return report


def validate_report_sections(
    report: str, entries: Sequence[Tuple[Trial, str, str, Optional[str]]]
) -> bool:
    for trial, _, _, _ in entries:
        header = f"## {trial.company.name} ({trial.company.ticker})"
        if header not in report:
            return False
    return True


def refine_report(
    report: str,
    api_key: Optional[str],
    trials: Sequence[Trial],
    perplexity_key: Optional[str],
) -> str:
    if not api_key:
        return report

    fresh_context_lines: List[str] = []
    if perplexity_key:
        for trial in trials:
            context = perplexity_research(perplexity_key, trial)
            fresh_context_lines.append(
                f"### {trial.company.name} ({trial.company.ticker})\n{context}"
            )
    else:
        fresh_context_lines.append(
            "Perplexity API key not provided; additional research context unavailable."
        )

    research_context = "\n\n".join(fresh_context_lines)

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a biotech equity research editor. Rewrite the provided report to resolve inconsistencies, "
        "clarify unsupported catalysts, and ensure every statement aligns with the verification details. "
        "For any trial without corroboration, clearly state that verification is pending and suggest how to validate it. "
        "Maintain the Markdown structure exactly as provided: preserve the Executive Summary if present and keep every '## Company (Ticker)' section intact without merging or removing opportunities."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            max_tokens=2500,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Here is fresh Perplexity research for each trial:\n\n"
                        f"{research_context}\n\n"
                        "Please use it to refine the current report below."
                    ),
                },
                {"role": "user", "content": report},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else report
    except Exception as exc:
        return report + "\n\n> OpenAI refinement failed: " + str(exc)


def run_report_review(report: str, api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a biotech equity research editor. Review the supplied report for coherence, internal consistency, "
        "and logical structure. Flag conflicting data or uncertain statements. If the report is sound, state that explicitly. "
        "Respond in markdown with a concise opening line followed by at most five bullet points summarizing observations."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception as exc:
        return f"OpenAI review failed: {exc}"


def perplexity_company_scan(
    api_key: Optional[str],
    company: Company,
    days_ahead: int,
) -> Optional[Trial]:
    if not api_key:
        return None

    endpoint = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    horizon_end = datetime.now(UTC) + timedelta(days=days_ahead)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    user_prompt = (
        f"Today is {today}. Investigate whether {company.name} ({company.ticker}) has a confirmed or management-guided"
        f" clinical trial readout, regulatory decision, or major data announcement expected on or before {horizon_end.strftime('%Y-%m-%d')}.")

    system_prompt = (
        "You are a biotech catalyst scout. Respond ONLY with compact JSON."
        " Return null if no credible announcement exists within the requested horizon."
        " When you find a catalyst, respond with:"
        " {\"title\": str, \"event_type\": str, \"announcement_date\": \"YYYY-MM-DD\","
        " \"source_url\": str, \"summary\": str}."
        " The date must be a single day (no ranges). Cite the most authoritative source."
    )

    payload = {
        "model": os.getenv("PERPLEXITY_MODEL", DEFAULT_PERPLEXITY_MODEL),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 600,
        "return_citations": True,
        "return_images": False,
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
        if response.status_code == 401:
            return None
        if response.status_code >= 400:
            return None
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None
        parsed_payload = extract_json_object(content)
        if not parsed_payload:
            return None
        if isinstance(parsed_payload, list):
            parsed_candidates = [item for item in parsed_payload if isinstance(item, dict)]
            parsed = parsed_candidates[0] if parsed_candidates else None
        elif isinstance(parsed_payload, dict):
            parsed = parsed_payload
        else:
            parsed = None
        if not parsed:
            return None

        announcement_date = parsed.get("announcement_date")
        source_url = parsed.get("source_url")
        title = parsed.get("title") or parsed.get("summary")
        if not announcement_date or not title:
            return None
        parsed_date = parse_iso_utc_date(announcement_date)
        if not parsed_date:
            return None
        now = datetime.now(UTC)
        if parsed_date < now or parsed_date > horizon_end:
            return None
        nct_fallback = f"MANUAL-{company.ticker}-{parsed_date.strftime('%Y%m%d')}"
        trial = Trial(
            company=company,
            sponsor=company.name,
            nct_id=nct_fallback,
            title=title,
            condition=parsed.get("event_type", "N/A"),
            phase="N/A",
            url=source_url or f"https://www.google.com/search?q={company.ticker}+clinical+trial",
            date=parsed_date,
        )
        return trial
    except requests.RequestException:
        return None


def run_supplemental_company_scan(
    companies: Sequence[Company],
    covered_tickers: Set[str],
    lookahead_days: int,
    perplexity_key: Optional[str],
) -> List[Trial]:
    supplemental_trials: List[Trial] = []
    if not perplexity_key:
        print("Perplexity API key not provided; skipping supplemental company scan.")
        return supplemental_trials

    companies_to_scan = [c for c in companies if c.ticker not in covered_tickers]
    if not companies_to_scan:
        return supplemental_trials

    cache = load_company_scan_cache()
    cache_entries: Dict[str, Any] = {}  # Preserve existing cache entries even when user declines reuse
    cache_window = 0
    use_cache = False
    cache_freshness: Optional[str] = None
    if cache:
        cache_entries = cache.get("entries", {}) or {}
        cache_window = int(cache.get("lookahead_days", 0))
        cache_timestamp = cache.get("generated_at")
        if cache_timestamp:
            parsed_ts = parse_iso_utc_date(cache_timestamp)
            if parsed_ts:
                cache_freshness = parsed_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                cache_freshness = str(cache_timestamp)
            print(
                "Supplemental scan cache last updated: "
                f"{cache_freshness if cache_freshness else cache_timestamp}."
            )
        if cache_window >= lookahead_days:
            use_cache = prompt_use_cache(cache_window, lookahead_days, cache_freshness)
            if use_cache:
                print("Reusing cached supplemental scan results.")  # Add message when cache is reused
                cache_entries = cache.get("entries", {}) or {}
        else:
            print(
                f"Existing supplemental scan cache covers {cache_window} days, below requested {lookahead_days}. Re-running scans."
            )

    updated_entries: Dict[str, Any] = dict(cache_entries)
    scanned_any = False

    for company in companies_to_scan:
        cache_entry = cache_entries.get(company.ticker) if use_cache else None

        if cache_entry:
            status = cache_entry.get("status")
            if status == "found":
                announcement_date = cache_entry.get("announcement_date")
                parsed_date = parse_iso_utc_date(announcement_date) if announcement_date else None
                if parsed_date:
                    manual_trial = Trial(
                        company=company,
                        sponsor=cache_entry.get("sponsor", company.name),
                        nct_id=cache_entry.get("nct_id")
                        or f"MANUAL-{company.ticker}-{parsed_date.strftime('%Y%m%d')}",
                        title=cache_entry.get("title", "Supplemental catalyst"),
                        condition=cache_entry.get("condition", "N/A"),
                        phase=cache_entry.get("phase", "N/A"),
                        url=cache_entry.get("url", ""),
                        date=parsed_date,
                    )
                    print(
                        f"Using cached supplemental catalyst for {company.ticker} (window {parsed_date.strftime('%Y-%m-%d')})."
                    )
                    supplemental_trials.append(manual_trial)
                    continue
                else:
                    print(
                        f"Cached catalyst for {company.ticker} has invalid date; re-running supplemental scan."
                    )
            elif status == "missing":
                print(f"Using cached result: no supplemental catalyst for {company.ticker}.")
                continue

        print(
            f"No qualifying ClinicalTrials.gov catalyst found for {company.ticker}; running supplemental news scan..."
        )
        manual_trial = perplexity_company_scan(perplexity_key, company, lookahead_days)
        scanned_any = True
        if manual_trial:
            print(
                f"  ↳ Supplemental catalyst identified on {manual_trial.date.strftime('%Y-%m-%d')} from news sources."
            )
            supplemental_trials.append(manual_trial)
            updated_entries[company.ticker] = {
                "status": "found",
                "company_name": company.name,
                "sponsor": manual_trial.sponsor,
                "nct_id": manual_trial.nct_id,
                "title": manual_trial.title,
                "condition": manual_trial.condition,
                "phase": manual_trial.phase,
                "url": manual_trial.url,
                "announcement_date": manual_trial.date.isoformat(),
            }
        else:
            print("  ↳ No confirmed supplemental catalyst detected.")
            updated_entries[company.ticker] = {
                "status": "missing",
                "company_name": company.name,
            }

    if not use_cache or scanned_any:
        target_window = max(cache_window, lookahead_days)
        save_company_scan_cache(target_window, updated_entries)

    return supplemental_trials


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Watchlist data file not found: {DATA_FILE}")

    companies = load_companies(DATA_FILE)
    print(f"Loaded {len(companies)} companies from watchlist.")
    studies = fetch_trials(LOOKAHEAD_DAYS)
    print(f"Fetched {len(studies)} clinical trial records from ClinicalTrials.gov.")
    matched_trials = match_trials_to_companies(companies, studies)
    print(f"Matched {len(matched_trials)} trials to watchlist sponsors.")
    upcoming_trials = filter_upcoming(matched_trials, LOOKAHEAD_DAYS)
    print(f"Within {LOOKAHEAD_DAYS} days: {len(upcoming_trials)} upcoming readouts to analyze.")

    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    covered_tickers = {trial.company.ticker for trial in upcoming_trials}
    supplemental_trials = run_supplemental_company_scan(
        companies,
        covered_tickers,
        LOOKAHEAD_DAYS,
        perplexity_key,
    )

    if supplemental_trials:
        upcoming_trials = sorted(upcoming_trials + supplemental_trials, key=lambda t: t.date)
        covered_tickers.update(trial.company.ticker for trial in supplemental_trials)
        print(f"Supplemental catalysts added: {len(supplemental_trials)} opportunities.")

    enriched: List[Tuple[Trial, str, str, Optional[str]]] = []
    for idx, trial in enumerate(upcoming_trials, start=1):
        print(
            f"[{idx}/{len(upcoming_trials)}] Researching {trial.company.ticker} (NCT {trial.nct_id}) via Perplexity..."
        )
        research = perplexity_research(perplexity_key, trial)
        confirmed_window = extract_confirmed_window(research)
        if confirmed_window:
            print(
                f"  ↳ Confirmed window detected: {confirmed_window}. Requesting OpenAI price impact view..."
            )
        else:
            print("  ↳ No confirmed window; recording placeholder impact commentary.")
        impact_comment = generate_price_impact_comment(
            openai_key,
            trial,
            research,
            confirmed_window,
        )
        enriched.append((trial, research, impact_comment, confirmed_window))

    executive_summary = generate_executive_summary(enriched)
    report = build_markdown(enriched, executive_summary)
    print("Skipping OpenAI refinement step (disabled).")
    report = enforce_opportunity_count(report, len(enriched))
    print("Skipping OpenAI review step (disabled).")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"biotech_readout_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
