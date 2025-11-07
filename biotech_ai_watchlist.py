#!/usr/bin/env python3
"""Biotech catalyst tracker that enriches matches with Perplexity research."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from openai import OpenAI


DATA_FILE = Path(__file__).with_name("small_cap_biotech_phase2_3_readouts_2025_2026.json")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


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


def parse_date(struct: Optional[Dict[str, str]]) -> Optional[datetime]:
    if not struct:
        return None
    value = struct.get("date")
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def fetch_trials(days_ahead: int) -> List[Dict]:
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    start = datetime.utcnow().strftime("%Y-%m-%d")
    end = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    query = (
        f"(AREA[PrimaryCompletionDate]RANGE[{start},{end}] OR "
        f"AREA[CompletionDate]RANGE[{start},{end}]) "
        "AND AREA[StudyType]Interventional AND AREA[Phase](Phase 2 OR Phase 3)"
    )

    from openai import OpenAI
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
    company_map = {normalize(c.name): c for c in companies}
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
        matched_company: Optional[Company] = None

        for key, company in company_map.items():
            if key in sponsor_norm or sponsor_norm in key:
                matched_company = company
                break

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
    now = datetime.utcnow()
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

    today = datetime.utcnow().strftime("%Y-%m-%d")
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


def build_markdown(entries: Sequence[Tuple[Trial, str]]) -> str:
    lines: List[str] = []
    lines.append("# Biotech Catalyst Research Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Lookahead Window: {LOOKAHEAD_DAYS} days")
    lines.append(f"Opportunities Covered: {len(entries)}")
    lines.append("")

    if not entries:
        lines.append("No qualifying trial readouts within the next 90 days for the provided company list.")
        return "\n".join(lines)

    for trial, research in entries:
        days_out = (trial.date - datetime.utcnow()).days
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
        lines.append(
            f"- **Expected announcement window:** {trial.date.strftime('%Y-%m-%d')} (subject to latest disclosures)"
        )
        lines.append("")
        lines.append(research)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


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
        "For any trial without corroboration, either remove it or clearly state that verification is pending and suggest how to validate it. "
        "Maintain the Markdown structure, preserve factual data that is supported, and keep sections organized by company."
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


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Watchlist data file not found: {DATA_FILE}")

    companies = load_companies(DATA_FILE)
    studies = fetch_trials(LOOKAHEAD_DAYS)
    matched_trials = match_trials_to_companies(companies, studies)
    upcoming_trials = filter_upcoming(matched_trials, LOOKAHEAD_DAYS)

    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    enriched: List[Tuple[Trial, str]] = []
    for trial in upcoming_trials:
        research = perplexity_research(perplexity_key, trial)
        enriched.append((trial, research))

    report = build_markdown(enriched)
    trials_only = [trial for trial, _ in enriched]
    report = refine_report(report, openai_key, trials_only, perplexity_key)
    review = run_report_review(report, openai_key)
    if review:
        report += "\n\n## AI Quality Review\n\n" + review
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"biotech_readout_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
