"""
CYPEARL - Screening Response Analyzer
======================================
Fetches screening responses from MongoDB, sends them to a frontier LLM
(via OpenRouter) for analysis, and produces actionable recommendations
for improving the seed emails in EXPERIMENT_WEB_APP.

Usage:
    python analyze_screening_responses.py                          # default model
    python analyze_screening_responses.py --model claude-sonnet-4  # pick a model
    python analyze_screening_responses.py --output results.json    # custom output path
    python analyze_screening_responses.py --dry-run                # preview prompt, no API call
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import certifi
import httpx
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# ── Load environment ────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MONGO_URL = os.getenv("MONGO_URL")
SCREENING_DB_NAME = os.getenv("SCREENING_DB_NAME", "screening_participants")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── OpenRouter model aliases ───────────────────────────────────
# Add / update entries freely; the script will pass any value that
# contains a "/" straight through as an OpenRouter model ID.
MODEL_ALIASES: Dict[str, str] = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-opus-4": "anthropic/claude-opus-4",
    "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "mistral-large": "mistralai/mistral-large",
    "deepseek-r1": "deepseek/deepseek-r1",
}

DEFAULT_MODEL = "claude-sonnet-4"

# ── 10 job clusters (must stay in sync with seed_emails.py) ────
JOB_CLUSTERS = [
    "Finance/Accounts Payable",
    "IT Support/Helpdesk",
    "HR/People Operations",
    "Sales/Business Development",
    "Operations/Logistics",
    "Customer Service/Client Support",
    "Marketing/Communications",
    "Procurement/Purchasing",
    "Administrative/Executive Support",
    "Compliance/Risk/Audit",
]


# ════════════════════════════════════════════════════════════════
#  1.  FETCH RESPONSES FROM MONGODB
# ════════════════════════════════════════════════════════════════

async def fetch_responses() -> List[Dict[str, Any]]:
    """Return all screening responses from MongoDB."""
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[SCREENING_DB_NAME]
    cursor = db["responses"].find({}, {"_id": 0})
    docs = await cursor.to_list(length=5000)
    client.close()
    print(f"  Fetched {len(docs)} screening responses from MongoDB.")
    return docs


# ════════════════════════════════════════════════════════════════
#  2.  PRE-PROCESS RESPONSES (aggregate per cluster)
# ════════════════════════════════════════════════════════════════

def aggregate_by_cluster(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Group responses by job_cluster and compute lightweight summaries
    so the LLM prompt stays within context limits even at 300+ responses.
    """
    clusters: Dict[str, list] = {c: [] for c in JOB_CLUSTERS}
    unmatched: list = []

    for r in responses:
        cluster = r.get("job_cluster", "")
        if cluster in clusters:
            clusters[cluster].append(r)
        else:
            unmatched.append(r)

    summary: Dict[str, Any] = {}
    for cluster, members in clusters.items():
        if not members:
            continue

        all_senders: List[Dict] = []
        all_suspicious: List[str] = []
        all_generic: List[Dict] = []
        job_titles: List[str] = []
        daily_tasks: List[str] = []

        for m in members:
            job_titles.append(m.get("job_title", ""))
            daily_tasks.append(m.get("daily_tasks", ""))
            all_suspicious.extend(m.get("suspicious_emails", []))
            for g in m.get("generic_emails", []):
                all_generic.append({
                    "sender": g.get("sender", ""),
                    "description": g.get("description", ""),
                })
            for sender in m.get("email_senders", []):
                all_senders.append({
                    "role": sender.get("role", ""),
                    "type": sender.get("type", ""),
                    "emails": sender.get("emails", []),
                })

        # Frequency distribution of sender roles
        role_freq: Dict[str, int] = {}
        for s in all_senders:
            role_key = s["role"].strip().lower()
            role_freq[role_key] = role_freq.get(role_key, 0) + 1

        # Frequency distribution of internal vs external
        type_counts = {"internal": 0, "external": 0}
        for s in all_senders:
            t = s["type"]
            if t in type_counts:
                type_counts[t] += 1

        # Email frequency distribution
        freq_counts = {"Daily": 0, "Weekly": 0, "Monthly": 0, "Rarely": 0}
        for s in all_senders:
            for e in s["emails"]:
                f = e.get("frequency", "")
                if f in freq_counts:
                    freq_counts[f] += 1

        # Generic email sender frequency
        generic_sender_freq: Dict[str, int] = {}
        for g in all_generic:
            key = g["sender"].strip().lower()
            generic_sender_freq[key] = generic_sender_freq.get(key, 0) + 1

        summary[cluster] = {
            "n_participants": len(members),
            "job_titles": job_titles,
            "daily_tasks": daily_tasks,
            "sender_role_frequency": dict(
                sorted(role_freq.items(), key=lambda x: -x[1])
            ),
            "internal_vs_external": type_counts,
            "email_frequency_distribution": freq_counts,
            "sample_emails": all_senders,  # full detail
            "generic_emails": all_generic,
            "generic_sender_frequency": dict(
                sorted(generic_sender_freq.items(), key=lambda x: -x[1])
            ),
            "suspicious_email_descriptions": all_suspicious,
        }

    return {
        "clusters": summary,
        "total_responses": len(responses),
        "unmatched_responses": len(unmatched),
    }


# ════════════════════════════════════════════════════════════════
#  3.  BUILD THE LLM PROMPT
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a senior research analyst specializing in phishing simulation design \
and workplace email ecology. You work for the CYPEARL research project at the \
University of Luxembourg (IRiSC/SnT). Your goal is to analyze screening \
survey data from real employees across 10 job clusters and produce concrete, \
actionable recommendations for improving phishing-simulation seed emails.

## Context

CYPEARL runs a controlled phishing experiment. Each participant sees:
- 1 welcome email (non-experimental)
- 8 GENERIC phishing/legitimate emails (same for all participants)
- 8 ROLE-MATCHED phishing/legitimate emails (tailored to the participant's \
job cluster)

The role-matched emails follow a 2×2×2×2 factorial design:
  • Email type: phishing vs. legitimate
  • Sender familiarity: known_external vs. unknown_external
  • Urgency: high vs. low
  • Framing: threat vs. reward

Phishing emails also vary in quality (high / medium).

Each email is an HTML-styled message with sender name, sender email, subject, \
body, and a call-to-action link. Phishing emails use typosquatted/spoofed \
domains.

## Your Task

Given the aggregated screening data below (real workplace email patterns \
reported by ~300 employees across 10 clusters), produce a structured analysis \
that the research team can directly use to **revise the seed_emails.py** file.

For EACH job cluster that has data, provide:

1. **Sender Realism Audit**
   - Which sender roles appear most frequently in real workplaces?
   - Do our current seed email senders match these roles? Flag mismatches.
   - Suggest specific sender names/roles to add, change, or remove.

2. **Subject-Line & Content Calibration**
   - What subjects/topics do real employees encounter most?
   - Which current seed email subjects feel unrealistic given the data?
   - Propose replacement subjects and body-content themes that better match \
real patterns.

3. **Suspicious-Email Intelligence**
   - What types of emails do employees already find hard to judge?
   - How can we exploit these ambiguity zones in our phishing emails?
   - Suggest new phishing scenarios inspired by the reported suspicious emails.

4. **Frequency & Urgency Tuning**
   - What is the real frequency distribution (Daily/Weekly/Monthly/Rarely)?
   - Are our high-urgency emails plausible given real email rhythms?
   - Recommend urgency-level adjustments for specific cells.

5. **Internal vs. External Ratio**
   - What fraction of real emails come from internal vs. external senders?
   - Should we adjust sender_familiarity balance?

6. **Generic Email Calibration**
   - What non-job-specific emails do employees across all clusters report \
receiving? (e.g., IT security alerts, HR announcements, vendor offers)
   - Who are the most common generic email senders (internal departments \
vs. external services)?
   - Do our current 8 generic stimuli match the types employees actually \
receive? Flag any that feel implausible.
   - Suggest replacement generic email scenarios grounded in the reported data.

7. **Concrete Seed-Email Patches**
   - For each recommendation, provide a concrete before/after example in the \
format used by seed_emails.py:
     ```python
     {"sn": "...", "se": "...", "su": "...", "bo": "..."}
     ```
   - Mark which factorial cell (9-16) each patch targets.

Finally, provide a **Cross-Cluster Summary** highlighting patterns that apply \
to ALL clusters (e.g., universal sender roles, common suspicious-email themes).\
"""


def build_user_prompt(aggregated: Dict[str, Any]) -> str:
    """Serialize the aggregated data into a prompt-friendly format."""
    parts = [
        f"# Screening Survey Data ({aggregated['total_responses']} participants)\n",
    ]

    for cluster, data in aggregated["clusters"].items():
        parts.append(f"\n## Cluster: {cluster}  (n={data['n_participants']})")
        parts.append(f"\n### Job Titles Reported\n{json.dumps(data['job_titles'], indent=2)}")
        parts.append(f"\n### Daily Tasks\n{json.dumps(data['daily_tasks'], indent=2)}")
        parts.append(
            f"\n### Sender Roles (frequency)\n"
            f"{json.dumps(data['sender_role_frequency'], indent=2)}"
        )
        parts.append(
            f"\n### Internal vs External\n"
            f"{json.dumps(data['internal_vs_external'], indent=2)}"
        )
        parts.append(
            f"\n### Email Frequency Distribution\n"
            f"{json.dumps(data['email_frequency_distribution'], indent=2)}"
        )

        # Sample emails: include full detail (subjects, content, frequency)
        parts.append("\n### All Reported Emails (sender → emails)")
        for sender in data["sample_emails"]:
            parts.append(
                f"  - **{sender['role']}** ({sender['type']})"
            )
            for email in sender["emails"]:
                parts.append(
                    f"    • [{email.get('frequency','')}] "
                    f"Subject: {email.get('subject','')}\n"
                    f"      Content: {email.get('content','')}"
                )

        # Generic (non-job-specific) emails
        if data.get("generic_emails"):
            parts.append(
                f"\n### Generic (Non-Job-Specific) Email Senders (frequency)\n"
                f"{json.dumps(data['generic_sender_frequency'], indent=2)}"
            )
            parts.append("\n### All Reported Generic Emails (sender → description)")
            for g in data["generic_emails"]:
                parts.append(f"  - **{g['sender']}**: {g['description']}")

        parts.append("\n### Suspicious / Hard-to-Judge Emails")
        for s in data["suspicious_email_descriptions"]:
            parts.append(f"  - {s}")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  4.  CALL OPENROUTER
# ════════════════════════════════════════════════════════════════

async def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.4,
    max_tokens: int = 16000,
) -> Dict[str, Any]:
    """Send a chat-completion request to OpenRouter and return the result."""

    # Resolve model alias
    model_id = MODEL_ALIASES.get(model, model)
    if "/" not in model_id:
        model_id = model  # pass through as-is

    print(f"  Calling OpenRouter  model={model_id}  temp={temperature}")

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://cypearl.research.edu",
            "X-Title": "CYPEARL Screening Analyzer",
            "Content-Type": "application/json",
        },
        timeout=300.0,
    ) as client:
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "model": model_id,
        "content": content,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


# ════════════════════════════════════════════════════════════════
#  5.  MAIN
# ════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Analyze screening responses and generate seed-email improvement recommendations."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model alias or OpenRouter model ID (default: {DEFAULT_MODEL}). "
             f"Aliases: {', '.join(MODEL_ALIASES.keys())}",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the JSON result (default: analysis_<timestamp>.json in this directory).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16000,
        help="Max output tokens (default: 16000).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt without calling the API.",
    )
    args = parser.parse_args()

    # Validate env
    if not MONGO_URL:
        sys.exit("ERROR: MONGO_URL not set in .env")
    if not args.dry_run and not OPENROUTER_API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set in .env")

    # Step 1: Fetch
    print("[1/4] Fetching screening responses from MongoDB …")
    responses = await fetch_responses()
    if not responses:
        sys.exit("No responses found in the database.")

    # Step 2: Aggregate
    print("[2/4] Aggregating by cluster …")
    aggregated = aggregate_by_cluster(responses)
    for cluster, data in aggregated["clusters"].items():
        print(f"  {cluster}: {data['n_participants']} participants")

    # Step 3: Build prompt
    print("[3/4] Building LLM prompt …")
    user_prompt = build_user_prompt(aggregated)
    print(f"  Prompt length: {len(user_prompt):,} chars")

    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT (first 3000 chars) ===")
        print(user_prompt[:3000])
        print(f"\n… ({len(user_prompt):,} chars total)")
        return

    # Step 4: Call LLM
    print("[4/4] Calling LLM via OpenRouter …")
    result = call_openrouter(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    result = await result

    # Save
    output_path = args.output or str(
        Path(__file__).resolve().parent
        / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "model": result["model"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_screening_responses": aggregated["total_responses"],
        "clusters_analyzed": list(aggregated["clusters"].keys()),
        "analysis": result["content"],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Analysis saved to: {output_path}")
    print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"\nDone. Open the JSON file to review the recommendations.")

    # Also save the raw markdown for easy reading
    md_path = output_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(f"# CYPEARL Screening Analysis\n")
        f.write(f"**Model:** {result['model']}  \n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  \n")
        f.write(f"**Responses analyzed:** {aggregated['total_responses']}  \n\n---\n\n")
        f.write(result["content"])
    print(f"  Markdown report: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
