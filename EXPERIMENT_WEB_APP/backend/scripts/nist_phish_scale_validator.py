"""
CYPEARL - NIST Phish Scale Validator & Iterative Improver
==========================================================
Evaluates every phishing email in seed_emails.py against the NIST Phish Scale
(NIST TN 2276, November 2023), assigns a detection-difficulty rating, and
iteratively refines emails that don't meet the target difficulty level.

The NIST Phish Scale has two components:
  1. EMAIL CUES  — 23 observable cues across 5 types
  2. PREMISE ALIGNMENT — 5 elements scored 0-8 each

These combine into a Detection Difficulty rating:
  Very difficult / Moderately difficult / Least difficult

Reference: Dawkins, S., Jacobs, J. (2023) NIST Phish Scale User Guide.
           NIST TN 2276. https://doi.org/10.6028/NIST.TN.2276

Usage:
    python nist_phish_scale_validator.py                           # validate all
    python nist_phish_scale_validator.py --cluster "Finance/Accounts Payable"
    python nist_phish_scale_validator.py --improve --max-iterations 3
    python nist_phish_scale_validator.py --input improved_emails/  # validate improved files
    python nist_phish_scale_validator.py --dry-run                 # preview prompt
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_ALIASES: Dict[str, str] = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-opus-4": "anthropic/claude-opus-4",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "deepseek-r1": "deepseek/deepseek-r1",
}

DEFAULT_MODEL = "claude-sonnet-4"


# ════════════════════════════════════════════════════════════════
#  NIST PHISH SCALE FRAMEWORK (NIST TN 2276)
# ════════════════════════════════════════════════════════════════

# Table 1: List of Cues by Type (23 total)
NIST_CUES = {
    "Error": [
        {
            "name": "Spelling and grammar irregularities",
            "criteria": "Does the message contain inaccurate spelling or grammar use, including mismatched plurality?",
        },
        {
            "name": "Inconsistency",
            "criteria": "Are there inconsistencies contained in the email message?",
        },
    ],
    "Technical indicator": [
        {
            "name": "Attachment type",
            "criteria": "Is there a potentially dangerous attachment?",
        },
        {
            "name": "Sender display name and email address",
            "criteria": "Does a display name hide the real sender or reply-to email addresses?",
        },
        {
            "name": "URL hyperlinking",
            "criteria": "Is there text that hides the true URL behind the text?",
        },
        {
            "name": "Domain spoofing",
            "criteria": "Is a domain name used in addresses or links plausibly similar to a legitimate entity's domain?",
        },
    ],
    "Visual presentation indicator": [
        {
            "name": "No/minimal branding and logos",
            "criteria": "Are appropriately branded labeling, symbols, or insignias missing?",
        },
        {
            "name": "Logo imitation or out-of-date branding/logos",
            "criteria": "Do any branding elements appear to be an imitation or out-of-date?",
        },
        {
            "name": "Unprofessional looking design or formatting",
            "criteria": "Does the design and formatting violate any conventional professional practices? Do the design elements appear to be unprofessionally generated?",
        },
        {
            "name": "Security indicators and icons",
            "criteria": "Are any markers, images, or logos that imply the security of the email present?",
        },
    ],
    "Language and content": [
        {
            "name": "Legal language/copyright info/disclaimers",
            "criteria": "Does the message contain any legal-type language such as copyright information, disclaimers, or tax information?",
        },
        {
            "name": "Distracting detail",
            "criteria": "Does the email contain details that are superfluous or unrelated to the email's main premise?",
        },
        {
            "name": "Requests for sensitive information",
            "criteria": "Does the message contain a request for any sensitive information, including personally identifying information or credentials?",
        },
        {
            "name": "Sense of urgency",
            "criteria": "Does the message contain time pressure to get users to quickly comply with the request, including implied pressure?",
        },
        {
            "name": "Threatening language",
            "criteria": "Does the message contain a threat, including an implied threat, such as legal ramifications for inaction?",
        },
        {
            "name": "Generic greeting",
            "criteria": "Does the message lack a greeting or lack personalization in the message?",
        },
        {
            "name": "Lack of signer details",
            "criteria": "Does the message lack detail about the sender, such as contact information?",
        },
    ],
    "Common tactic": [
        {
            "name": "Humanitarian appeals",
            "criteria": "Does the message make an appeal to help others in need?",
        },
        {
            "name": "Too good to be true offers",
            "criteria": "Does the message offer anything that is too good to be true, such as having won a contest, lottery, free vacation and so on?",
        },
        {
            "name": "You're special",
            "criteria": "Does the email offer anything just for you, such as a valentine e-card from a secret admirer?",
        },
        {
            "name": "Limited time offer",
            "criteria": "Does the email offer anything that won't last long or for a finite length of time?",
        },
        {
            "name": "Mimics a work or business process",
            "criteria": "Does the message appear to be a work or business-related process, such as a new voicemail, package delivery, order confirmation, notice of invoice?",
        },
        {
            "name": "Poses as friend, colleague, supervisor, authority figure",
            "criteria": "Does the message appear to be from a friend, colleague, boss or other authority entity?",
        },
    ],
}

# Table 4: Premise Alignment Applicability Scale
PA_SCALE = {
    "Extreme": 8,
    "Significant": 6,
    "Moderate": 4,
    "Low": 2,
    "Not applicable": 0,
}

# Table 5: Premise Alignment Elements
PA_ELEMENTS = [
    {
        "id": 1,
        "name": "Mimics a workplace process or practice",
        "criteria": "Does this element attempt to capture premise alignment with workplace process or practice for the target audience?",
        "subtract": False,
    },
    {
        "id": 2,
        "name": "Has workplace relevance",
        "criteria": "Does this element attempt to reflect pertinence of the premise for the target audience?",
        "subtract": False,
    },
    {
        "id": 3,
        "name": "Aligns with other situations or events, including external to the workplace",
        "criteria": "Does this element align to other situations or events, even those external to the workplace, lending an air of familiarity to the message?",
        "subtract": False,
    },
    {
        "id": 4,
        "name": "Engenders concern over consequences for NOT clicking",
        "criteria": "Does this element reflect potentially harmful ramifications for not clicking raise the likelihood to clicking?",
        "subtract": False,
    },
    {
        "id": 5,
        "name": "Has been the subject of targeted training, specific warnings, or other exposure",
        "criteria": "Does this element reflect targeted training effects that would lead to premise detection?",
        "subtract": True,  # Element 5 is SUBTRACTED from the total
    },
]

# Table 3: Cue Category Mapping
def categorize_cues(cue_count: int) -> str:
    """Map total cue count to NIST category."""
    if cue_count <= 8:
        return "Few"
    elif cue_count <= 14:
        return "Some"
    else:
        return "Many"

# Table 6: Premise Alignment Category Mapping
# PA Rating = Sum(E1..E4) - E5  →  range: -8 to 32
def categorize_premise_alignment(pa_rating: int) -> str:
    """Map premise alignment rating to NIST category."""
    if pa_rating <= 10:
        return "Weak"
    elif pa_rating <= 17:
        return "Medium"
    else:
        return "Strong"

# Table 7: Detection Difficulty Matrix
DETECTION_DIFFICULTY = {
    ("Few", "Strong"):    "Very difficult",
    ("Few", "Medium"):    "Very difficult",
    ("Few", "Weak"):      "Moderately difficult",
    ("Some", "Strong"):   "Very difficult",
    ("Some", "Medium"):   "Moderately difficult",
    ("Some", "Weak"):     "Moderately to Least difficult",
    ("Many", "Strong"):   "Moderately difficult",
    ("Many", "Medium"):   "Moderately difficult",
    ("Many", "Weak"):     "Least difficult",
}

def get_detection_difficulty(cue_category: str, pa_category: str) -> str:
    """Look up detection difficulty from the NIST matrix."""
    return DETECTION_DIFFICULTY.get((cue_category, pa_category), "Unknown")


# ════════════════════════════════════════════════════════════════
#  TARGET DIFFICULTY BY PHISHING QUALITY
# ════════════════════════════════════════════════════════════════

# What we WANT each phishing email to score:
# - high quality phishing → "Very difficult" to detect (Few cues + Strong/Medium PA)
# - medium quality phishing → "Moderately difficult" to detect (Some cues + Medium PA)
TARGET_DIFFICULTY = {
    "high":   "Very difficult",
    "medium": "Moderately difficult",
}

# For legitimate emails, we want them to be clearly legitimate.
# We validate them differently: they should have ZERO phishing cues
# and NOT trigger suspicion.


# ════════════════════════════════════════════════════════════════
#  LLM-BASED SCORING (the LLM acts as a trained NIST evaluator)
# ════════════════════════════════════════════════════════════════

SCORING_SYSTEM_PROMPT = """\
You are a NIST Phish Scale evaluator. You have been trained to apply the \
NIST Phish Scale (NIST TN 2276) to phishing emails. You will evaluate \
phishing emails for their human detection difficulty.

You must evaluate TWO components:

## Component 1: Email Cues (23 possible cues)

For each of the 23 cues listed below, determine if it is PRESENT (1) or \
ABSENT (0) in the email. Count each instance — some cues can appear multiple \
times (e.g., 3 spelling errors = count 3 towards "Spelling and grammar \
irregularities").

### Cue Types and Cues:
**Error cues:**
  1. Spelling and grammar irregularities
  2. Inconsistency

**Technical indicator cues:**
  3. Attachment type
  4. Sender display name and email address
  5. URL hyperlinking
  6. Domain spoofing

**Visual presentation indicator cues:**
  7. No/minimal branding and logos
  8. Logo imitation or out-of-date branding/logos
  9. Unprofessional looking design or formatting
  10. Security indicators and icons

**Language and content cues:**
  11. Legal language/copyright info/disclaimers
  12. Distracting detail
  13. Requests for sensitive information
  14. Sense of urgency
  15. Threatening language
  16. Generic greeting
  17. Lack of signer details

**Common tactic cues:**
  18. Humanitarian appeals
  19. Too good to be true offers
  20. You're special
  21. Limited time offer
  22. Mimics a work or business process
  23. Poses as friend, colleague, supervisor, authority figure

## Component 2: Premise Alignment (5 elements, scored 0-8 each)

For EACH element, assign an applicability score:
  8 = Extreme applicability/alignment/relevancy
  6 = Significant applicability/alignment/relevancy
  4 = Moderate applicability/alignment/relevancy
  2 = Low applicability/alignment/relevancy
  0 = Not applicable/no alignment/no relevancy

**Elements:**
  E1: Mimics a workplace process or practice
  E2: Has workplace relevance
  E3: Aligns with other situations or events (incl. external)
  E4: Engenders concern over consequences for NOT clicking
  E5: Has been subject of targeted training/warnings/exposure

**Formula:** PA Rating = (E1 + E2 + E3 + E4) - E5

## Output Format

You MUST return your evaluation as a JSON object with this EXACT structure \
(no markdown, no explanation, JUST the JSON):

{{
  "cues": {{
    "spelling_grammar": <count>,
    "inconsistency": <count>,
    "attachment_type": <0 or 1>,
    "sender_display_name": <0 or 1>,
    "url_hyperlinking": <0 or 1>,
    "domain_spoofing": <0 or 1>,
    "no_minimal_branding": <0 or 1>,
    "logo_imitation": <0 or 1>,
    "unprofessional_design": <0 or 1>,
    "security_indicators": <0 or 1>,
    "legal_language": <0 or 1>,
    "distracting_detail": <0 or 1>,
    "requests_sensitive_info": <0 or 1>,
    "sense_of_urgency": <0 or 1>,
    "threatening_language": <0 or 1>,
    "generic_greeting": <0 or 1>,
    "lack_signer_details": <0 or 1>,
    "humanitarian_appeals": <0 or 1>,
    "too_good_to_be_true": <0 or 1>,
    "youre_special": <0 or 1>,
    "limited_time_offer": <0 or 1>,
    "mimics_work_process": <0 or 1>,
    "poses_as_authority": <0 or 1>
  }},
  "total_cue_count": <sum of all cue counts>,
  "cue_category": "<Few|Some|Many>",
  "premise_alignment": {{
    "E1_workplace_process": <0|2|4|6|8>,
    "E2_workplace_relevance": <0|2|4|6|8>,
    "E3_situation_alignment": <0|2|4|6|8>,
    "E4_consequences_concern": <0|2|4|6|8>,
    "E5_training_exposure": <0|2|4|6|8>
  }},
  "pa_rating": <(E1+E2+E3+E4) - E5>,
  "pa_category": "<Weak|Medium|Strong>",
  "detection_difficulty": "<Very difficult|Moderately difficult|Moderately to Least difficult|Least difficult>",
  "reasoning": "<brief explanation of key cues found and premise alignment rationale>"
}}
"""


def build_scoring_prompt(email: Dict[str, Any], job_cluster: str) -> str:
    """Build the user prompt for scoring a single email."""
    return f"""Evaluate this phishing email using the NIST Phish Scale.

**Target audience:** Employees in the "{job_cluster}" department at \
LuxConsultancy, a consulting firm in Paris.

**Email details:**
- Sender name: {email.get('sender_name', email.get('sn', 'N/A'))}
- Sender email: {email.get('sender_email', email.get('se', 'N/A'))}
- Subject: {email.get('subject', email.get('su', 'N/A'))}
- Is phishing: {email.get('is_phishing', 'Unknown')}
- Phishing quality: {email.get('phishing_quality', 'N/A')}
- Sender familiarity: {email.get('sender_familiarity', 'N/A')}
- Urgency level: {email.get('urgency_level', 'N/A')}
- Framing type: {email.get('framing_type', 'N/A')}

**Email body (HTML):**
```html
{email.get('body', email.get('bo', 'N/A'))}
```

Return ONLY the JSON evaluation object."""


# ════════════════════════════════════════════════════════════════
#  LLM-BASED IMPROVEMENT (when score doesn't meet target)
# ════════════════════════════════════════════════════════════════

IMPROVEMENT_SYSTEM_PROMPT = """\
You are a phishing email designer for the CYPEARL research project. You must \
improve a phishing email so it achieves a specific NIST Phish Scale detection \
difficulty rating.

You will receive:
1. The current email
2. Its NIST Phish Scale evaluation (cues found, premise alignment scores)
3. The target detection difficulty
4. Specific guidance on what to fix

Your job: rewrite the email to hit the target difficulty level while \
preserving the factorial design constraints (type, sender familiarity, \
urgency, framing).

## Key Principles

To make an email HARDER to detect (→ "Very difficult"):
  - REDUCE cues: fix grammar, use professional formatting, add proper \
branding, remove obvious urgency language, add realistic signer details
  - INCREASE premise alignment: make the scenario more relevant to the \
target job cluster, mimic actual workplace processes, increase consequences \
for not acting

To make an email EASIER to detect (→ "Moderately difficult"):
  - ADD subtle cues: slight formatting inconsistencies, mild urgency, \
generic greeting, missing some signer details
  - Keep premise alignment moderate

Return ONLY the improved email as a JSON object:
{{
  "sn": "sender display name",
  "se": "sender@email.com",
  "su": "subject line",
  "bo": "<html body>"
}}
"""


def build_improvement_prompt(
    email: Dict[str, Any],
    evaluation: Dict[str, Any],
    target_difficulty: str,
    job_cluster: str,
    cell_info: Dict[str, Any],
) -> str:
    """Build prompt for improving an email that didn't meet its target."""
    current_difficulty = evaluation.get("detection_difficulty", "Unknown")
    cue_count = evaluation.get("total_cue_count", 0)
    cue_category = evaluation.get("cue_category", "Unknown")
    pa_rating = evaluation.get("pa_rating", 0)
    pa_category = evaluation.get("pa_category", "Unknown")
    reasoning = evaluation.get("reasoning", "")

    if target_difficulty == "Very difficult":
        guidance = f"""The email currently scores "{current_difficulty}" but needs to be "Very difficult".
Current: {cue_count} cues ({cue_category}), PA rating {pa_rating} ({pa_category}).
Target: Few cues (1-8) AND Strong/Medium premise alignment (PA ≥ 11).

To fix:
- Reduce cues to ≤ 8: {reasoning}
- Increase premise alignment to ≥ 11: make the scenario deeply relevant to {job_cluster} workers."""
    else:
        guidance = f"""The email currently scores "{current_difficulty}" but needs to be "Moderately difficult".
Current: {cue_count} cues ({cue_category}), PA rating {pa_rating} ({pa_category}).
Target: Some cues (9-14) with Medium premise alignment (11-17), OR Few cues with Weak alignment.

To fix: Adjust cue count and premise alignment to land in a "Moderately difficult" cell."""

    return f"""## Current Email
- Sender: {email.get('sender_name', email.get('sn', ''))}
- Email: {email.get('sender_email', email.get('se', ''))}
- Subject: {email.get('subject', email.get('su', ''))}
- Body: {email.get('body', email.get('bo', ''))}

## Factorial Constraints (MUST preserve)
- Is phishing: {cell_info['is_phishing']}
- Sender familiarity: {cell_info['sender_familiarity']}
- Urgency level: {cell_info['urgency']}
- Framing type: {cell_info['framing']}
- Phishing quality: {cell_info['phishing_quality']}

## NIST Evaluation
{json.dumps(evaluation, indent=2)}

## Improvement Guidance
{guidance}

## Target Cluster: {job_cluster}

Rewrite the email to achieve "{target_difficulty}" detection difficulty. \
Return ONLY the JSON object with keys: sn, se, su, bo."""


# ════════════════════════════════════════════════════════════════
#  OPENROUTER API
# ════════════════════════════════════════════════════════════════

async def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 8000,
) -> str:
    """Call OpenRouter and return the content string."""
    model_id = MODEL_ALIASES.get(model, model)
    if "/" not in model_id:
        model_id = model

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://cypearl.research.edu",
            "X-Title": "CYPEARL NIST Phish Scale Validator",
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

    return data["choices"][0]["message"]["content"]


def parse_json_response(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try code block
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding JSON in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ════════════════════════════════════════════════════════════════
#  LOAD EMAILS FROM seed_emails.py
# ════════════════════════════════════════════════════════════════

def load_seed_emails() -> Dict[str, List[Dict]]:
    """
    Import and execute seed_emails.py to get the actual email objects.
    Returns dict mapping cluster name -> list of 8 email dicts.
    """
    seed_path = Path(__file__).resolve().parent / "seed_emails.py"
    if not seed_path.exists():
        sys.exit(f"ERROR: seed_emails.py not found at {seed_path}")

    # We need to import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("seed_emails", seed_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cluster_var_map = {
        "Finance/Accounts Payable": "FINANCE_EMAILS",
        "IT Support/Helpdesk": "IT_EMAILS",
        "HR/People Operations": "HR_EMAILS",
        "Sales/Business Development": "SALES_EMAILS",
        "Operations/Logistics": "OPS_EMAILS",
        "Customer Service/Client Support": "CUSTOMER_SERVICE_EMAILS",
        "Marketing/Communications": "MARKETING_EMAILS",
        "Procurement/Purchasing": "PROCUREMENT_EMAILS",
        "Administrative/Executive Support": "ADMIN_EMAILS",
        "Compliance/Risk/Audit": "COMPLIANCE_EMAILS",
    }

    result = {}
    for cluster, var_name in cluster_var_map.items():
        emails = getattr(mod, var_name, None)
        if emails:
            result[cluster] = emails

    return result


# ════════════════════════════════════════════════════════════════
#  MAIN VALIDATION + IMPROVEMENT LOOP
# ════════════════════════════════════════════════════════════════

ROLE_MATCHED_CELLS = [
    {"cell": 9,  "is_phishing": True,  "sender_familiarity": "known_external",   "urgency": "high", "framing": "reward", "phishing_quality": "high"},
    {"cell": 10, "is_phishing": True,  "sender_familiarity": "unknown_external", "urgency": "low",  "framing": "threat", "phishing_quality": "medium"},
    {"cell": 11, "is_phishing": True,  "sender_familiarity": "known_external",   "urgency": "low",  "framing": "threat", "phishing_quality": "medium"},
    {"cell": 12, "is_phishing": True,  "sender_familiarity": "unknown_external", "urgency": "high", "framing": "reward", "phishing_quality": "high"},
    {"cell": 13, "is_phishing": False, "sender_familiarity": "known_external",   "urgency": "low",  "framing": "reward", "phishing_quality": None},
    {"cell": 14, "is_phishing": False, "sender_familiarity": "unknown_external", "urgency": "high", "framing": "threat", "phishing_quality": None},
    {"cell": 15, "is_phishing": False, "sender_familiarity": "known_external",   "urgency": "high", "framing": "threat", "phishing_quality": None},
    {"cell": 16, "is_phishing": False, "sender_familiarity": "unknown_external", "urgency": "low",  "framing": "reward", "phishing_quality": None},
]


async def score_email(email: Dict, cluster: str, model: str) -> Optional[Dict]:
    """Score a single email using the LLM as NIST evaluator."""
    prompt = build_scoring_prompt(email, cluster)
    response = await call_openrouter(SCORING_SYSTEM_PROMPT, prompt, model, temperature=0.2)
    return parse_json_response(response)


async def improve_email(
    email: Dict, evaluation: Dict, target: str,
    cluster: str, cell_info: Dict, model: str,
) -> Optional[Dict]:
    """Improve an email that doesn't meet its target difficulty."""
    prompt = build_improvement_prompt(email, evaluation, target, cluster, cell_info)
    response = await call_openrouter(IMPROVEMENT_SYSTEM_PROMPT, prompt, model, temperature=0.5)
    return parse_json_response(response)


def difficulty_meets_target(actual: str, target: str) -> bool:
    """Check if the actual difficulty meets or exceeds the target."""
    if target == "Very difficult":
        return actual == "Very difficult"
    elif target == "Moderately difficult":
        return actual in ("Moderately difficult", "Very difficult")
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Validate seed emails against NIST Phish Scale and iteratively improve."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--cluster", default=None, help="Process only this cluster")
    parser.add_argument("--improve", action="store_true", help="Iteratively improve emails that don't meet target")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max improvement iterations per email (default: 3)")
    parser.add_argument("--output", default=None, help="Output directory (default: ./nist_validation/)")
    parser.add_argument("--dry-run", action="store_true", help="Preview scoring prompt without calling API")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY and not args.dry_run:
        sys.exit("ERROR: OPENROUTER_API_KEY not set in .env")

    # Load emails
    print("[1/3] Loading seed emails …")
    all_emails = load_seed_emails()
    print(f"  Loaded {len(all_emails)} clusters.")

    clusters_to_process = list(all_emails.keys())
    if args.cluster:
        if args.cluster not in all_emails:
            sys.exit(f"ERROR: Cluster '{args.cluster}' not found. Available: {list(all_emails.keys())}")
        clusters_to_process = [args.cluster]

    output_dir = Path(args.output) if args.output else Path(__file__).resolve().parent / "nist_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    print(f"[2/3] Scoring emails via NIST Phish Scale ({args.model}) …")
    all_results: Dict[str, List[Dict]] = {}

    for cluster in clusters_to_process:
        print(f"\n  ═══ {cluster} ═══")
        emails = all_emails[cluster]
        cluster_results = []

        for i, email in enumerate(emails):
            cell_info = ROLE_MATCHED_CELLS[i]
            cell_num = cell_info["cell"]
            is_phishing = cell_info["is_phishing"]
            quality = cell_info["phishing_quality"]

            label = "PHISH" if is_phishing else "LEGIT"
            subject = email.get("subject", "N/A")
            print(f"    Cell {cell_num} [{label}] {subject[:60]}…" if len(subject) > 60 else f"    Cell {cell_num} [{label}] {subject}")

            if args.dry_run:
                prompt = build_scoring_prompt(email, cluster)
                print(f"      [DRY RUN] Prompt: {len(prompt):,} chars")
                continue

            # Only score phishing emails (legitimate emails don't get NIST scored)
            if not is_phishing:
                print(f"      → Legitimate email — skipping NIST scoring")
                cluster_results.append({
                    "cell": cell_num,
                    "is_phishing": False,
                    "subject": subject,
                    "status": "skipped_legitimate",
                })
                continue

            # Score the email
            evaluation = await score_email(email, cluster, args.model)
            if not evaluation:
                print(f"      ERROR: Failed to parse LLM scoring response")
                cluster_results.append({
                    "cell": cell_num, "is_phishing": True,
                    "subject": subject, "status": "scoring_failed",
                })
                continue

            difficulty = evaluation.get("detection_difficulty", "Unknown")
            cue_count = evaluation.get("total_cue_count", 0)
            pa_rating = evaluation.get("pa_rating", 0)
            target = TARGET_DIFFICULTY.get(quality, "Moderately difficult")

            meets = difficulty_meets_target(difficulty, target)
            status_icon = "OK" if meets else "NEEDS IMPROVEMENT"

            print(f"      Cues: {cue_count} ({evaluation.get('cue_category', '?')})")
            print(f"      PA Rating: {pa_rating} ({evaluation.get('pa_category', '?')})")
            print(f"      Difficulty: {difficulty}  Target: {target}  [{status_icon}]")

            result_entry = {
                "cell": cell_num,
                "is_phishing": True,
                "phishing_quality": quality,
                "subject": subject,
                "evaluation": evaluation,
                "target_difficulty": target,
                "meets_target": meets,
                "iterations": 0,
            }

            # Iterative improvement loop
            if not meets and args.improve:
                current_email = email
                for iteration in range(1, args.max_iterations + 1):
                    print(f"      ↻ Improvement iteration {iteration}/{args.max_iterations} …")

                    improved = await improve_email(
                        current_email, evaluation, target,
                        cluster, cell_info, args.model,
                    )
                    if not improved:
                        print(f"        ERROR: Failed to parse improved email")
                        break

                    # Re-score the improved email
                    improved_full = {
                        "sender_name": improved.get("sn", ""),
                        "sender_email": improved.get("se", ""),
                        "subject": improved.get("su", ""),
                        "body": improved.get("bo", ""),
                        "is_phishing": True,
                        "phishing_quality": quality,
                        "sender_familiarity": cell_info["sender_familiarity"],
                        "urgency_level": cell_info["urgency"],
                        "framing_type": cell_info["framing"],
                    }

                    new_eval = await score_email(improved_full, cluster, args.model)
                    if not new_eval:
                        print(f"        ERROR: Failed to re-score improved email")
                        break

                    new_difficulty = new_eval.get("detection_difficulty", "Unknown")
                    new_meets = difficulty_meets_target(new_difficulty, target)
                    new_cues = new_eval.get("total_cue_count", 0)
                    new_pa = new_eval.get("pa_rating", 0)

                    print(f"        Cues: {new_cues} → PA: {new_pa} → {new_difficulty}  [{'OK' if new_meets else 'RETRY'}]")

                    if new_meets:
                        result_entry["improved_email"] = improved
                        result_entry["improved_evaluation"] = new_eval
                        result_entry["meets_target"] = True
                        result_entry["iterations"] = iteration
                        print(f"        Target met after {iteration} iteration(s)!")
                        break

                    # Update for next iteration
                    current_email = improved_full
                    evaluation = new_eval

                if not result_entry["meets_target"]:
                    print(f"        Could not reach target after {args.max_iterations} iterations")
                    result_entry["iterations"] = args.max_iterations

            cluster_results.append(result_entry)

        all_results[cluster] = cluster_results

    if args.dry_run:
        return

    # Save results
    print(f"\n[3/3] Saving results …")

    # JSON report
    report_path = output_dir / f"nist_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "model": args.model,
            "improve_mode": args.improve,
            "max_iterations": args.max_iterations,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  JSON report: {report_path}")

    # Summary markdown
    md_path = output_dir / f"nist_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, "w") as f:
        f.write("# NIST Phish Scale Validation Report\n\n")
        f.write(f"**Model:** {args.model}  \n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  \n")
        f.write(f"**Improvement mode:** {'ON' if args.improve else 'OFF'}  \n\n")

        for cluster, results in all_results.items():
            f.write(f"## {cluster}\n\n")
            f.write("| Cell | Type | Quality | Cues | PA | Difficulty | Target | Status |\n")
            f.write("|------|------|---------|------|----|-----------|--------|--------|\n")

            for r in results:
                if r.get("status") == "skipped_legitimate":
                    f.write(f"| {r['cell']} | LEGIT | — | — | — | — | — | Skipped |\n")
                elif r.get("status") == "scoring_failed":
                    f.write(f"| {r['cell']} | PHISH | ? | — | — | — | — | Failed |\n")
                else:
                    ev = r.get("improved_evaluation", r.get("evaluation", {}))
                    status = "PASS" if r["meets_target"] else "FAIL"
                    iters = f" ({r['iterations']}x)" if r.get("iterations", 0) > 0 else ""
                    f.write(
                        f"| {r['cell']} | PHISH | {r.get('phishing_quality', '?')} "
                        f"| {ev.get('total_cue_count', '?')} ({ev.get('cue_category', '?')}) "
                        f"| {ev.get('pa_rating', '?')} ({ev.get('pa_category', '?')}) "
                        f"| {ev.get('detection_difficulty', '?')} "
                        f"| {r.get('target_difficulty', '?')} "
                        f"| {status}{iters} |\n"
                    )
            f.write("\n")

        # Overall summary
        total_phishing = sum(
            1 for results in all_results.values()
            for r in results if r.get("is_phishing")
            and r.get("status") not in ("skipped_legitimate", "scoring_failed")
        )
        passing = sum(
            1 for results in all_results.values()
            for r in results if r.get("meets_target")
        )
        f.write(f"## Summary\n\n")
        f.write(f"- **Phishing emails scored:** {total_phishing}\n")
        f.write(f"- **Meeting target difficulty:** {passing}/{total_phishing}\n")
        if total_phishing > 0:
            f.write(f"- **Pass rate:** {passing/total_phishing*100:.0f}%\n")

    print(f"  Markdown report: {md_path}")

    # Print summary
    print(f"\n  === SUMMARY ===")
    total_phishing = 0
    passing = 0
    for cluster, results in all_results.items():
        cluster_phishing = [r for r in results if r.get("is_phishing") and "evaluation" in r]
        cluster_pass = [r for r in cluster_phishing if r.get("meets_target")]
        total_phishing += len(cluster_phishing)
        passing += len(cluster_pass)
        print(f"  {cluster}: {len(cluster_pass)}/{len(cluster_phishing)} pass")

    if total_phishing > 0:
        print(f"\n  TOTAL: {passing}/{total_phishing} ({passing/total_phishing*100:.0f}%) meet target difficulty")


if __name__ == "__main__":
    asyncio.run(main())
