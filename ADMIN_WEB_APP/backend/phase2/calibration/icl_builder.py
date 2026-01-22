"""
CYPEARL Phase 2 - In-Context Learning (ICL) Example Builder

Builds diverse, representative examples from training trials for inclusion
in prompts. These examples help LLMs learn persona-specific behavioral patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random
from collections import defaultdict

from .data_splitter import HeldOutTrial


@dataclass
class ICLExample:
    """A single in-context learning example."""
    email_id: str
    email_type: str  # phishing or legitimate
    urgency_level: str
    sender_familiarity: str
    subject: str
    action: str  # click, report, ignore
    reasoning: str  # Inferred or actual reasoning for this decision
    response_time_ms: Optional[int] = None

    # Actual participant qualitative responses (for authentic CoT)
    details_noticed: Optional[str] = None      # What details participant noticed
    steps_taken: Optional[str] = None          # Steps taken to evaluate email
    decision_reason: Optional[str] = None      # Why they made this decision
    confidence_reason: Optional[str] = None    # What made them confident
    unsure_about: Optional[str] = None         # What they were unsure about

    def has_actual_reasoning(self) -> bool:
        """Check if this example has actual participant reasoning."""
        return bool(self.decision_reason or self.details_noticed)

    def format_minimal(self) -> str:
        """Format for BASELINE config - just action, minimal context."""
        type_label = "[PHISHING]" if self.email_type == "phishing" else "[LEGIT]"
        return f"- {type_label} {self.subject} → {self.action.upper()}"

    def format_with_reasoning(self) -> str:
        """Format for STATS config - action with brief reasoning."""
        type_label = "[PHISHING]" if self.email_type == "phishing" else "[LEGIT]"
        # Use actual decision_reason if available, else fallback to inferred
        reason = self.decision_reason if self.decision_reason else self.reasoning
        return f"""- {type_label} "{self.subject}"
  → {self.action.upper()} ({reason})"""

    def format_full_cot(self, is_impulsive: bool = False) -> str:
        """
        Format for COT config - full chain-of-thought reasoning.
        Uses ACTUAL participant reasoning when available.
        """
        type_label = "[PHISHING]" if self.email_type == "phishing" else "[LEGIT]"

        # If we have actual participant reasoning, use it
        if self.has_actual_reasoning():
            return self._format_actual_cot(type_label, is_impulsive)
        else:
            return self._format_synthetic_cot(type_label, is_impulsive)

    def _format_actual_cot(self, type_label: str, is_impulsive: bool) -> str:
        """Format using actual participant reasoning."""
        speed_note = ""
        if self.response_time_ms and self.response_time_ms < 3000:
            speed_note = f" (decided in {self.response_time_ms/1000:.1f}s)"

        if is_impulsive:
            # Shorter format for impulsive personas
            return f"""--- Example ---
Email: {type_label} "{self.subject}"
MY THOUGHT PROCESS:
- What I noticed: {self.details_noticed or 'Quick scan'}
- My decision: {self.decision_reason or self.reasoning}
- Action: {self.action.upper()}{speed_note}
"""
        else:
            # Full detailed format with all qualitative fields
            parts = [f'--- Example ---\nEmail: {type_label} "{self.subject}"\nMY THOUGHT PROCESS:']

            if self.details_noticed:
                parts.append(f"- Details I noticed: {self.details_noticed}")
            if self.steps_taken:
                parts.append(f"- Steps I took: {self.steps_taken}")
            if self.decision_reason:
                parts.append(f"- Why I decided this: {self.decision_reason}")
            if self.confidence_reason:
                parts.append(f"- What made me confident: {self.confidence_reason}")
            if self.unsure_about and self.unsure_about.lower() not in ['nothing', 'none', 'n/a', 'na']:
                parts.append(f"- What I was unsure about: {self.unsure_about}")

            parts.append(f"- Final action: {self.action.upper()}")
            return "\n".join(parts) + "\n"

    def _format_synthetic_cot(self, type_label: str, is_impulsive: bool) -> str:
        """Fallback format using synthetic/inferred reasoning."""
        if is_impulsive:
            speed_note = ""
            if self.response_time_ms and self.response_time_ms < 3000:
                speed_note = f" (decided in {self.response_time_ms/1000:.1f}s)"
            return f"""--- Example ---
Email: {type_label} "{self.subject}"
MY THOUGHT PROCESS:
- First reaction: {self.reasoning}
- Decision: {self.action.upper()}{speed_note}
"""
        else:
            return f"""--- Example ---
Email: {type_label} "{self.subject}"
MY THOUGHT PROCESS:
- First noticed: {self._get_first_notice()}
- Considered: {self.reasoning}
- Decision: {self.action.upper()}
"""

    def _get_first_notice(self) -> str:
        """Generate what an analytical person would notice first."""
        notices = []
        if self.urgency_level == "high":
            notices.append("Urgent tone")
        if self.sender_familiarity == "unfamiliar":
            notices.append("Unfamiliar sender")
        elif self.sender_familiarity == "familiar":
            notices.append("Familiar sender name")
        if self.email_type == "phishing":
            notices.append("Some unusual elements")
        return ", ".join(notices) if notices else "Standard email format"


class ICLExampleBuilder:
    """
    Builds diverse in-context learning examples from training trials.

    Selection strategy:
    - Ensures diversity across email types, actions, and conditions
    - Prioritizes examples that demonstrate persona's distinctive patterns
    - Generates persona-appropriate reasoning for each example
    """

    def __init__(self, persona_traits: Optional[Dict[str, Any]] = None):
        """
        Initialize the builder.

        Args:
            persona_traits: Optional trait z-scores for reasoning generation
        """
        self.persona_traits = persona_traits or {}

    def select_diverse_examples(
        self,
        training_trials: List[Dict[str, Any]],
        emails_dict: Dict[str, Dict[str, Any]],
        max_examples: int = 6
    ) -> List[ICLExample]:
        """
        Select diverse examples from training trials.

        Ensures representation of:
        - At least 1 phishing email that was clicked
        - At least 1 phishing email that was ignored/reported
        - At least 1 legitimate email that was clicked
        - Mix of urgency levels
        - Mix of familiarity levels

        Args:
            training_trials: List of trial dicts with human responses
            emails_dict: Dict mapping email_id to email details
            max_examples: Maximum examples to select (default 6)

        Returns:
            List of ICLExample objects
        """
        if not training_trials:
            return []

        # Categorize trials
        categories = self._categorize_trials(training_trials)

        selected = []
        used_email_ids = set()

        # Priority selection to ensure diversity
        priority_categories = [
            ("phishing", "click"),      # Phishing clicked - shows susceptibility
            ("phishing", "ignore"),     # Phishing ignored - shows detection
            ("phishing", "report"),     # Phishing reported - shows vigilance
            ("legitimate", "click"),    # Legit clicked - normal behavior
            ("legitimate", "ignore"),   # Legit ignored - varies by persona
        ]

        # First pass: ensure at least one from each priority category
        for email_type, action in priority_categories:
            key = f"{email_type}_{action}"
            if key in categories and categories[key]:
                # Try to pick one with different urgency/familiarity if possible
                candidates = [t for t in categories[key]
                             if t.get('email_id') not in used_email_ids]

                if candidates:
                    # Prefer diverse urgency/familiarity
                    trial = self._select_for_diversity(
                        candidates,
                        [self._get_trial_features(t) for t in selected]
                    )
                    selected.append(trial)
                    used_email_ids.add(trial.get('email_id'))

            if len(selected) >= max_examples:
                break

        # Second pass: fill remaining slots with diverse examples
        remaining_slots = max_examples - len(selected)
        if remaining_slots > 0:
            all_remaining = [
                t for t in training_trials
                if t.get('email_id') not in used_email_ids
            ]

            for _ in range(remaining_slots):
                if not all_remaining:
                    break

                trial = self._select_for_diversity(
                    all_remaining,
                    [self._get_trial_features(t) for t in selected]
                )
                selected.append(trial)
                used_email_ids.add(trial.get('email_id'))
                all_remaining = [t for t in all_remaining
                                if t.get('email_id') != trial.get('email_id')]

        # Convert to ICLExample objects
        examples = []
        for trial in selected:
            email_id = trial.get('email_id', '')
            email = emails_dict.get(email_id, {})

            example = ICLExample(
                email_id=email_id,
                email_type=trial.get('email_type', email.get('email_type', 'unknown')),
                urgency_level=trial.get('urgency_level', email.get('urgency_level', 'low')),
                sender_familiarity=trial.get('sender_familiarity', email.get('sender_familiarity', 'unfamiliar')),
                subject=email.get('subject_line', email.get('subject', 'Email'))[:60],
                action=trial.get('action', trial.get('human_action', 'ignore')),
                reasoning=self.infer_reasoning(trial, email),
                response_time_ms=trial.get('response_time_ms', trial.get('human_response_time_ms')),
                # Include actual qualitative responses from participants
                details_noticed=trial.get('details_noticed'),
                steps_taken=trial.get('steps_taken'),
                decision_reason=trial.get('decision_reason'),
                confidence_reason=trial.get('confidence_reason'),
                unsure_about=trial.get('unsure_about')
            )
            examples.append(example)

        return examples

    def infer_reasoning(
        self,
        trial: Dict[str, Any],
        email: Dict[str, Any],
        persona_description: str = ""
    ) -> str:
        """
        Get reasoning for this decision - prefer actual participant reasoning when available.

        Uses (in order of preference):
        1. Actual decision_reason from participant
        2. Actual details_noticed from participant
        3. Inferred reasoning based on observable data

        Args:
            trial: Trial data with action and response time
            email: Email content and characteristics
            persona_description: Optional persona description for context

        Returns:
            Brief reasoning string (1 sentence)
        """
        # PREFER actual participant reasoning when available
        actual_reason = trial.get('decision_reason')
        if actual_reason and actual_reason.strip() and actual_reason.lower() not in ['n/a', 'na', 'none', '']:
            # Truncate to reasonable length for single-line display
            return actual_reason[:100] + ('...' if len(actual_reason) > 100 else '')

        # Fallback to details_noticed if no decision_reason
        details = trial.get('details_noticed')
        if details and details.strip() and details.lower() not in ['n/a', 'na', 'none', '']:
            return f"Noticed: {details[:80]}" + ('...' if len(details) > 80 else '')

        # Otherwise, generate synthetic reasoning
        return self._generate_synthetic_reasoning(trial, email)

    def _generate_synthetic_reasoning(
        self,
        trial: Dict[str, Any],
        email: Dict[str, Any]
    ) -> str:
        """Generate synthetic reasoning when actual participant data is unavailable."""
        action = trial.get('action', trial.get('human_action', 'ignore')).lower()
        email_type = trial.get('email_type', email.get('email_type', 'unknown'))
        urgency = trial.get('urgency_level', email.get('urgency_level', 'low'))
        familiarity = trial.get('sender_familiarity', email.get('sender_familiarity', 'unfamiliar'))
        response_time = trial.get('response_time_ms', trial.get('human_response_time_ms', 5000))

        # Fast decision (< 3 seconds)
        is_fast = response_time and response_time < 3000

        # Generate reasoning based on action + context
        if action == 'click':
            if email_type == 'phishing':
                # Clicked phishing - need believable reason
                if urgency == 'high' and is_fast:
                    return "Urgent message, needed to act fast"
                elif familiarity == 'familiar':
                    return "Sender looked familiar, seemed trustworthy"
                elif urgency == 'high':
                    return "Time pressure made me act quickly"
                else:
                    return "Looked legitimate enough, didn't think twice"
            else:
                # Clicked legitimate
                if is_fast:
                    return "Quick action on expected email"
                else:
                    return "Verified it was legitimate and clicked"

        elif action == 'report':
            if email_type == 'phishing':
                # Correctly reported phishing
                if familiarity == 'unfamiliar':
                    return "Unfamiliar sender raised red flags"
                elif urgency == 'high':
                    return "Too pushy - suspicious"
                else:
                    return "Something felt off about this one"
            else:
                # False positive - reported legitimate
                return "Wasn't sure, better safe than sorry"

        else:  # ignore
            if email_type == 'phishing':
                # Ignored phishing (good but passive)
                if familiarity == 'unfamiliar':
                    return "Don't recognize sender, not worth my time"
                else:
                    return "Didn't seem important enough to click"
            else:
                # Ignored legitimate
                return "Not relevant to me right now"

    def format_for_prompt(
        self,
        examples: List[ICLExample],
        style: str = "minimal",
        is_impulsive: bool = False
    ) -> str:
        """
        Format examples for inclusion in a prompt.

        Args:
            examples: List of ICLExample objects
            style: "minimal" (BASELINE), "reasoning" (STATS), or "cot" (COT)
            is_impulsive: Whether the persona is impulsive (affects COT format)

        Returns:
            Formatted string block for prompt
        """
        if not examples:
            return ""

        if style == "minimal":
            header = "Here's how I've responded to emails before:"
            lines = [ex.format_minimal() for ex in examples]
            return f"{header}\n" + "\n".join(lines)

        elif style == "reasoning":
            header = "Examples of my past email decisions:"
            lines = [ex.format_with_reasoning() for ex in examples]
            return f"{header}\n" + "\n".join(lines)

        else:  # cot
            header = "HOW I'VE ACTUALLY THOUGHT THROUGH EMAILS:"
            blocks = [ex.format_full_cot(is_impulsive) for ex in examples]
            return f"{header}\n\n" + "\n".join(blocks)

    def _categorize_trials(
        self,
        trials: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize trials by email_type + action."""
        categories = defaultdict(list)

        for trial in trials:
            email_type = trial.get('email_type', 'unknown')
            action = trial.get('action', trial.get('human_action', 'unknown')).lower()
            key = f"{email_type}_{action}"
            categories[key].append(trial)

        return dict(categories)

    def _get_trial_features(self, trial: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract key features for diversity comparison."""
        return (
            trial.get('email_type', 'unknown'),
            trial.get('urgency_level', 'low'),
            trial.get('sender_familiarity', 'unfamiliar')
        )

    def _select_for_diversity(
        self,
        candidates: List[Dict[str, Any]],
        existing_features: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Select candidate that maximizes diversity from existing selections.
        """
        if not candidates:
            return {}

        if not existing_features:
            return random.choice(candidates)

        # Score candidates by how different they are from existing
        best_score = -1
        best_candidate = candidates[0]

        for candidate in candidates:
            features = self._get_trial_features(candidate)

            # Count how many dimensions differ from all existing
            score = 0
            for existing in existing_features:
                diff_count = sum(1 for a, b in zip(features, existing) if a != b)
                score += diff_count

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate


def build_icl_block(
    training_trials: List[Dict[str, Any]],
    emails_dict: Dict[str, Dict[str, Any]],
    style: str = "minimal",
    is_impulsive: bool = False,
    max_examples: int = 6,
    persona_traits: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to build ICL block in one call.

    Args:
        training_trials: List of trial dicts with human responses
        emails_dict: Dict mapping email_id to email details
        style: "minimal", "reasoning", or "cot"
        is_impulsive: Whether persona is impulsive
        max_examples: Maximum examples to include
        persona_traits: Optional persona traits for reasoning

    Returns:
        Formatted ICL block string
    """
    builder = ICLExampleBuilder(persona_traits)
    examples = builder.select_diverse_examples(
        training_trials, emails_dict, max_examples
    )
    return builder.format_for_prompt(examples, style, is_impulsive)
