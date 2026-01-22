"""
CYPEARL Phase 2 - Decision Pattern Extractor

Analyzes training trials to extract conditional behavioral patterns.
These patterns are used to augment prompts with explicit decision rules
derived from actual persona behavior.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class DecisionPattern:
    """A conditional behavioral pattern extracted from training data."""
    condition: str  # e.g., "phishing|high|familiar"
    action: str  # click, report, ignore
    frequency: float  # 0.0-1.0, how often this action occurs under this condition
    count: int  # Number of trials matching this condition
    reasoning: str  # Human-readable explanation of the pattern

    def format_for_prompt(self) -> str:
        """Format pattern as a prompt-friendly string."""
        parts = self.condition.split("|")
        if len(parts) >= 3:
            email_type, urgency, familiarity = parts[0], parts[1], parts[2]
            condition_text = f"{urgency.capitalize()} urgency + {familiarity} sender"
            if email_type == "phishing":
                condition_text += " (phishing)"
        else:
            condition_text = self.condition

        pct = int(self.frequency * 100)
        return f"- {condition_text} → I {self.action} ~{pct}% of the time ({self.reasoning})"


@dataclass
class PatternSummary:
    """Summary of extracted patterns for a persona."""
    persona_id: str
    total_trials: int
    patterns: List[DecisionPattern] = field(default_factory=list)

    # Aggregate statistics
    overall_click_rate: float = 0.0
    phishing_click_rate: float = 0.0
    legitimate_click_rate: float = 0.0

    # Effect sizes
    urgency_effect: float = 0.0  # high_urgency_click - low_urgency_click
    familiarity_effect: float = 0.0  # familiar_click - unfamiliar_click

    def get_top_patterns(self, n: int = 5) -> List[DecisionPattern]:
        """Get top N patterns by frequency * count (most impactful)."""
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: p.frequency * p.count,
            reverse=True
        )
        return sorted_patterns[:n]

    def format_for_prompt(self, max_patterns: int = 5) -> str:
        """Format summary as prompt-friendly text."""
        lines = ["PATTERNS I'VE NOTICED IN MY BEHAVIOR:"]

        top_patterns = self.get_top_patterns(max_patterns)
        for pattern in top_patterns:
            if pattern.frequency >= 0.20:  # Only show patterns with >= 20% frequency
                lines.append(pattern.format_for_prompt())

        return "\n".join(lines)


class DecisionPatternExtractor:
    """
    Extracts conditional decision patterns from training trials.

    Analyzes how persona behavior varies by:
    - Email type (phishing vs legitimate)
    - Urgency level (high vs low)
    - Sender familiarity (familiar vs unfamiliar)
    - Framing type (threat vs reward)
    """

    def __init__(self, min_frequency: float = 0.20, min_count: int = 2):
        """
        Initialize extractor.

        Args:
            min_frequency: Minimum action frequency to include (default 0.20)
            min_count: Minimum trials per condition to include pattern
        """
        self.min_frequency = min_frequency
        self.min_count = min_count

    def extract_patterns(
        self,
        training_trials: List[Dict[str, Any]],
        persona_id: str = "",
        persona_traits: Optional[Dict[str, Any]] = None
    ) -> PatternSummary:
        """
        Extract decision patterns from training trials.

        Args:
            training_trials: List of trial dicts with human responses
            persona_id: Persona identifier
            persona_traits: Optional trait z-scores for reasoning generation

        Returns:
            PatternSummary with extracted patterns
        """
        if not training_trials:
            return PatternSummary(persona_id=persona_id, total_trials=0)

        # Group trials by condition
        condition_groups = self._group_by_conditions(training_trials)

        # Extract patterns from each group
        patterns = []
        for condition, trials in condition_groups.items():
            action_patterns = self._extract_action_patterns(
                condition, trials, persona_traits
            )
            patterns.extend(action_patterns)

        # Filter by minimum frequency and count
        patterns = [
            p for p in patterns
            if p.frequency >= self.min_frequency and p.count >= self.min_count
        ]

        # Calculate aggregate statistics
        summary = PatternSummary(
            persona_id=persona_id,
            total_trials=len(training_trials),
            patterns=patterns
        )

        self._calculate_aggregate_stats(summary, training_trials)

        return summary

    def _group_by_conditions(
        self,
        trials: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group trials by condition key (email_type|urgency|familiarity)."""
        groups = defaultdict(list)

        for trial in trials:
            email_type = trial.get('email_type', 'unknown')
            urgency = trial.get('urgency_level', 'low')
            familiarity = trial.get('sender_familiarity', 'unfamiliar')

            condition = f"{email_type}|{urgency}|{familiarity}"
            groups[condition].append(trial)

        return dict(groups)

    def _extract_action_patterns(
        self,
        condition: str,
        trials: List[Dict[str, Any]],
        persona_traits: Optional[Dict[str, Any]] = None
    ) -> List[DecisionPattern]:
        """Extract action frequency patterns for a condition."""
        if not trials:
            return []

        # Count actions
        action_counts = defaultdict(int)
        response_times = defaultdict(list)

        for trial in trials:
            action = trial.get('action', trial.get('human_action', 'unknown')).lower()
            action_counts[action] += 1

            rt = trial.get('response_time_ms', trial.get('human_response_time_ms'))
            if rt:
                response_times[action].append(rt)

        total = len(trials)
        patterns = []

        for action, count in action_counts.items():
            if action in ['click', 'report', 'ignore']:
                frequency = count / total

                # Generate reasoning
                avg_rt = None
                if response_times[action]:
                    avg_rt = statistics.mean(response_times[action])

                reasoning = self._generate_reasoning(
                    condition, action, frequency, avg_rt, persona_traits
                )

                patterns.append(DecisionPattern(
                    condition=condition,
                    action=action,
                    frequency=frequency,
                    count=count,
                    reasoning=reasoning
                ))

        return patterns

    def _generate_reasoning(
        self,
        condition: str,
        action: str,
        frequency: float,
        avg_response_time: Optional[float],
        persona_traits: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate human-readable reasoning for a pattern."""
        parts = condition.split("|")
        email_type = parts[0] if len(parts) > 0 else "unknown"
        urgency = parts[1] if len(parts) > 1 else "low"
        familiarity = parts[2] if len(parts) > 2 else "unfamiliar"

        # Check if this is impulsive behavior
        is_fast = avg_response_time and avg_response_time < 3000
        traits = persona_traits or {}
        is_impulsive = (
            traits.get('impulsivity_total', 0) > 0.5 or
            traits.get('crt_score', 0) < -0.5
        )

        # Generate context-appropriate reasoning
        if action == "click":
            if urgency == "high" and is_fast:
                return "urgency triggers quick action"
            elif familiarity == "familiar":
                return "familiar sender = trusted"
            elif is_impulsive:
                return "gut says click"
            else:
                return "looks legitimate enough"

        elif action == "report":
            if familiarity == "unfamiliar":
                return "unfamiliar sender raises suspicion"
            elif email_type == "phishing":
                return "something felt off"
            else:
                return "better safe than sorry"

        else:  # ignore
            if urgency == "low":
                return "not urgent, not important"
            elif familiarity == "unfamiliar":
                return "don't know them, don't care"
            else:
                return "not relevant right now"

    def _calculate_aggregate_stats(
        self,
        summary: PatternSummary,
        trials: List[Dict[str, Any]]
    ):
        """Calculate aggregate statistics for the summary."""
        if not trials:
            return

        total = len(trials)
        clicks = sum(1 for t in trials
                    if t.get('action', t.get('human_action', '')).lower() == 'click')

        summary.overall_click_rate = clicks / total if total > 0 else 0

        # Phishing-specific
        phishing_trials = [t for t in trials if t.get('email_type') == 'phishing']
        if phishing_trials:
            phishing_clicks = sum(1 for t in phishing_trials
                                 if t.get('action', t.get('human_action', '')).lower() == 'click')
            summary.phishing_click_rate = phishing_clicks / len(phishing_trials)

        # Legitimate-specific
        legit_trials = [t for t in trials if t.get('email_type') == 'legitimate']
        if legit_trials:
            legit_clicks = sum(1 for t in legit_trials
                              if t.get('action', t.get('human_action', '')).lower() == 'click')
            summary.legitimate_click_rate = legit_clicks / len(legit_trials)

        # Urgency effect
        high_urgency = [t for t in trials if t.get('urgency_level') == 'high']
        low_urgency = [t for t in trials if t.get('urgency_level') == 'low']

        if high_urgency and low_urgency:
            high_clicks = sum(1 for t in high_urgency
                             if t.get('action', t.get('human_action', '')).lower() == 'click')
            low_clicks = sum(1 for t in low_urgency
                            if t.get('action', t.get('human_action', '')).lower() == 'click')

            high_rate = high_clicks / len(high_urgency)
            low_rate = low_clicks / len(low_urgency)
            summary.urgency_effect = high_rate - low_rate

        # Familiarity effect
        familiar = [t for t in trials if t.get('sender_familiarity') == 'familiar']
        unfamiliar = [t for t in trials if t.get('sender_familiarity') == 'unfamiliar']

        if familiar and unfamiliar:
            fam_clicks = sum(1 for t in familiar
                            if t.get('action', t.get('human_action', '')).lower() == 'click')
            unfam_clicks = sum(1 for t in unfamiliar
                              if t.get('action', t.get('human_action', '')).lower() == 'click')

            fam_rate = fam_clicks / len(familiar)
            unfam_rate = unfam_clicks / len(unfamiliar)
            summary.familiarity_effect = fam_rate - unfam_rate


def extract_patterns_for_prompt(
    training_trials: List[Dict[str, Any]],
    persona_id: str = "",
    persona_traits: Optional[Dict[str, Any]] = None,
    max_patterns: int = 5
) -> str:
    """
    Convenience function to extract patterns and format for prompt.

    Args:
        training_trials: List of trial dicts
        persona_id: Persona identifier
        persona_traits: Optional trait z-scores
        max_patterns: Maximum patterns to include

    Returns:
        Formatted string for prompt inclusion
    """
    extractor = DecisionPatternExtractor()
    summary = extractor.extract_patterns(training_trials, persona_id, persona_traits)
    return summary.format_for_prompt(max_patterns)
