"""
CYPEARL Phase 2 - Data Splitter for Calibration

Splits Phase 1 behavioral data into training and test sets for prompt validation.
The training set is used to build persona descriptions, the test set validates
if LLMs can predict held-out human responses.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
import hashlib


@dataclass
class HeldOutTrial:
    """A single held-out trial from Phase 1 data."""
    trial_id: str
    email_id: str
    email_type: str  # phishing or legitimate
    urgency_level: str
    sender_familiarity: str
    framing_type: str

    # Human response (ground truth for calibration)
    human_action: str  # click, report, ignore
    human_confidence: Optional[str] = None
    human_response_time_ms: Optional[int] = None

    # Email content
    subject: str = ""
    sender: str = ""
    body: str = ""

    # Qualitative reasoning from participants (for Chain-of-Thought prompting)
    details_noticed: Optional[str] = None      # What details participant noticed
    steps_taken: Optional[str] = None          # Steps taken to evaluate email
    decision_reason: Optional[str] = None      # Why they made this decision
    confidence_reason: Optional[str] = None    # What made them confident
    unsure_about: Optional[str] = None         # What they were unsure about


@dataclass
class SplitResult:
    """Result of splitting behavioral data."""
    persona_id: str
    persona_name: str

    # Training data (used to build prompt)
    train_trials: List[HeldOutTrial]
    train_statistics: Dict[str, Any]

    # Test data (held out for validation)
    test_trials: List[HeldOutTrial]
    test_statistics: Dict[str, Any]

    # Split metadata
    split_ratio: float
    random_seed: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def n_train(self) -> int:
        return len(self.train_trials)

    @property
    def n_test(self) -> int:
        return len(self.test_trials)


class DataSplitter:
    """
    Splits Phase 1 behavioral data for calibration validation.

    The key insight: we use real human responses as ground truth.
    Training data builds the persona description.
    Test data validates if the LLM can predict unseen human responses.
    """

    def __init__(
        self,
        split_ratio: float = 0.8,
        stratify_by_email_type: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize splitter.

        Args:
            split_ratio: Fraction of data for training (default 0.8)
            stratify_by_email_type: Ensure phishing/legit ratio preserved
            random_seed: For reproducibility (None = random)
        """
        self.split_ratio = split_ratio
        self.stratify = stratify_by_email_type
        self.seed = random_seed if random_seed else random.randint(0, 999999)

    def split(
        self,
        persona_data: Dict[str, Any],
        behavioral_trials: List[Dict[str, Any]],
        emails: List[Dict[str, Any]]
    ) -> SplitResult:
        """
        Split behavioral data for a single persona.

        Args:
            persona_data: Full persona definition from Phase 1
            behavioral_trials: List of individual trial responses from Phase 1
            emails: Email definitions to include content

        Returns:
            SplitResult with train/test splits
        """
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Convert trials to HeldOutTrial objects
        # FIX Issue #4: Create email lookup with multiple key formats
        # Phase 1 uses "E01", "E12" format; email_stimuli_phase2.json uses "email_01", "email_12"
        email_lookup = {}
        for e in emails:
            email_id = e.get('email_id', e.get('id', ''))
            # Add with original key
            email_lookup[email_id] = e
            # Also add normalized variants for cross-format lookup
            # "email_01" -> also accessible as "E01", "E1", "1"
            # "E01" -> also accessible as "email_01", "email_1"
            if email_id.startswith('email_'):
                num_part = email_id.replace('email_', '')
                email_lookup[f'E{num_part}'] = e  # email_01 -> E01
                email_lookup[f'E{int(num_part)}'] = e  # email_01 -> E1
            elif email_id.startswith('E'):
                num_part = email_id[1:]  # E01 -> 01
                email_lookup[f'email_{num_part}'] = e  # E01 -> email_01
                email_lookup[f'email_{int(num_part):02d}'] = e  # E1 -> email_01

        trials = []

        for t in behavioral_trials:
            email_id = t.get('email_id', '')
            email = email_lookup.get(email_id, {})

            # FIX: If not found, also check trial dict for email content (from synthetic trials)
            if not email:
                # Try to get email content from trial itself (synthetic trials include this)
                email = {
                    'subject_line': t.get('subject_line', t.get('subject', '')),
                    'sender_display': t.get('sender_display', t.get('sender', '')),
                    'body_text': t.get('body_text', t.get('body', '')),
                    'email_type': t.get('email_type', 'unknown'),
                    'urgency_level': t.get('urgency_level', 'low'),
                    'sender_familiarity': t.get('sender_familiarity', 'unfamiliar'),
                    'framing_type': t.get('framing_type', 'neutral'),
                }

            trials.append(HeldOutTrial(
                trial_id=t.get('trial_id', f"trial_{len(trials)}"),
                email_id=email_id,
                email_type=t.get('email_type', email.get('email_type', 'unknown')),
                urgency_level=t.get('urgency_level', email.get('urgency_level', 'low')),
                sender_familiarity=t.get('sender_familiarity', email.get('sender_familiarity', 'unfamiliar')),
                framing_type=t.get('framing_type', email.get('framing_type', 'neutral')),
                human_action=t.get('action', t.get('response', 'unknown')).lower(),
                human_confidence=t.get('confidence'),
                human_response_time_ms=t.get('response_time_ms', t.get('latency_ms')),
                subject=email.get('subject_line', email.get('subject', '')),
                sender=email.get('sender_display', email.get('sender', '')),
                body=email.get('body_text', email.get('body', '')),
                # Qualitative reasoning fields from participant responses
                details_noticed=t.get('details_noticed'),
                steps_taken=t.get('steps_taken'),
                decision_reason=t.get('decision_reason'),
                confidence_reason=t.get('confidence_reason'),
                unsure_about=t.get('unsure_about')
            ))

        if self.stratify:
            train_trials, test_trials = self._stratified_split(trials)
        else:
            train_trials, test_trials = self._random_split(trials)

        # Calculate statistics for each set
        train_stats = self._calculate_statistics(train_trials)
        test_stats = self._calculate_statistics(test_trials)

        return SplitResult(
            persona_id=persona_data.get('persona_id', ''),
            persona_name=persona_data.get('name', ''),
            train_trials=train_trials,
            train_statistics=train_stats,
            test_trials=test_trials,
            test_statistics=test_stats,
            split_ratio=self.split_ratio,
            random_seed=self.seed
        )

    def _stratified_split(
        self, trials: List[HeldOutTrial]
    ) -> Tuple[List[HeldOutTrial], List[HeldOutTrial]]:
        """Split while maintaining email type ratio."""
        phishing = [t for t in trials if t.email_type == 'phishing']
        legitimate = [t for t in trials if t.email_type == 'legitimate']

        random.shuffle(phishing)
        random.shuffle(legitimate)

        n_phish_train = int(len(phishing) * self.split_ratio)
        n_legit_train = int(len(legitimate) * self.split_ratio)

        train = phishing[:n_phish_train] + legitimate[:n_legit_train]
        test = phishing[n_phish_train:] + legitimate[n_legit_train:]

        random.shuffle(train)
        random.shuffle(test)

        return train, test

    def _random_split(
        self, trials: List[HeldOutTrial]
    ) -> Tuple[List[HeldOutTrial], List[HeldOutTrial]]:
        """Simple random split."""
        shuffled = trials.copy()
        random.shuffle(shuffled)

        n_train = int(len(shuffled) * self.split_ratio)
        return shuffled[:n_train], shuffled[n_train:]

    def _calculate_statistics(self, trials: List[HeldOutTrial]) -> Dict[str, Any]:
        """Calculate behavioral statistics from trials."""
        if not trials:
            return {}

        n = len(trials)

        # Action counts
        clicks = sum(1 for t in trials if t.human_action == 'click')
        reports = sum(1 for t in trials if t.human_action == 'report')
        ignores = sum(1 for t in trials if t.human_action == 'ignore')

        # By email type
        phishing_trials = [t for t in trials if t.email_type == 'phishing']
        legit_trials = [t for t in trials if t.email_type == 'legitimate']

        phishing_clicks = sum(1 for t in phishing_trials if t.human_action == 'click')
        legit_clicks = sum(1 for t in legit_trials if t.human_action == 'click')

        # Response times
        response_times = [t.human_response_time_ms for t in trials if t.human_response_time_ms]

        return {
            'n_trials': n,
            'n_phishing': len(phishing_trials),
            'n_legitimate': len(legit_trials),

            # Overall rates
            'click_rate': clicks / n if n > 0 else 0,
            'report_rate': reports / n if n > 0 else 0,
            'ignore_rate': ignores / n if n > 0 else 0,

            # Phishing-specific
            'phishing_click_rate': phishing_clicks / len(phishing_trials) if phishing_trials else 0,
            'phishing_report_rate': sum(1 for t in phishing_trials if t.human_action == 'report') / len(phishing_trials) if phishing_trials else 0,

            # Legitimate-specific (false positive = reporting legit as phishing)
            'legit_click_rate': legit_clicks / len(legit_trials) if legit_trials else 0,
            'false_positive_rate': sum(1 for t in legit_trials if t.human_action == 'report') / len(legit_trials) if legit_trials else 0,

            # Response times
            'mean_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,

            # By urgency
            'high_urgency_click_rate': self._rate_by_factor(trials, 'urgency_level', 'high', 'click'),
            'low_urgency_click_rate': self._rate_by_factor(trials, 'urgency_level', 'low', 'click'),

            # By familiarity
            'familiar_click_rate': self._rate_by_factor(trials, 'sender_familiarity', 'familiar', 'click'),
            'unfamiliar_click_rate': self._rate_by_factor(trials, 'sender_familiarity', 'unfamiliar', 'click'),
        }

    def _rate_by_factor(
        self,
        trials: List[HeldOutTrial],
        factor: str,
        value: str,
        action: str
    ) -> float:
        """Calculate action rate for specific factor value."""
        filtered = [t for t in trials if getattr(t, factor, None) == value]
        if not filtered:
            return 0.0
        return sum(1 for t in filtered if t.human_action == action) / len(filtered)

    def trials_to_dicts(self, trials: List[HeldOutTrial]) -> List[Dict[str, Any]]:
        """
        Convert HeldOutTrial objects to dicts for ICL processing.

        Args:
            trials: List of HeldOutTrial objects

        Returns:
            List of trial dicts suitable for ICL builder
        """
        return [
            {
                'trial_id': t.trial_id,
                'email_id': t.email_id,
                'email_type': t.email_type,
                'urgency_level': t.urgency_level,
                'sender_familiarity': t.sender_familiarity,
                'framing_type': t.framing_type,
                'action': t.human_action,
                'human_action': t.human_action,
                'response_time_ms': t.human_response_time_ms,
                'human_response_time_ms': t.human_response_time_ms,
                'confidence': t.human_confidence,
                'subject': t.subject,
                'sender': t.sender,
                'body': t.body,
                # Qualitative reasoning fields for Chain-of-Thought prompting
                'details_noticed': t.details_noticed,
                'steps_taken': t.steps_taken,
                'decision_reason': t.decision_reason,
                'confidence_reason': t.confidence_reason,
                'unsure_about': t.unsure_about
            }
            for t in trials
        ]

    def split_multiple(
        self,
        personas: List[Dict[str, Any]],
        all_behavioral_data: Dict[str, List[Dict[str, Any]]],
        emails: List[Dict[str, Any]]
    ) -> Dict[str, SplitResult]:
        """
        Split data for multiple personas.

        Args:
            personas: List of persona definitions
            all_behavioral_data: Dict mapping persona_id to list of trials
            emails: Email definitions

        Returns:
            Dict mapping persona_id to SplitResult
        """
        results = {}
        for persona in personas:
            pid = persona.get('persona_id', '')
            trials = all_behavioral_data.get(pid, [])

            if trials:
                results[pid] = self.split(persona, trials, emails)

        return results


def generate_synthetic_trials(
    persona: Dict[str, Any],
    emails: List[Dict[str, Any]],
    n_trials_per_email: int = 1
) -> List[Dict[str, Any]]:
    """
    Generate synthetic behavioral trials from persona statistics.

    This is useful when we don't have individual trial data but have
    aggregate statistics. We sample actions according to the persona's
    known behavioral rates.

    Args:
        persona: Persona definition with behavioral_statistics
        emails: List of email definitions
        n_trials_per_email: How many synthetic trials per email

    Returns:
        List of synthetic trial dicts
    """
    import random

    stats = persona.get('behavioral_statistics', {})
    effects = persona.get('email_interaction_effects', {})

    base_click_rate = stats.get('phishing_click_rate', 0.3)
    base_report_rate = stats.get('report_rate', 0.2)
    urgency_effect = effects.get('urgency_effect', 0.1)
    familiarity_effect = effects.get('familiarity_effect', 0.1)

    trials = []

    for email in emails:
        for trial_num in range(n_trials_per_email):
            # Adjust click rate based on email factors
            click_rate = base_click_rate

            # Urgency increases clicking
            if email.get('urgency_level') == 'high':
                click_rate += urgency_effect

            # Familiarity increases clicking
            if email.get('sender_familiarity') == 'familiar':
                click_rate += familiarity_effect

            # For legitimate emails, reduce click rate (they're not deceptive)
            if email.get('email_type') == 'legitimate':
                click_rate = max(0.5, 1 - base_click_rate)  # Mostly should click legit

            # Sample action
            rand = random.random()
            if rand < click_rate:
                action = 'click'
            elif rand < click_rate + base_report_rate:
                action = 'report'
            else:
                action = 'ignore'

            # Sample response time (log-normal distribution)
            # lognormvariate takes (mu, sigma) - mu=8.5 gives ~5000ms median
            response_time = int(random.lognormvariate(8.5, 0.5))

            trials.append({
                'trial_id': f"synth_{persona.get('persona_id')}_{email.get('email_id')}_{trial_num}",
                'email_id': email.get('email_id', ''),
                'email_type': email.get('email_type', 'unknown'),
                'urgency_level': email.get('urgency_level', 'low'),
                'sender_familiarity': email.get('sender_familiarity', 'unfamiliar'),
                'framing_type': email.get('framing_type', 'neutral'),
                'action': action,
                'response_time_ms': response_time,
                # CRITICAL FIX: Include email content for prompt building
                'subject_line': email.get('subject_line', email.get('subject', '')),
                'subject': email.get('subject_line', email.get('subject', '')),
                'sender_display': email.get('sender_display', email.get('sender', '')),
                'sender': email.get('sender_display', email.get('sender', '')),
                'body_text': email.get('body_text', email.get('body', '')),
                'body': email.get('body_text', email.get('body', '')),
            })

    return trials
