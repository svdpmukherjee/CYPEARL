"""
CYPEARL Phase 2 - Uncertainty Quantification for LLM Outputs

Quantifies how confident the LLM is in its persona simulation.
Instead of single predictions, runs multiple samples to measure:
- Response distribution (entropy)
- Confidence calibration
- Epistemic uncertainty

This addresses the research critique: "You only get point estimates everywhere"
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import numpy as np
from collections import Counter
import math

from core.schemas import (
    Persona, EmailStimulus, PromptConfiguration, ActionType
)


class UncertaintyLevel(str, Enum):
    """Categorical uncertainty levels."""
    VERY_CONFIDENT = "very_confident"   # Entropy < 0.2
    CONFIDENT = "confident"              # Entropy 0.2-0.5
    MODERATE = "moderate"                # Entropy 0.5-0.8
    UNCERTAIN = "uncertain"              # Entropy 0.8-1.0
    HIGHLY_UNCERTAIN = "highly_uncertain"  # Entropy > 1.0


@dataclass
class ActionDistribution:
    """Distribution of LLM responses across actions."""
    click_prob: float = 0.0
    report_prob: float = 0.0
    ignore_prob: float = 0.0
    error_prob: float = 0.0  # Parse failures

    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy of distribution."""
        probs = [self.click_prob, self.report_prob, self.ignore_prob]
        probs = [p for p in probs if p > 0]  # Filter zeros
        if not probs:
            return 0.0
        return -sum(p * math.log2(p) for p in probs)

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution over 3 actions)."""
        return math.log2(3)  # ~1.585

    @property
    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1]."""
        return self.entropy / self.max_entropy if self.max_entropy > 0 else 0

    @property
    def modal_action(self) -> str:
        """Most frequent action."""
        actions = {
            'click': self.click_prob,
            'report': self.report_prob,
            'ignore': self.ignore_prob
        }
        return max(actions, key=actions.get)

    @property
    def modal_probability(self) -> float:
        """Probability of most frequent action."""
        return max(self.click_prob, self.report_prob, self.ignore_prob)

    @property
    def margin(self) -> float:
        """Margin between top two actions (higher = more confident)."""
        probs = sorted([self.click_prob, self.report_prob, self.ignore_prob], reverse=True)
        return probs[0] - probs[1] if len(probs) > 1 else probs[0]

    @property
    def uncertainty_level(self) -> UncertaintyLevel:
        """Categorize uncertainty level based on entropy."""
        ent = self.normalized_entropy
        if ent < 0.2:
            return UncertaintyLevel.VERY_CONFIDENT
        elif ent < 0.4:
            return UncertaintyLevel.CONFIDENT
        elif ent < 0.6:
            return UncertaintyLevel.MODERATE
        elif ent < 0.8:
            return UncertaintyLevel.UNCERTAIN
        else:
            return UncertaintyLevel.HIGHLY_UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            'click_prob': self.click_prob,
            'report_prob': self.report_prob,
            'ignore_prob': self.ignore_prob,
            'error_prob': self.error_prob,
            'entropy': self.entropy,
            'normalized_entropy': self.normalized_entropy,
            'modal_action': self.modal_action,
            'modal_probability': self.modal_probability,
            'margin': self.margin,
            'uncertainty_level': self.uncertainty_level.value
        }


@dataclass
class ConfidenceDistribution:
    """Distribution of LLM's self-reported confidence."""
    high_prob: float = 0.0
    medium_prob: float = 0.0
    low_prob: float = 0.0
    unknown_prob: float = 0.0  # When confidence not parsed

    @property
    def mean_confidence(self) -> float:
        """Mean confidence (HIGH=1, MEDIUM=0.5, LOW=0)."""
        if self.high_prob + self.medium_prob + self.low_prob == 0:
            return 0.5  # Default
        total = self.high_prob + self.medium_prob + self.low_prob
        return (self.high_prob * 1.0 + self.medium_prob * 0.5 + self.low_prob * 0.0) / total


@dataclass
class UncertaintySample:
    """Single sample from multi-sample uncertainty estimation."""
    sample_id: int
    action: str
    confidence: Optional[str]
    reasoning: Optional[str]
    raw_response: str


@dataclass
class UncertaintyEstimate:
    """Complete uncertainty estimate for a single trial."""
    trial_id: str
    email_id: str
    persona_id: str

    # Sample statistics
    n_samples: int = 0
    n_valid_samples: int = 0

    # Distributions
    action_distribution: ActionDistribution = field(default_factory=ActionDistribution)
    confidence_distribution: ConfidenceDistribution = field(default_factory=ConfidenceDistribution)

    # Derived metrics
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    margin: float = 0.0
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MODERATE

    # Final decision (majority vote)
    final_action: str = ""
    final_confidence: float = 0.0

    # Raw samples for analysis
    samples: List[UncertaintySample] = field(default_factory=list)

    # Comparison to human
    human_action: Optional[str] = None
    matches_human: bool = False

    # Calibration metrics
    confidence_calibration_gap: float = 0.0  # Difference between confidence and accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trial_id': self.trial_id,
            'email_id': self.email_id,
            'persona_id': self.persona_id,
            'n_samples': self.n_samples,
            'n_valid_samples': self.n_valid_samples,
            'action_distribution': self.action_distribution.to_dict(),
            'entropy': self.entropy,
            'normalized_entropy': self.normalized_entropy,
            'margin': self.margin,
            'uncertainty_level': self.uncertainty_level.value,
            'final_action': self.final_action,
            'final_confidence': self.final_confidence,
            'human_action': self.human_action,
            'matches_human': self.matches_human
        }


@dataclass
class UncertaintyAnalysisResult:
    """Aggregate uncertainty analysis across multiple trials."""
    persona_id: str
    persona_name: str
    model_id: str
    prompt_config: str

    # Trial-level results
    estimates: List[UncertaintyEstimate] = field(default_factory=list)

    # Aggregate metrics
    mean_entropy: float = 0.0
    mean_normalized_entropy: float = 0.0
    mean_margin: float = 0.0

    # Accuracy with uncertainty awareness
    majority_vote_accuracy: float = 0.0  # Accuracy using majority vote
    high_confidence_accuracy: float = 0.0  # Accuracy on high-confidence predictions
    low_confidence_accuracy: float = 0.0  # Accuracy on low-confidence predictions

    # Calibration
    calibration_error: float = 0.0  # Expected Calibration Error (ECE)
    overconfidence_rate: float = 0.0  # How often confidence > accuracy

    # Uncertainty distribution
    uncertainty_distribution: Dict[str, float] = field(default_factory=dict)

    # Metadata
    n_trials: int = 0
    samples_per_trial: int = 0
    total_samples: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'persona_id': self.persona_id,
            'persona_name': self.persona_name,
            'model_id': self.model_id,
            'prompt_config': self.prompt_config,
            'n_trials': self.n_trials,
            'samples_per_trial': self.samples_per_trial,
            'mean_entropy': self.mean_entropy,
            'mean_normalized_entropy': self.mean_normalized_entropy,
            'mean_margin': self.mean_margin,
            'majority_vote_accuracy': self.majority_vote_accuracy,
            'high_confidence_accuracy': self.high_confidence_accuracy,
            'low_confidence_accuracy': self.low_confidence_accuracy,
            'calibration_error': self.calibration_error,
            'overconfidence_rate': self.overconfidence_rate,
            'uncertainty_distribution': self.uncertainty_distribution
        }


class LLMUncertaintyQuantifier:
    """
    Quantifies uncertainty in LLM persona simulation.

    Instead of single-shot predictions, runs multiple samples
    to estimate the LLM's uncertainty about the persona's behavior.
    """

    def __init__(
        self,
        router,
        prompt_builder,
        n_samples: int = 10,
        temperature_range: Tuple[float, float] = (0.7, 1.0)
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            router: ProviderRouter for LLM calls
            prompt_builder: PromptBuilder instance
            n_samples: Number of samples per trial (default 10)
            temperature_range: Range of temperatures for sampling
        """
        self.router = router
        self.prompt_builder = prompt_builder
        self.n_samples = n_samples
        self.temp_min, self.temp_max = temperature_range

    async def estimate_uncertainty(
        self,
        persona: Persona,
        email: EmailStimulus,
        model_id: str,
        prompt_config: PromptConfiguration,
        human_action: Optional[str] = None,
        trial_id: str = "",
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for a single trial by running multiple samples.

        Args:
            persona: Persona to simulate
            email: Email stimulus
            model_id: Model to use
            prompt_config: Prompt configuration
            human_action: Ground truth human action (optional)
            trial_id: Trial identifier
            training_trials: Training data for ICL
            emails_dict: Email lookup for ICL

        Returns:
            UncertaintyEstimate with full uncertainty analysis
        """
        from phase2.providers.base import LLMRequest
        from phase2.simulation.prompt_builder import ResponseParser

        estimate = UncertaintyEstimate(
            trial_id=trial_id,
            email_id=email.email_id,
            persona_id=persona.persona_id,
            n_samples=self.n_samples,
            human_action=human_action
        )

        # Build base prompt
        built_prompt = self.prompt_builder.build(
            persona=persona,
            email=email,
            config=prompt_config,
            trial_number=0,
            training_trials=training_trials,
            emails_dict=emails_dict
        )

        # Collect samples with varying temperature
        action_counts = Counter()
        confidence_counts = Counter()
        samples = []

        for i in range(self.n_samples):
            # Vary temperature slightly to get diverse samples
            temp = built_prompt.temperature + np.random.uniform(-0.1, 0.1)
            temp = max(0.1, min(1.5, temp))  # Clamp to valid range

            try:
                request = LLMRequest(
                    system_prompt=built_prompt.system_prompt,
                    user_prompt=built_prompt.user_prompt,
                    temperature=temp,
                    top_p=built_prompt.top_p,
                    max_tokens=300
                )

                response = await self.router.complete(model_id, request)

                if response.success:
                    parsed = ResponseParser.parse(
                        response.content,
                        built_prompt.expected_format
                    )

                    action = parsed['action'].value if parsed['action'] else 'error'
                    confidence = str(parsed.get('confidence', 'unknown'))

                    action_counts[action] += 1
                    confidence_counts[confidence.lower()] += 1

                    samples.append(UncertaintySample(
                        sample_id=i,
                        action=action,
                        confidence=confidence,
                        reasoning=parsed.get('reasoning', ''),
                        raw_response=response.content[:500]
                    ))
                else:
                    action_counts['error'] += 1

            except Exception as e:
                action_counts['error'] += 1

        # Calculate distributions
        total_valid = sum(v for k, v in action_counts.items() if k != 'error')
        total_all = sum(action_counts.values())

        if total_valid > 0:
            estimate.action_distribution = ActionDistribution(
                click_prob=action_counts.get('click', 0) / total_valid,
                report_prob=action_counts.get('report', 0) / total_valid,
                ignore_prob=action_counts.get('ignore', 0) / total_valid,
                error_prob=action_counts.get('error', 0) / total_all if total_all > 0 else 0
            )

        total_conf = sum(v for k, v in confidence_counts.items() if k != 'unknown')
        if total_conf > 0:
            estimate.confidence_distribution = ConfidenceDistribution(
                high_prob=confidence_counts.get('high', 0) / total_conf,
                medium_prob=confidence_counts.get('medium', 0) / total_conf,
                low_prob=confidence_counts.get('low', 0) / total_conf,
                unknown_prob=confidence_counts.get('unknown', 0) / total_all if total_all > 0 else 0
            )

        # Populate derived metrics
        estimate.n_valid_samples = total_valid
        estimate.entropy = estimate.action_distribution.entropy
        estimate.normalized_entropy = estimate.action_distribution.normalized_entropy
        estimate.margin = estimate.action_distribution.margin
        estimate.uncertainty_level = estimate.action_distribution.uncertainty_level

        # Final decision via majority vote
        estimate.final_action = estimate.action_distribution.modal_action
        estimate.final_confidence = estimate.action_distribution.modal_probability

        # Compare to human
        if human_action:
            estimate.matches_human = (
                estimate.final_action.lower() == human_action.lower()
            )

            # Calibration gap: difference between confidence and actual accuracy
            # If we're 80% confident but wrong, calibration gap is high
            expected_accuracy = estimate.final_confidence
            actual_accuracy = 1.0 if estimate.matches_human else 0.0
            estimate.confidence_calibration_gap = abs(expected_accuracy - actual_accuracy)

        estimate.samples = samples

        return estimate

    async def run_uncertainty_analysis(
        self,
        persona: Persona,
        test_trials: List,
        model_id: str,
        prompt_config: PromptConfiguration,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        stop_check: callable = None,
        progress_callback: callable = None
    ) -> UncertaintyAnalysisResult:
        """
        Run uncertainty analysis across multiple trials.

        Args:
            persona: Persona to analyze
            test_trials: Test trials with human ground truth
            model_id: Model to use
            prompt_config: Prompt configuration
            training_trials: Training data for ICL
            emails_dict: Email lookup for ICL
            stop_check: Stop check callback
            progress_callback: Progress callback

        Returns:
            UncertaintyAnalysisResult with aggregate metrics
        """
        result = UncertaintyAnalysisResult(
            persona_id=persona.persona_id,
            persona_name=persona.name,
            model_id=model_id,
            prompt_config=prompt_config.value,
            samples_per_trial=self.n_samples,
            started_at=datetime.now().isoformat()
        )

        print(f"\n{'='*60}")
        print(f"UNCERTAINTY ANALYSIS: {persona.name}")
        print(f"Running {self.n_samples} samples per trial x {len(test_trials)} trials")
        print(f"{'='*60}\n")

        for idx, trial in enumerate(test_trials):
            if stop_check and stop_check():
                print(f"\n⛔ Stopped at trial {idx + 1}/{len(test_trials)}")
                break

            if progress_callback:
                progress_callback(idx, len(test_trials))

            # Build email stimulus
            email = EmailStimulus(
                email_id=trial.email_id,
                email_type=trial.email_type,
                sender_familiarity=trial.sender_familiarity,
                urgency_level=trial.urgency_level,
                framing_type=trial.framing_type,
                ground_truth=1 if trial.email_type == 'phishing' else 0,
                subject_line=trial.subject,
                sender_display=trial.sender,
                body_text=trial.body
            )

            estimate = await self.estimate_uncertainty(
                persona=persona,
                email=email,
                model_id=model_id,
                prompt_config=prompt_config,
                human_action=trial.human_action,
                trial_id=trial.trial_id,
                training_trials=training_trials,
                emails_dict=emails_dict
            )

            result.estimates.append(estimate)

            # Log progress
            match_sym = "✅" if estimate.matches_human else "❌"
            print(f"  Trial {idx+1}: {estimate.final_action} "
                  f"(conf: {estimate.final_confidence:.0%}, "
                  f"entropy: {estimate.normalized_entropy:.2f}) {match_sym}")

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(result)

        result.n_trials = len(result.estimates)
        result.total_samples = result.n_trials * self.n_samples
        result.completed_at = datetime.now().isoformat()

        return result

    def _calculate_aggregate_metrics(self, result: UncertaintyAnalysisResult):
        """Calculate aggregate metrics from all estimates."""
        estimates = result.estimates
        if not estimates:
            return

        # Mean entropy/margin
        result.mean_entropy = np.mean([e.entropy for e in estimates])
        result.mean_normalized_entropy = np.mean([e.normalized_entropy for e in estimates])
        result.mean_margin = np.mean([e.margin for e in estimates])

        # Accuracy metrics
        n_correct = sum(1 for e in estimates if e.matches_human)
        result.majority_vote_accuracy = n_correct / len(estimates)

        # Split by confidence level
        high_conf_estimates = [e for e in estimates
                              if e.uncertainty_level in [UncertaintyLevel.VERY_CONFIDENT,
                                                        UncertaintyLevel.CONFIDENT]]
        low_conf_estimates = [e for e in estimates
                             if e.uncertainty_level in [UncertaintyLevel.UNCERTAIN,
                                                       UncertaintyLevel.HIGHLY_UNCERTAIN]]

        if high_conf_estimates:
            result.high_confidence_accuracy = (
                sum(1 for e in high_conf_estimates if e.matches_human) / len(high_conf_estimates)
            )

        if low_conf_estimates:
            result.low_confidence_accuracy = (
                sum(1 for e in low_conf_estimates if e.matches_human) / len(low_conf_estimates)
            )

        # Calibration metrics
        # Expected Calibration Error (ECE)
        calibration_gaps = [e.confidence_calibration_gap for e in estimates if e.human_action]
        if calibration_gaps:
            result.calibration_error = np.mean(calibration_gaps)

        # Overconfidence rate
        overconfident = sum(1 for e in estimates
                          if e.final_confidence > 0.7 and not e.matches_human)
        confident_predictions = sum(1 for e in estimates if e.final_confidence > 0.7)
        if confident_predictions > 0:
            result.overconfidence_rate = overconfident / confident_predictions

        # Uncertainty distribution
        level_counts = Counter(e.uncertainty_level.value for e in estimates)
        total = len(estimates)
        result.uncertainty_distribution = {
            level: count / total for level, count in level_counts.items()
        }

    def get_analysis_summary(self, result: UncertaintyAnalysisResult) -> str:
        """Generate human-readable summary of uncertainty analysis."""
        lines = [
            f"\n{'='*60}",
            f"UNCERTAINTY ANALYSIS SUMMARY: {result.persona_name}",
            f"{'='*60}",
            "",
            f"Trials: {result.n_trials} x {result.samples_per_trial} samples = {result.total_samples} total",
            "",
            "UNCERTAINTY METRICS:",
            "-" * 40,
            f"  Mean Entropy: {result.mean_entropy:.3f} (normalized: {result.mean_normalized_entropy:.2%})",
            f"  Mean Margin: {result.mean_margin:.2%}",
            "",
            "ACCURACY BY CONFIDENCE:",
            "-" * 40,
            f"  Overall (majority vote): {result.majority_vote_accuracy:.1%}",
            f"  High-confidence trials:  {result.high_confidence_accuracy:.1%}",
            f"  Low-confidence trials:   {result.low_confidence_accuracy:.1%}",
            "",
            "CALIBRATION:",
            "-" * 40,
            f"  Expected Calibration Error: {result.calibration_error:.3f}",
            f"  Overconfidence Rate: {result.overconfidence_rate:.1%}",
            "",
            "UNCERTAINTY DISTRIBUTION:",
            "-" * 40
        ]

        for level, pct in sorted(result.uncertainty_distribution.items()):
            lines.append(f"  {level:20s}: {pct:.1%}")

        lines.append(f"\n{'='*60}")

        return "\n".join(lines)


class UncertaintyAwarePredictor:
    """
    Makes predictions with uncertainty awareness.

    Can abstain from prediction when uncertainty is too high,
    or provide confidence intervals around predictions.
    """

    def __init__(
        self,
        quantifier: LLMUncertaintyQuantifier,
        abstention_threshold: float = 0.6  # Normalized entropy threshold
    ):
        """
        Initialize uncertainty-aware predictor.

        Args:
            quantifier: LLMUncertaintyQuantifier instance
            abstention_threshold: Entropy threshold above which to abstain
        """
        self.quantifier = quantifier
        self.abstention_threshold = abstention_threshold

    async def predict_with_uncertainty(
        self,
        persona: Persona,
        email: EmailStimulus,
        model_id: str,
        prompt_config: PromptConfiguration,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with uncertainty estimate.

        Returns dict with:
        - action: Predicted action (or "ABSTAIN" if too uncertain)
        - confidence: Confidence in prediction [0, 1]
        - distribution: Full action distribution
        - should_abstain: Whether to abstain due to high uncertainty
        - uncertainty_reason: Why uncertainty is high (if applicable)
        """
        estimate = await self.quantifier.estimate_uncertainty(
            persona=persona,
            email=email,
            model_id=model_id,
            prompt_config=prompt_config,
            training_trials=training_trials,
            emails_dict=emails_dict
        )

        should_abstain = estimate.normalized_entropy > self.abstention_threshold

        uncertainty_reason = ""
        if should_abstain:
            # Analyze why uncertain
            dist = estimate.action_distribution
            if dist.margin < 0.2:
                uncertainty_reason = f"Close decision between {dist.modal_action} and alternatives"
            elif dist.error_prob > 0.2:
                uncertainty_reason = "Many parsing errors - unreliable responses"
            else:
                uncertainty_reason = "High entropy - no clear dominant action"

        return {
            'action': "ABSTAIN" if should_abstain else estimate.final_action,
            'confidence': estimate.final_confidence,
            'distribution': estimate.action_distribution.to_dict(),
            'entropy': estimate.entropy,
            'normalized_entropy': estimate.normalized_entropy,
            'uncertainty_level': estimate.uncertainty_level.value,
            'should_abstain': should_abstain,
            'uncertainty_reason': uncertainty_reason,
            'n_samples': estimate.n_samples,
            'samples_breakdown': {
                'click': int(estimate.action_distribution.click_prob * estimate.n_valid_samples),
                'report': int(estimate.action_distribution.report_prob * estimate.n_valid_samples),
                'ignore': int(estimate.action_distribution.ignore_prob * estimate.n_valid_samples)
            }
        }


# Convenience functions

async def quick_uncertainty_check(
    persona: Persona,
    email: EmailStimulus,
    model_id: str,
    router,
    prompt_builder,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Quick uncertainty check for a single email.

    Returns action distribution and uncertainty metrics.
    """
    quantifier = LLMUncertaintyQuantifier(
        router=router,
        prompt_builder=prompt_builder,
        n_samples=n_samples
    )

    estimate = await quantifier.estimate_uncertainty(
        persona=persona,
        email=email,
        model_id=model_id,
        prompt_config=PromptConfiguration.COT
    )

    return estimate.to_dict()
