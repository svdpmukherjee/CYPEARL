"""
CYPEARL Phase 2 - Prompt Ablation Study

Implements factorial design to systematically test which prompt components
actually contribute to fidelity. Instead of assuming incremental is best,
we test ALL combinations to find optimal prompt structure.

Factorial Design (2^3 = 8 combinations):
- T = Traits (29 psychological traits)
- S = Stats (8 behavioral outcomes)
- C = CoT (reasoning examples/chain-of-thought)

This provides scientific ablation study showing which components matter.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import itertools
import numpy as np
from scipy import stats as scipy_stats

from core.schemas import (
    Persona, EmailStimulus, PromptConfiguration,
    ActionType, BehavioralStatistics, EmailInteractionEffects
)


class PromptComponent(str, Enum):
    """Individual prompt components that can be toggled."""
    TRAITS = "traits"          # 29 psychological traits
    STATS = "stats"            # 8 behavioral outcomes
    COT = "cot"                # Chain-of-thought reasoning examples
    ICL = "icl"                # In-context learning examples


@dataclass
class ComponentConfig:
    """Configuration for which components to include in prompt."""
    include_traits: bool = True
    include_stats: bool = False
    include_cot: bool = False
    include_icl: bool = True

    @property
    def name(self) -> str:
        """Generate descriptive name for this configuration."""
        parts = []
        if self.include_traits:
            parts.append("T")
        if self.include_stats:
            parts.append("S")
        if self.include_cot:
            parts.append("C")
        if self.include_icl:
            parts.append("+ICL")
        return "_".join(parts) if parts else "MINIMAL"

    @property
    def complexity_score(self) -> int:
        """Score representing prompt complexity (0-4)."""
        return sum([
            self.include_traits,
            self.include_stats,
            self.include_cot,
            self.include_icl
        ])

    def to_dict(self) -> Dict[str, bool]:
        return {
            "include_traits": self.include_traits,
            "include_stats": self.include_stats,
            "include_cot": self.include_cot,
            "include_icl": self.include_icl
        }


@dataclass
class AblationTrialResult:
    """Result of a single trial in ablation study."""
    trial_id: str
    email_id: str
    human_action: str
    llm_action: str
    is_correct: bool
    email_type: str
    urgency_level: str
    sender_familiarity: str
    llm_confidence: Optional[str] = None
    llm_reasoning: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class AblationConfigResult:
    """Result for a single prompt configuration in ablation study."""
    config: ComponentConfig
    config_name: str

    # Core metrics
    accuracy: float = 0.0
    n_trials: int = 0
    n_correct: int = 0

    # Detailed metrics
    click_rate_error: float = 0.0  # |AI_click_rate - human_click_rate|
    report_rate_error: float = 0.0

    # Effect preservation
    urgency_effect_preserved: float = 0.0  # Correlation of urgency effect
    familiarity_effect_preserved: float = 0.0

    # Breakdown by email type
    phishing_accuracy: float = 0.0
    legitimate_accuracy: float = 0.0

    # Cost metrics
    avg_prompt_tokens: int = 0
    avg_response_tokens: int = 0
    total_cost_usd: float = 0.0

    # Trial details
    trials: List[AblationTrialResult] = field(default_factory=list)

    # Metadata
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class AblationStudyResult:
    """Complete result of factorial ablation study."""
    persona_id: str
    persona_name: str
    model_id: str

    # All configuration results
    config_results: Dict[str, AblationConfigResult] = field(default_factory=dict)

    # Analysis results
    best_config: Optional[str] = None
    best_accuracy: float = 0.0

    # Component importance scores (from ablation analysis)
    component_importance: Dict[str, float] = field(default_factory=dict)

    # Statistical tests
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommended_config: Optional[str] = None
    recommendation_reason: str = ""

    # Metadata
    total_trials: int = 0
    total_cost_usd: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class FactorialPromptBuilder:
    """
    Builds prompts with configurable components for ablation study.

    Unlike the standard PromptBuilder which uses fixed incremental configs,
    this allows any combination of components to be toggled on/off.
    """

    def __init__(self, base_prompt_builder):
        """
        Initialize with reference to base prompt builder.

        Args:
            base_prompt_builder: The standard PromptBuilder instance
        """
        self.base_builder = base_prompt_builder

    def build(
        self,
        persona: Persona,
        email: EmailStimulus,
        config: ComponentConfig,
        trial_number: int = 0,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None
    ) -> 'BuiltPrompt':
        """
        Build prompt with specified component configuration.

        Args:
            persona: The persona to simulate
            email: The email stimulus
            config: Which components to include
            trial_number: Trial number for context variation
            training_trials: Training data for ICL
            emails_dict: Email lookup for ICL

        Returns:
            BuiltPrompt ready for LLM
        """
        from phase2.simulation.prompt_builder import BuiltPrompt

        # Determine cognitive style
        is_impulsive = self.base_builder._is_impulsive(persona)

        # Build prompt sections based on config
        sections = []

        # Always include minimal frame
        sections.append(self._build_minimal_frame(persona))

        # Conditionally add components
        if config.include_traits:
            sections.append(self._build_traits_section(persona))

        if config.include_stats:
            sections.append(self._build_stats_section(persona))

        if config.include_cot:
            sections.append(self._build_cot_section(
                persona, training_trials, emails_dict, is_impulsive
            ))
        elif config.include_icl and training_trials and emails_dict:
            # ICL without CoT (simpler examples)
            sections.append(self._build_icl_section(
                persona, training_trials, emails_dict, "minimal"
            ))

        # Add situational context for variability
        sections.append(self.base_builder._generate_situational_context(
            persona, trial_number
        ))

        # Add response format based on complexity
        if config.include_cot:
            response_format = self.base_builder.RESPONSE_FORMAT_COT
            expected_format = "cot"
        elif config.include_stats:
            response_format = self.base_builder.RESPONSE_FORMAT_STATS
            expected_format = "structured"
        else:
            response_format = self.base_builder.RESPONSE_FORMAT_BASELINE
            expected_format = "single_action"

        sections.append(response_format)

        system_prompt = "\n\n".join(sections)
        user_prompt = self.base_builder._build_email_prompt(email, is_impulsive)

        return BuiltPrompt(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt,
            config=PromptConfiguration.BASELINE,  # Placeholder
            persona_id=persona.persona_id,
            email_id=email.email_id,
            expected_format=expected_format,
            action_keywords=self.base_builder.ACTION_KEYWORDS,
            temperature=self.base_builder._get_temperature(persona),
            top_p=0.95
        )

    def _build_minimal_frame(self, persona: Persona) -> str:
        """Build minimal roleplaying frame without traits."""
        return f"""You are experiencing life as "{persona.name}" for a moment.

WHO YOU ARE:
{persona.description}

Respond to emails as this person would - not as an AI, but as THIS specific person."""

    def _build_traits_section(self, persona: Persona) -> str:
        """Build the 29 traits section."""
        traits_text = self.base_builder._format_traits_simple_language(persona)
        return f"""YOUR CHARACTERISTICS (psychological and cognitive profile):
{traits_text}"""

    def _build_stats_section(self, persona: Persona) -> str:
        """Build the 8 behavioral outcomes section."""
        return self.base_builder._build_behavioral_outcomes_block(persona)

    def _build_cot_section(
        self,
        persona: Persona,
        training_trials: List[Dict],
        emails_dict: Dict[str, Dict],
        is_impulsive: bool
    ) -> str:
        """Build chain-of-thought reasoning section."""
        return self.base_builder._build_cot_reasoning_block(
            persona, training_trials, emails_dict, is_impulsive
        )

    def _build_icl_section(
        self,
        persona: Persona,
        training_trials: List[Dict],
        emails_dict: Dict[str, Dict],
        style: str
    ) -> str:
        """Build ICL examples section."""
        return self.base_builder._build_icl_block(
            persona, training_trials, emails_dict, style
        )


class PromptAblationStudy:
    """
    Runs factorial ablation study to determine which prompt components matter.

    Tests all 2^3 = 8 combinations of (Traits, Stats, CoT) systematically
    and provides statistical analysis of component importance.
    """

    def __init__(self, router, prompt_builder):
        """
        Initialize ablation study.

        Args:
            router: ProviderRouter for making LLM calls
            prompt_builder: Standard PromptBuilder instance
        """
        self.router = router
        self.factorial_builder = FactorialPromptBuilder(prompt_builder)
        self.base_builder = prompt_builder

    def generate_factorial_configs(
        self,
        include_icl: bool = True
    ) -> List[ComponentConfig]:
        """
        Generate all 2^3 = 8 factorial combinations.

        Args:
            include_icl: Whether to include ICL in all configs

        Returns:
            List of 8 ComponentConfig objects
        """
        configs = []

        # Generate all combinations of (traits, stats, cot)
        for traits, stats, cot in itertools.product([False, True], repeat=3):
            configs.append(ComponentConfig(
                include_traits=traits,
                include_stats=stats,
                include_cot=cot,
                include_icl=include_icl
            ))

        # Sort by complexity for readable output
        configs.sort(key=lambda c: c.complexity_score)

        return configs

    async def run_ablation_study(
        self,
        persona: Persona,
        test_trials: List,
        model_id: str,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        include_icl: bool = True,
        stop_check: callable = None,
        progress_callback: callable = None
    ) -> AblationStudyResult:
        """
        Run complete factorial ablation study.

        Tests all 8 configurations and analyzes which components matter.

        Args:
            persona: The persona to test
            test_trials: Held-out test trials (from DataSplitter)
            model_id: Which model to use
            training_trials: Training data for ICL (optional)
            emails_dict: Email lookup for ICL (optional)
            include_icl: Whether to use ICL in all configs
            stop_check: Optional callable to check if study should stop
            progress_callback: Optional progress callback(completed, total)

        Returns:
            AblationStudyResult with all analysis
        """
        result = AblationStudyResult(
            persona_id=persona.persona_id,
            persona_name=persona.name,
            model_id=model_id,
            started_at=datetime.now().isoformat()
        )

        configs = self.generate_factorial_configs(include_icl)
        total_configs = len(configs)
        total_trials = len(test_trials) * total_configs
        completed_trials = 0

        print(f"\n{'='*60}")
        print(f"FACTORIAL ABLATION STUDY: {persona.name}")
        print(f"Testing {total_configs} configurations x {len(test_trials)} trials = {total_trials} total")
        print(f"{'='*60}\n")

        for config_idx, config in enumerate(configs):
            if stop_check and stop_check():
                print(f"\n⛔ Ablation study stopped at config {config_idx + 1}/{total_configs}")
                break

            print(f"\n📋 Testing config {config_idx + 1}/{total_configs}: {config.name}")
            print(f"   Components: Traits={config.include_traits}, Stats={config.include_stats}, CoT={config.include_cot}")

            config_result = await self._run_config_trials(
                persona=persona,
                test_trials=test_trials,
                config=config,
                model_id=model_id,
                training_trials=training_trials,
                emails_dict=emails_dict,
                stop_check=stop_check,
                progress_callback=lambda c, t: progress_callback(
                    completed_trials + c, total_trials
                ) if progress_callback else None
            )

            result.config_results[config.name] = config_result
            result.total_cost_usd += config_result.total_cost_usd
            completed_trials += config_result.n_trials

            print(f"   ✅ Accuracy: {config_result.accuracy:.1%} ({config_result.n_correct}/{config_result.n_trials})")

        # Analyze results
        self._analyze_results(result)

        result.total_trials = completed_trials
        result.completed_at = datetime.now().isoformat()

        return result

    async def _run_config_trials(
        self,
        persona: Persona,
        test_trials: List,
        config: ComponentConfig,
        model_id: str,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        stop_check: callable = None,
        progress_callback: callable = None
    ) -> AblationConfigResult:
        """Run all trials for a single configuration."""
        from phase2.providers.base import LLMRequest
        from phase2.simulation.prompt_builder import ResponseParser

        config_result = AblationConfigResult(
            config=config,
            config_name=config.name,
            started_at=datetime.now().isoformat()
        )

        for idx, test_trial in enumerate(test_trials):
            if stop_check and stop_check():
                break

            if progress_callback:
                progress_callback(idx, len(test_trials))

            # Build email stimulus
            email = EmailStimulus(
                email_id=test_trial.email_id,
                email_type=test_trial.email_type,
                sender_familiarity=test_trial.sender_familiarity,
                urgency_level=test_trial.urgency_level,
                framing_type=test_trial.framing_type,
                ground_truth=1 if test_trial.email_type == 'phishing' else 0,
                subject_line=test_trial.subject,
                sender_display=test_trial.sender,
                body_text=test_trial.body
            )

            # Build prompt with this config
            built_prompt = self.factorial_builder.build(
                persona=persona,
                email=email,
                config=config,
                trial_number=idx,
                training_trials=training_trials,
                emails_dict=emails_dict
            )

            trial_result = AblationTrialResult(
                trial_id=test_trial.trial_id,
                email_id=test_trial.email_id,
                human_action=test_trial.human_action,
                llm_action="",
                is_correct=False,
                email_type=test_trial.email_type,
                urgency_level=test_trial.urgency_level,
                sender_familiarity=test_trial.sender_familiarity
            )

            try:
                request = LLMRequest(
                    system_prompt=built_prompt.system_prompt,
                    user_prompt=built_prompt.user_prompt,
                    temperature=built_prompt.temperature,
                    top_p=built_prompt.top_p,
                    max_tokens=300
                )

                response = await self.router.complete(model_id, request)

                if response.success:
                    parsed = ResponseParser.parse(
                        response.content,
                        built_prompt.expected_format
                    )

                    trial_result.llm_action = parsed['action'].value if parsed['action'] else 'error'
                    trial_result.llm_confidence = str(parsed.get('confidence', ''))
                    trial_result.llm_reasoning = parsed.get('reasoning', '')
                    trial_result.is_correct = (
                        trial_result.llm_action.lower() == trial_result.human_action.lower()
                    )
                else:
                    trial_result.llm_action = 'error'

            except Exception as e:
                trial_result.llm_action = 'error'

            config_result.trials.append(trial_result)
            if trial_result.is_correct:
                config_result.n_correct += 1

        # Calculate metrics
        config_result.n_trials = len(config_result.trials)
        if config_result.n_trials > 0:
            config_result.accuracy = config_result.n_correct / config_result.n_trials
            self._calculate_config_metrics(config_result, persona)

        config_result.completed_at = datetime.now().isoformat()
        return config_result

    def _calculate_config_metrics(
        self,
        config_result: AblationConfigResult,
        persona: Persona
    ):
        """Calculate detailed metrics for a configuration result."""
        trials = config_result.trials
        n = len(trials)
        if n == 0:
            return

        # Click rate comparison
        human_clicks = sum(1 for t in trials if t.human_action == 'click')
        llm_clicks = sum(1 for t in trials if t.llm_action == 'click')
        human_click_rate = human_clicks / n
        llm_click_rate = llm_clicks / n
        config_result.click_rate_error = abs(llm_click_rate - human_click_rate)

        # Report rate comparison
        human_reports = sum(1 for t in trials if t.human_action == 'report')
        llm_reports = sum(1 for t in trials if t.llm_action == 'report')
        human_report_rate = human_reports / n
        llm_report_rate = llm_reports / n
        config_result.report_rate_error = abs(llm_report_rate - human_report_rate)

        # Accuracy by email type
        phishing_trials = [t for t in trials if t.email_type == 'phishing']
        if phishing_trials:
            config_result.phishing_accuracy = (
                sum(1 for t in phishing_trials if t.is_correct) / len(phishing_trials)
            )

        legit_trials = [t for t in trials if t.email_type == 'legitimate']
        if legit_trials:
            config_result.legitimate_accuracy = (
                sum(1 for t in legit_trials if t.is_correct) / len(legit_trials)
            )

        # Effect preservation (simplified - would need more trials for full analysis)
        # Check if urgency effect is preserved
        high_urgency = [t for t in trials if t.urgency_level == 'high']
        low_urgency = [t for t in trials if t.urgency_level == 'low']

        if high_urgency and low_urgency:
            human_urgency_effect = (
                sum(1 for t in high_urgency if t.human_action == 'click') / len(high_urgency) -
                sum(1 for t in low_urgency if t.human_action == 'click') / len(low_urgency)
            )
            llm_urgency_effect = (
                sum(1 for t in high_urgency if t.llm_action == 'click') / len(high_urgency) -
                sum(1 for t in low_urgency if t.llm_action == 'click') / len(low_urgency)
            )

            if abs(human_urgency_effect) > 0.01:
                config_result.urgency_effect_preserved = 1 - abs(
                    llm_urgency_effect - human_urgency_effect
                ) / abs(human_urgency_effect)

    def _analyze_results(self, result: AblationStudyResult):
        """
        Analyze ablation study results to determine component importance.

        Uses ANOVA-style analysis to estimate main effects of each component.
        """
        configs = result.config_results
        if not configs:
            return

        # Find best configuration
        best_config = max(configs.items(), key=lambda x: x[1].accuracy)
        result.best_config = best_config[0]
        result.best_accuracy = best_config[1].accuracy

        # Calculate component importance using factorial analysis
        # Main effect of component X = mean(with X) - mean(without X)

        importance = {}

        # Traits effect
        with_traits = [c.accuracy for name, c in configs.items()
                      if c.config.include_traits]
        without_traits = [c.accuracy for name, c in configs.items()
                        if not c.config.include_traits]
        if with_traits and without_traits:
            importance['traits'] = np.mean(with_traits) - np.mean(without_traits)

        # Stats effect
        with_stats = [c.accuracy for name, c in configs.items()
                     if c.config.include_stats]
        without_stats = [c.accuracy for name, c in configs.items()
                        if not c.config.include_stats]
        if with_stats and without_stats:
            importance['stats'] = np.mean(with_stats) - np.mean(without_stats)

        # CoT effect
        with_cot = [c.accuracy for name, c in configs.items()
                   if c.config.include_cot]
        without_cot = [c.accuracy for name, c in configs.items()
                      if not c.config.include_cot]
        if with_cot and without_cot:
            importance['cot'] = np.mean(with_cot) - np.mean(without_cot)

        result.component_importance = importance

        # Statistical tests (t-tests for each component)
        tests = {}

        if with_traits and without_traits and len(with_traits) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(with_traits, without_traits)
            tests['traits'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

        if with_stats and without_stats and len(with_stats) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(with_stats, without_stats)
            tests['stats'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

        if with_cot and without_cot and len(with_cot) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(with_cot, without_cot)
            tests['cot'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

        result.statistical_tests = tests

        # Generate recommendation
        self._generate_recommendation(result)

    def _generate_recommendation(self, result: AblationStudyResult):
        """Generate recommendation based on ablation analysis."""
        importance = result.component_importance
        tests = result.statistical_tests

        # Find components that significantly improve accuracy
        significant_components = []
        for comp, test_result in tests.items():
            if test_result.get('significant', False) and importance.get(comp, 0) > 0:
                significant_components.append((comp, importance[comp]))

        # Sort by importance
        significant_components.sort(key=lambda x: x[1], reverse=True)

        if not significant_components:
            # No significant effects - recommend simplest
            result.recommended_config = "T" if importance.get('traits', 0) > 0 else "MINIMAL"
            result.recommendation_reason = (
                "No component showed statistically significant improvement. "
                "Recommend simplest configuration to minimize cost."
            )
        else:
            # Build config from significant components
            config_parts = []
            reasons = []
            for comp, imp in significant_components:
                if comp == 'traits':
                    config_parts.append('T')
                    reasons.append(f"Traits improve accuracy by {imp:.1%}")
                elif comp == 'stats':
                    config_parts.append('S')
                    reasons.append(f"Stats improve accuracy by {imp:.1%}")
                elif comp == 'cot':
                    config_parts.append('C')
                    reasons.append(f"CoT improves accuracy by {imp:.1%}")

            result.recommended_config = "_".join(config_parts) if config_parts else "MINIMAL"
            result.recommendation_reason = "; ".join(reasons)

    def get_study_summary(self, result: AblationStudyResult) -> str:
        """Generate human-readable summary of ablation study."""
        lines = [
            f"\n{'='*60}",
            f"ABLATION STUDY SUMMARY: {result.persona_name}",
            f"{'='*60}",
            "",
            "CONFIGURATION RESULTS:",
            "-" * 40
        ]

        # Sort by accuracy
        sorted_configs = sorted(
            result.config_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )

        for name, config_result in sorted_configs:
            marker = "🏆" if name == result.best_config else "  "
            lines.append(
                f"{marker} {name:20s}: {config_result.accuracy:6.1%} accuracy "
                f"(click_err: {config_result.click_rate_error:.1%})"
            )

        lines.extend([
            "",
            "COMPONENT IMPORTANCE:",
            "-" * 40
        ])

        for comp, imp in sorted(
            result.component_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            direction = "+" if imp > 0 else ""
            sig = "**" if result.statistical_tests.get(comp, {}).get('significant') else ""
            lines.append(f"  {comp.upper():10s}: {direction}{imp:.1%} {sig}")

        lines.extend([
            "",
            "RECOMMENDATION:",
            "-" * 40,
            f"  Config: {result.recommended_config}",
            f"  Reason: {result.recommendation_reason}",
            "",
            f"Total trials: {result.total_trials}",
            f"Total cost: ${result.total_cost_usd:.4f}",
            f"{'='*60}"
        ])

        return "\n".join(lines)


# Convenience function to run quick ablation
async def run_quick_ablation(
    persona: Persona,
    test_trials: List,
    model_id: str,
    router,
    prompt_builder,
    training_trials: List[Dict] = None,
    emails_dict: Dict[str, Dict] = None
) -> AblationStudyResult:
    """
    Quick ablation study runner.

    Args:
        persona: Persona to test
        test_trials: Held-out test data
        model_id: Model to use
        router: LLM router
        prompt_builder: Standard prompt builder
        training_trials: Training data for ICL
        emails_dict: Email lookup

    Returns:
        AblationStudyResult with full analysis
    """
    study = PromptAblationStudy(router, prompt_builder)
    result = await study.run_ablation_study(
        persona=persona,
        test_trials=test_trials,
        model_id=model_id,
        training_trials=training_trials,
        emails_dict=emails_dict
    )
    print(study.get_study_summary(result))
    return result
