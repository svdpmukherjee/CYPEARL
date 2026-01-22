"""
CYPEARL Phase 2 - Calibration Engine

Runs held-out validation trials to test if LLM prompts accurately
represent persona behavior before full benchmarking.

Enhanced with ICL (In-Context Learning) support for improved fidelity.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

from .data_splitter import SplitResult, HeldOutTrial, DataSplitter
from core.schemas import (
    Persona, EmailStimulus, PromptConfiguration,
    ActionType, BehavioralStatistics, EmailInteractionEffects
)


class CalibrationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CalibrationTrial:
    """Result of a single calibration trial."""
    trial_id: str
    email_id: str

    # Ground truth
    human_action: str
    human_confidence: Optional[str] = None

    # LLM prediction
    llm_action: str = ""
    llm_confidence: Optional[str] = None
    llm_reasoning: Optional[str] = None

    # Match result
    is_correct: bool = False

    # Email context (for analysis)
    email_type: str = ""
    urgency_level: str = ""
    sender_familiarity: str = ""

    # Debug
    prompt_used: str = ""
    raw_response: str = ""
    error_message: Optional[str] = None


@dataclass
class CalibrationResult:
    """Result of calibration for a persona-model-prompt combination."""
    persona_id: str
    persona_name: str
    model_id: str
    prompt_config: str

    # Core metrics
    accuracy: float = 0.0  # % of correct predictions
    n_trials: int = 0
    n_correct: int = 0

    # Breakdown by action
    click_precision: float = 0.0  # When LLM says click, how often human clicked
    click_recall: float = 0.0  # When human clicked, how often LLM said click
    report_precision: float = 0.0
    report_recall: float = 0.0

    # Rate comparison
    human_click_rate: float = 0.0
    llm_click_rate: float = 0.0
    click_rate_error: float = 0.0

    human_report_rate: float = 0.0
    llm_report_rate: float = 0.0
    report_rate_error: float = 0.0

    # Factor-specific accuracy
    phishing_accuracy: float = 0.0
    legitimate_accuracy: float = 0.0
    high_urgency_accuracy: float = 0.0
    low_urgency_accuracy: float = 0.0

    # Trial details
    trials: List[CalibrationTrial] = field(default_factory=list)

    # Failures for analysis
    failed_trials: List[CalibrationTrial] = field(default_factory=list)

    # Metadata
    status: CalibrationStatus = CalibrationStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    # Cost
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # NEW: Store prompts used for logging
    prompts_used: List[Dict[str, Any]] = field(default_factory=list)
    emails_used: List[Dict[str, Any]] = field(default_factory=list)

    def meets_threshold(self, threshold: float = 0.80) -> bool:
        """Check if calibration accuracy meets threshold."""
        return self.accuracy >= threshold

    def get_failure_summary(self) -> Dict[str, Any]:
        """Summarize where predictions failed."""
        if not self.failed_trials:
            return {"no_failures": True}

        failures_by_type = {}
        for t in self.failed_trials:
            key = f"{t.human_action}_as_{t.llm_action}"
            if key not in failures_by_type:
                failures_by_type[key] = []
            failures_by_type[key].append({
                "email_type": t.email_type,
                "urgency": t.urgency_level,
                "familiarity": t.sender_familiarity,
                "human_action": t.human_action,
                "llm_action": t.llm_action,
                "llm_reasoning": t.llm_reasoning
            })

        return {
            "total_failures": len(self.failed_trials),
            "by_confusion": failures_by_type,
            "click_as_other": len([t for t in self.failed_trials if t.human_action == 'click' and t.llm_action != 'click']),
            "other_as_click": len([t for t in self.failed_trials if t.human_action != 'click' and t.llm_action == 'click']),
        }


class CalibrationEngine:
    """
    Engine for running prompt calibration against held-out human data.

    This validates that a prompt configuration can make LLMs produce
    behavior matching real human responses.
    """

    def __init__(self, router, prompt_builder):
        """
        Initialize calibration engine.

        Args:
            router: ProviderRouter for making LLM calls
            prompt_builder: PromptBuilder for constructing prompts
        """
        self.router = router
        self.prompt_builder = prompt_builder

    async def run_calibration(
        self,
        persona: Persona,
        split_result: SplitResult,
        prompt_config: PromptConfiguration,
        model_id: str,
        use_train_stats: bool = True,
        use_icl: bool = True,
        emails: List[Dict] = None,
        stop_check: callable = None,
        progress_callback: callable = None
    ) -> CalibrationResult:
        """
        Run calibration trials on held-out test data.

        Args:
            persona: The persona to calibrate
            split_result: Train/test split data
            prompt_config: Which prompt configuration to test
            model_id: Which model to use for calibration
            use_train_stats: If True, use train set statistics in prompt
                           If False, use original persona statistics
            use_icl: If True, include ICL examples in prompts
            emails: List of email dicts for ICL example lookup
            stop_check: Optional callable that returns True if calibration should stop
            progress_callback: Optional callable(completed, total) to report progress

        Returns:
            CalibrationResult with accuracy metrics
        """
        result = CalibrationResult(
            persona_id=persona.persona_id,
            persona_name=persona.name,
            model_id=model_id,
            prompt_config=prompt_config.value,
            status=CalibrationStatus.RUNNING,
            started_at=datetime.now().isoformat()
        )

        try:
            # Optionally update persona stats from training data
            calibration_persona = self._update_persona_from_train(
                persona, split_result.train_statistics
            ) if use_train_stats else persona

            # Prepare training data for ICL
            training_trials_list = None
            emails_dict = None

            if use_icl:
                # Convert train trials to list of dicts for ICL
                splitter = DataSplitter()
                training_trials_list = splitter.trials_to_dicts(split_result.train_trials)

                # Create emails_dict from provided emails or from trial data
                # FIX Issue #4: Create with multiple key formats for cross-format lookup
                if emails:
                    emails_dict = {}
                    for e in emails:
                        email_id = e.get('email_id', e.get('id', ''))
                        emails_dict[email_id] = e
                        # Add normalized variants
                        if email_id.startswith('email_'):
                            num_part = email_id.replace('email_', '')
                            emails_dict[f'E{num_part}'] = e
                            try:
                                emails_dict[f'E{int(num_part)}'] = e
                            except ValueError:
                                pass
                        elif email_id.startswith('E'):
                            num_part = email_id[1:]
                            emails_dict[f'email_{num_part}'] = e
                            try:
                                emails_dict[f'email_{int(num_part):02d}'] = e
                            except ValueError:
                                pass
                else:
                    # Build from trial data
                    emails_dict = {}
                    for trial in split_result.train_trials + split_result.test_trials:
                        if trial.email_id not in emails_dict:
                            emails_dict[trial.email_id] = {
                                'email_id': trial.email_id,
                                'email_type': trial.email_type,
                                'urgency_level': trial.urgency_level,
                                'sender_familiarity': trial.sender_familiarity,
                                'framing_type': trial.framing_type,
                                'subject_line': trial.subject,
                                'subject': trial.subject,
                                'sender_display': trial.sender,
                                'sender': trial.sender,
                                'body_text': trial.body,
                                'body': trial.body
                            }

                print(f"\n🎯 ICL ENABLED: Using {len(training_trials_list)} training examples")

            # Log the prompt being used (first 500 chars)
            sample_trial = split_result.test_trials[0] if split_result.test_trials else None
            if sample_trial:
                from core.schemas import EmailStimulus
                sample_email = EmailStimulus(
                    email_id=sample_trial.email_id,
                    email_type=sample_trial.email_type,
                    sender_familiarity=sample_trial.sender_familiarity,
                    urgency_level=sample_trial.urgency_level,
                    framing_type=sample_trial.framing_type,
                    content_domain="general",
                    ground_truth=1 if sample_trial.email_type == 'phishing' else 0,
                    subject_line=sample_trial.subject,
                    sender_display=sample_trial.sender,
                    body_text=sample_trial.body
                )
                sample_prompt = self.prompt_builder.build(
                    persona=calibration_persona,
                    email=sample_email,
                    config=prompt_config,
                    trial_number=0,
                    training_trials=training_trials_list,
                    emails_dict=emails_dict,
                    use_icl=use_icl
                )
                print(f"\n📝 PROMPT BEING USED (first 600 chars):")
                print(f"   Config: {prompt_config.value}")
                print(f"   Temperature: {sample_prompt.temperature}")
                print(f"   ICL: {'Enabled' if use_icl else 'Disabled'}")
                print("-"*50)
                print(sample_prompt.system_prompt[:600] + "...")
                print("-"*50 + "\n")

            # Run predictions on test set
            from phase2.providers.base import LLMRequest

            # Store emails used for logging
            result.emails_used = emails if emails else []

            total_trials = len(split_result.test_trials)
            stopped_early = False

            for idx, test_trial in enumerate(split_result.test_trials, 1):
                # Check if stop requested
                if stop_check and stop_check():
                    print(f"\n⛔ CALIBRATION STOPPED at trial {idx}/{total_trials}")
                    stopped_early = True
                    break

                # Report progress
                if progress_callback:
                    progress_callback(idx - 1, total_trials)

                # Log trial start
                print(f"   Trial {idx}/{total_trials}: Email '{test_trial.email_id}' ({test_trial.email_type})")
                print(f"      → Human did: {test_trial.human_action.upper()}")

                # Build prompt and capture it for logging
                trial_result, prompt_data = await self._run_single_trial_with_prompt(
                    calibration_persona,
                    test_trial,
                    prompt_config,
                    model_id,
                    training_trials=training_trials_list,
                    emails_dict=emails_dict,
                    use_icl=use_icl
                )
                result.trials.append(trial_result)
                result.prompts_used.append(prompt_data)

                # Log trial result
                match_symbol = "✅" if trial_result.is_correct else "❌"
                print(f"      → LLM predicted: {trial_result.llm_action.upper()} {match_symbol}")

                if trial_result.is_correct:
                    result.n_correct += 1
                else:
                    result.failed_trials.append(trial_result)
                    if trial_result.llm_reasoning:
                        print(f"      → LLM reasoning: {trial_result.llm_reasoning[:80]}...")

                result.total_cost_usd += 0  # TODO: Track cost from router

            # Final progress update
            if progress_callback:
                progress_callback(len(result.trials), total_trials)

            # Calculate metrics
            result.n_trials = len(result.trials)
            result.accuracy = result.n_correct / result.n_trials if result.n_trials > 0 else 0

            self._calculate_detailed_metrics(result, split_result.test_statistics)

            if stopped_early:
                result.status = CalibrationStatus.FAILED
                result.error_message = f"Stopped by user at trial {len(result.trials)}/{total_trials}"
            else:
                result.status = CalibrationStatus.COMPLETED
            result.completed_at = datetime.now().isoformat()

        except Exception as e:
            result.status = CalibrationStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now().isoformat()

        return result

    def _update_persona_from_train(
        self,
        persona: Persona,
        train_stats: Dict[str, Any]
    ) -> Persona:
        """Update persona behavioral stats from training data."""
        # Create updated behavioral statistics
        updated_stats = BehavioralStatistics(
            phishing_click_rate=train_stats.get('phishing_click_rate', persona.behavioral_statistics.phishing_click_rate),
            overall_accuracy=1 - train_stats.get('phishing_click_rate', 0.3),  # Approximate
            report_rate=train_stats.get('report_rate', persona.behavioral_statistics.report_rate),
            mean_response_latency_ms=train_stats.get('mean_response_time_ms', persona.behavioral_statistics.mean_response_latency_ms),
            hover_rate=persona.behavioral_statistics.hover_rate,
            sender_inspection_rate=persona.behavioral_statistics.sender_inspection_rate,
        )

        # Calculate interaction effects from training data
        urgency_effect = (
            train_stats.get('high_urgency_click_rate', 0) -
            train_stats.get('low_urgency_click_rate', 0)
        )
        familiarity_effect = (
            train_stats.get('familiar_click_rate', 0) -
            train_stats.get('unfamiliar_click_rate', 0)
        )

        updated_effects = EmailInteractionEffects(
            urgency_effect=urgency_effect,
            familiarity_effect=familiarity_effect,
            framing_effect=persona.email_interaction_effects.framing_effect
        )

        # Create updated persona (don't modify original)
        return Persona(
            persona_id=persona.persona_id,
            cluster_id=persona.cluster_id,
            name=persona.name,
            archetype=persona.archetype,
            risk_level=persona.risk_level,
            n_participants=persona.n_participants,
            pct_of_population=persona.pct_of_population,
            description=persona.description,
            trait_zscores=persona.trait_zscores,
            distinguishing_high_traits=persona.distinguishing_high_traits,
            distinguishing_low_traits=persona.distinguishing_low_traits,
            cognitive_style=persona.cognitive_style,
            behavioral_statistics=updated_stats,
            email_interaction_effects=updated_effects,
            boundary_conditions=persona.boundary_conditions,
            reasoning_examples=persona.reasoning_examples,
            target_accuracy=persona.target_accuracy,
            acceptance_range=persona.acceptance_range,
        )

    async def _run_single_trial(
        self,
        persona: Persona,
        test_trial: HeldOutTrial,
        prompt_config: PromptConfiguration,
        model_id: str,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        use_icl: bool = True
    ) -> CalibrationTrial:
        """Run a single calibration trial with optional ICL support."""
        from phase2.providers.base import LLMRequest
        from phase2.simulation.prompt_builder import ResponseParser

        # Create email stimulus
        email = EmailStimulus(
            email_id=test_trial.email_id,
            email_type=test_trial.email_type,
            sender_familiarity=test_trial.sender_familiarity,
            urgency_level=test_trial.urgency_level,
            framing_type=test_trial.framing_type,
            content_domain="general",
            ground_truth=1 if test_trial.email_type == 'phishing' else 0,
            subject_line=test_trial.subject,
            sender_display=test_trial.sender,
            body_text=test_trial.body
        )

        # Build prompt with ICL support
        built_prompt = self.prompt_builder.build(
            persona=persona,
            email=email,
            config=prompt_config,
            trial_number=0,
            training_trials=training_trials,
            emails_dict=emails_dict,
            use_icl=use_icl
        )

        trial_result = CalibrationTrial(
            trial_id=test_trial.trial_id,
            email_id=test_trial.email_id,
            human_action=test_trial.human_action,
            human_confidence=test_trial.human_confidence,
            email_type=test_trial.email_type,
            urgency_level=test_trial.urgency_level,
            sender_familiarity=test_trial.sender_familiarity,
            prompt_used=built_prompt.system_prompt[:500]  # Truncate for storage
        )

        try:
            # Make LLM request
            request = LLMRequest(
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                temperature=built_prompt.temperature,
                top_p=built_prompt.top_p,
                max_tokens=300
            )

            response = await self.router.complete(model_id, request)

            if response.success:
                trial_result.raw_response = response.content

                # Parse response
                parsed = ResponseParser.parse(
                    response.content,
                    built_prompt.expected_format
                )

                trial_result.llm_action = parsed['action'].value if parsed['action'] else 'error'
                trial_result.llm_confidence = parsed.get('confidence', {})
                trial_result.llm_reasoning = parsed.get('reasoning', '')

                # Check if correct
                trial_result.is_correct = (
                    trial_result.llm_action.lower() == trial_result.human_action.lower()
                )
            else:
                trial_result.error_message = response.error_message
                trial_result.llm_action = 'error'

        except Exception as e:
            trial_result.error_message = str(e)
            trial_result.llm_action = 'error'

        return trial_result

    async def _run_single_trial_with_prompt(
        self,
        persona: Persona,
        test_trial: HeldOutTrial,
        prompt_config: PromptConfiguration,
        model_id: str,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        use_icl: bool = True
    ) -> tuple:
        """
        Run a single calibration trial and return both result and full prompt data.

        Returns:
            Tuple of (CalibrationTrial, Dict with prompt data for logging)
        """
        from phase2.providers.base import LLMRequest
        from phase2.simulation.prompt_builder import ResponseParser

        # Create email stimulus
        email = EmailStimulus(
            email_id=test_trial.email_id,
            email_type=test_trial.email_type,
            sender_familiarity=test_trial.sender_familiarity,
            urgency_level=test_trial.urgency_level,
            framing_type=test_trial.framing_type,
            content_domain="general",
            ground_truth=1 if test_trial.email_type == 'phishing' else 0,
            subject_line=test_trial.subject,
            sender_display=test_trial.sender,
            body_text=test_trial.body
        )

        # Build prompt with ICL support
        built_prompt = self.prompt_builder.build(
            persona=persona,
            email=email,
            config=prompt_config,
            trial_number=0,
            training_trials=training_trials,
            emails_dict=emails_dict,
            use_icl=use_icl
        )

        # Capture full prompt data for logging
        prompt_data = {
            'system_prompt': built_prompt.system_prompt,
            'user_prompt': built_prompt.user_prompt,
            'temperature': built_prompt.temperature,
            'top_p': built_prompt.top_p,
            'config': built_prompt.config.value,
            'email_id': test_trial.email_id,
            'email_subject': test_trial.subject,
            'email_sender': test_trial.sender,
            'email_body': test_trial.body,
        }

        trial_result = CalibrationTrial(
            trial_id=test_trial.trial_id,
            email_id=test_trial.email_id,
            human_action=test_trial.human_action,
            human_confidence=test_trial.human_confidence,
            email_type=test_trial.email_type,
            urgency_level=test_trial.urgency_level,
            sender_familiarity=test_trial.sender_familiarity,
            prompt_used=built_prompt.system_prompt[:500]  # Truncate for storage
        )

        try:
            # Make LLM request
            request = LLMRequest(
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                temperature=built_prompt.temperature,
                top_p=built_prompt.top_p,
                max_tokens=300
            )

            response = await self.router.complete(model_id, request)

            if response.success:
                trial_result.raw_response = response.content

                # Parse response
                parsed = ResponseParser.parse(
                    response.content,
                    built_prompt.expected_format
                )

                trial_result.llm_action = parsed['action'].value if parsed['action'] else 'error'
                trial_result.llm_confidence = parsed.get('confidence', {})
                trial_result.llm_reasoning = parsed.get('reasoning', '')

                # Check if correct
                trial_result.is_correct = (
                    trial_result.llm_action.lower() == trial_result.human_action.lower()
                )
            else:
                trial_result.error_message = response.error_message
                trial_result.llm_action = 'error'

        except Exception as e:
            trial_result.error_message = str(e)
            trial_result.llm_action = 'error'

        return trial_result, prompt_data

    def _calculate_detailed_metrics(
        self,
        result: CalibrationResult,
        test_stats: Dict[str, Any]
    ):
        """Calculate detailed accuracy metrics."""
        trials = result.trials

        # Count actions
        human_clicks = sum(1 for t in trials if t.human_action == 'click')
        human_reports = sum(1 for t in trials if t.human_action == 'report')
        llm_clicks = sum(1 for t in trials if t.llm_action == 'click')
        llm_reports = sum(1 for t in trials if t.llm_action == 'report')

        n = len(trials)
        if n == 0:
            return

        # Overall rates
        result.human_click_rate = human_clicks / n
        result.llm_click_rate = llm_clicks / n
        result.click_rate_error = abs(result.llm_click_rate - result.human_click_rate)

        result.human_report_rate = human_reports / n
        result.llm_report_rate = llm_reports / n
        result.report_rate_error = abs(result.llm_report_rate - result.human_report_rate)

        # Precision/recall for click
        llm_click_trials = [t for t in trials if t.llm_action == 'click']
        if llm_click_trials:
            result.click_precision = sum(1 for t in llm_click_trials if t.human_action == 'click') / len(llm_click_trials)

        human_click_trials = [t for t in trials if t.human_action == 'click']
        if human_click_trials:
            result.click_recall = sum(1 for t in human_click_trials if t.llm_action == 'click') / len(human_click_trials)

        # Precision/recall for report
        llm_report_trials = [t for t in trials if t.llm_action == 'report']
        if llm_report_trials:
            result.report_precision = sum(1 for t in llm_report_trials if t.human_action == 'report') / len(llm_report_trials)

        human_report_trials = [t for t in trials if t.human_action == 'report']
        if human_report_trials:
            result.report_recall = sum(1 for t in human_report_trials if t.llm_action == 'report') / len(human_report_trials)

        # Factor-specific accuracy
        phishing_trials = [t for t in trials if t.email_type == 'phishing']
        if phishing_trials:
            result.phishing_accuracy = sum(1 for t in phishing_trials if t.is_correct) / len(phishing_trials)

        legit_trials = [t for t in trials if t.email_type == 'legitimate']
        if legit_trials:
            result.legitimate_accuracy = sum(1 for t in legit_trials if t.is_correct) / len(legit_trials)

        high_urgency = [t for t in trials if t.urgency_level == 'high']
        if high_urgency:
            result.high_urgency_accuracy = sum(1 for t in high_urgency if t.is_correct) / len(high_urgency)

        low_urgency = [t for t in trials if t.urgency_level == 'low']
        if low_urgency:
            result.low_urgency_accuracy = sum(1 for t in low_urgency if t.is_correct) / len(low_urgency)

    async def run_multi_config_calibration(
        self,
        persona: Persona,
        split_result: SplitResult,
        model_id: str,
        use_icl: bool = True,
        emails: List[Dict] = None
    ) -> Dict[str, CalibrationResult]:
        """
        Run calibration for all three prompt configurations.

        Args:
            persona: The persona to calibrate
            split_result: Train/test split data
            model_id: Which model to use
            use_icl: Whether to use ICL examples
            emails: List of email dicts for ICL

        Returns:
            Dict mapping config name to CalibrationResult
        """
        results = {}

        for config in [PromptConfiguration.BASELINE, PromptConfiguration.STATS, PromptConfiguration.COT]:
            results[config.value] = await self.run_calibration(
                persona, split_result, config, model_id,
                use_icl=use_icl, emails=emails
            )

        return results

    async def run_multi_model_calibration(
        self,
        persona: Persona,
        split_result: SplitResult,
        prompt_config: PromptConfiguration,
        model_ids: List[str]
    ) -> Dict[str, CalibrationResult]:
        """
        Run calibration for same config across multiple models.

        Returns:
            Dict mapping model_id to CalibrationResult
        """
        results = {}

        for model_id in model_ids:
            results[model_id] = await self.run_calibration(
                persona, split_result, prompt_config, model_id
            )

        return results
