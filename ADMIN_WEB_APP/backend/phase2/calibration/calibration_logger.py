"""
CYPEARL Phase 2 - Calibration Logger

Generates detailed log files for calibration runs to help debug
prompt configurations and understand LLM behavior.

Creates a comprehensive text file with:
- Persona characteristics (all 29 traits)
- Model and prompt configuration
- Full prompts used
- Email details for each trial
- LLM responses and reasoning
- Accuracy breakdown
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class CalibrationLogger:
    """
    Generates detailed calibration log files for debugging and analysis.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the logger.

        Args:
            output_dir: Directory to save log files. Defaults to data/calibration_logs/
        """
        if output_dir is None:
            # Default to ADMIN_WEB_APP/data/calibration_logs/
            base_dir = Path(__file__).parent.parent.parent  # backend/
            output_dir = base_dir / "data" / "calibration_logs"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_log_filename(
        self,
        persona_id: str,
        model_id: str,
        prompt_config: str
    ) -> str:
        """Generate a unique filename for the log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model_id.replace("/", "_").replace(":", "_")
        return f"calibration_{persona_id}_{safe_model}_{prompt_config}_{timestamp}.txt"

    def create_full_log(
        self,
        persona: Any,
        model_id: str,
        prompt_config: str,
        split_result: Any,
        calibration_result: Any,
        prompts_used: List[Dict[str, Any]],
        emails_used: List[Dict[str, Any]],
        use_icl: bool = True
    ) -> str:
        """
        Create a comprehensive calibration log file.

        Args:
            persona: The Persona object
            model_id: Model ID used for calibration
            prompt_config: Prompt configuration (baseline/stats/cot)
            split_result: SplitResult with train/test data
            calibration_result: CalibrationResult with trial outcomes
            prompts_used: List of full prompts used for each trial
            emails_used: List of email details for each trial
            use_icl: Whether ICL was enabled

        Returns:
            Path to the generated log file
        """
        filename = self.generate_log_filename(
            persona.persona_id, model_id, prompt_config
        )
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("CYPEARL PHASE 2 - CALIBRATION DETAILED LOG\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Log File: {filepath}\n\n")

            # Section 1: Configuration Summary
            self._write_config_section(f, persona, model_id, prompt_config, use_icl)

            # Section 2: Persona Full Profile
            self._write_persona_section(f, persona)

            # Section 3: Data Split Summary
            self._write_split_section(f, split_result)

            # Section 4: Calibration Results Summary
            self._write_results_section(f, calibration_result)

            # Section 5: Detailed Trial-by-Trial Log
            self._write_trials_section(
                f, calibration_result, prompts_used, emails_used
            )

            # Footer
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF CALIBRATION LOG\n")
            f.write("=" * 100 + "\n")

        return str(filepath)

    def _write_config_section(
        self,
        f,
        persona: Any,
        model_id: str,
        prompt_config: str,
        use_icl: bool
    ):
        """Write configuration summary section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 1: CONFIGURATION\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"Persona ID: {persona.persona_id}\n")
        f.write(f"Persona Name: {persona.name}\n")
        f.write(f"Persona Archetype: {persona.archetype}\n")
        f.write(f"Risk Level: {persona.risk_level}\n\n")

        f.write(f"Model: {model_id}\n")
        f.write(f"Prompt Configuration: {prompt_config}\n")
        f.write(f"In-Context Learning (ICL): {'ENABLED' if use_icl else 'DISABLED'}\n")
        f.write(f"Cognitive Style: {persona.cognitive_style}\n\n")

    def _write_persona_section(self, f, persona: Any):
        """Write full persona profile section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 2: PERSONA FULL PROFILE\n")
        f.write("-" * 100 + "\n\n")

        # Description
        f.write("DESCRIPTION:\n")
        f.write(f"{persona.description}\n\n")

        # Behavioral Statistics
        f.write("BEHAVIORAL STATISTICS (8 Outcomes):\n")
        stats = persona.behavioral_statistics
        f.write(f"  1. Phishing Click Rate: {stats.phishing_click_rate * 100:.1f}%\n")
        f.write(f"  2. Overall Accuracy: {(stats.overall_accuracy or 0) * 100:.1f}%\n")
        f.write(f"  3. Report Rate: {(stats.report_rate or 0) * 100:.1f}%\n")
        f.write(f"  4. Mean Response Latency: {stats.mean_response_latency_ms or 0:.0f}ms\n")
        f.write(f"  5. Hover Rate: {(stats.hover_rate or 0) * 100:.1f}%\n")
        f.write(f"  6. Sender Inspection Rate: {(stats.sender_inspection_rate or 0) * 100:.1f}%\n")

        # Email interaction effects
        effects = persona.email_interaction_effects
        if effects:
            urgency_effect = effects.urgency_effect if hasattr(effects, 'urgency_effect') else effects.get('urgency_effect', 0)
            familiarity_effect = effects.familiarity_effect if hasattr(effects, 'familiarity_effect') else effects.get('familiarity_effect', 0)
            f.write(f"  7. Urgency Effect: {'+' if urgency_effect > 0 else ''}{urgency_effect * 100:.1f}%\n")
            f.write(f"  8. Familiarity Effect: {'+' if familiarity_effect > 0 else ''}{familiarity_effect * 100:.1f}%\n")
        f.write("\n")

        # All 29 Traits
        f.write("ALL 29 PSYCHOLOGICAL/COGNITIVE TRAITS:\n")
        traits = persona.trait_zscores or {}

        # Group traits by category
        trait_categories = {
            'COGNITIVE': ['crt_score', 'need_for_cognition', 'working_memory', 'impulsivity_total'],
            'PERSONALITY (Big 5)': ['big5_extraversion', 'big5_agreeableness', 'big5_conscientiousness',
                                    'big5_neuroticism', 'big5_openness'],
            'PSYCHOLOGICAL': ['trust_propensity', 'risk_taking', 'state_anxiety', 'current_stress',
                             'fatigue_level', 'sensation_seeking'],
            'SECURITY AWARENESS': ['phishing_self_efficacy', 'perceived_risk', 'security_attitudes',
                                   'privacy_concern', 'phishing_knowledge', 'technical_expertise',
                                   'prior_victimization', 'security_training'],
            'SUSCEPTIBILITIES': ['authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility'],
            'BEHAVIORAL': ['link_click_tendency', 'email_volume_numeric', 'social_media_usage']
        }

        for category, trait_names in trait_categories.items():
            f.write(f"\n  {category}:\n")
            for trait in trait_names:
                z_score = traits.get(trait, 0)
                level = self._z_to_level(z_score)
                f.write(f"    - {trait.replace('_', ' ').title()}: {level} (z={z_score:+.2f})\n")

        # Distinguishing traits
        f.write("\n  DISTINGUISHING HIGH TRAITS:\n")
        for trait in (persona.distinguishing_high_traits or []):
            f.write(f"    - {trait}\n")

        f.write("\n  DISTINGUISHING LOW TRAITS:\n")
        for trait in (persona.distinguishing_low_traits or []):
            f.write(f"    - {trait}\n")

        f.write("\n")

    def _write_split_section(self, f, split_result: Any):
        """Write data split summary section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 3: DATA SPLIT SUMMARY\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"Split Ratio: {split_result.split_ratio * 100:.0f}% train / {(1 - split_result.split_ratio) * 100:.0f}% test\n")
        f.write(f"Random Seed: {split_result.random_seed}\n\n")

        f.write("TRAINING SET:\n")
        f.write(f"  - Total Trials: {split_result.n_train}\n")
        train_stats = split_result.train_statistics
        f.write(f"  - Phishing Click Rate: {train_stats.get('phishing_click_rate', 0) * 100:.1f}%\n")
        f.write(f"  - Report Rate: {train_stats.get('report_rate', 0) * 100:.1f}%\n")
        f.write(f"  - High Urgency Click Rate: {train_stats.get('high_urgency_click_rate', 0) * 100:.1f}%\n")
        f.write(f"  - Familiar Sender Click Rate: {train_stats.get('familiar_click_rate', 0) * 100:.1f}%\n\n")

        f.write("TEST SET:\n")
        f.write(f"  - Total Trials: {split_result.n_test}\n")
        test_stats = split_result.test_statistics
        f.write(f"  - Phishing Click Rate: {test_stats.get('phishing_click_rate', 0) * 100:.1f}%\n")
        f.write(f"  - Report Rate: {test_stats.get('report_rate', 0) * 100:.1f}%\n\n")

    def _write_results_section(self, f, calibration_result: Any):
        """Write calibration results summary section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 4: CALIBRATION RESULTS\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"OVERALL ACCURACY: {calibration_result.accuracy * 100:.1f}% ({calibration_result.n_correct}/{calibration_result.n_trials})\n")
        f.write(f"MEETS 80% THRESHOLD: {'YES' if calibration_result.meets_threshold(0.80) else 'NO'}\n\n")

        f.write("CLICK RATE COMPARISON:\n")
        f.write(f"  - Human Click Rate: {calibration_result.human_click_rate * 100:.1f}%\n")
        f.write(f"  - LLM Click Rate: {calibration_result.llm_click_rate * 100:.1f}%\n")
        f.write(f"  - Click Rate Error: {calibration_result.click_rate_error * 100:.1f}%\n\n")

        f.write("REPORT RATE COMPARISON:\n")
        f.write(f"  - Human Report Rate: {calibration_result.human_report_rate * 100:.1f}%\n")
        f.write(f"  - LLM Report Rate: {calibration_result.llm_report_rate * 100:.1f}%\n")
        f.write(f"  - Report Rate Error: {calibration_result.report_rate_error * 100:.1f}%\n\n")

        f.write("PRECISION/RECALL:\n")
        f.write(f"  - Click Precision: {calibration_result.click_precision * 100:.1f}%\n")
        f.write(f"  - Click Recall: {calibration_result.click_recall * 100:.1f}%\n")
        f.write(f"  - Report Precision: {calibration_result.report_precision * 100:.1f}%\n")
        f.write(f"  - Report Recall: {calibration_result.report_recall * 100:.1f}%\n\n")

        f.write("ACCURACY BY FACTOR:\n")
        f.write(f"  - Phishing Accuracy: {calibration_result.phishing_accuracy * 100:.1f}%\n")
        f.write(f"  - Legitimate Accuracy: {calibration_result.legitimate_accuracy * 100:.1f}%\n")
        f.write(f"  - High Urgency Accuracy: {calibration_result.high_urgency_accuracy * 100:.1f}%\n")
        f.write(f"  - Low Urgency Accuracy: {calibration_result.low_urgency_accuracy * 100:.1f}%\n\n")

        # Failure summary
        failure_summary = calibration_result.get_failure_summary()
        if not failure_summary.get('no_failures'):
            f.write("FAILURE BREAKDOWN:\n")
            f.write(f"  - Total Failures: {failure_summary.get('total_failures', 0)}\n")
            f.write(f"  - Click as Other: {failure_summary.get('click_as_other', 0)}\n")
            f.write(f"  - Other as Click: {failure_summary.get('other_as_click', 0)}\n\n")

    def _write_trials_section(
        self,
        f,
        calibration_result: Any,
        prompts_used: List[Dict[str, Any]],
        emails_used: List[Dict[str, Any]]
    ):
        """Write detailed trial-by-trial log section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 5: DETAILED TRIAL-BY-TRIAL LOG\n")
        f.write("-" * 100 + "\n\n")

        for idx, trial in enumerate(calibration_result.trials):
            f.write(f"\n{'='*80}\n")
            f.write(f"TRIAL {idx + 1}/{len(calibration_result.trials)}\n")
            f.write(f"{'='*80}\n\n")

            # Trial info
            f.write(f"Trial ID: {trial.trial_id}\n")
            f.write(f"Email ID: {trial.email_id}\n")
            f.write(f"Email Type: {trial.email_type}\n")
            f.write(f"Urgency: {trial.urgency_level}\n")
            f.write(f"Sender Familiarity: {trial.sender_familiarity}\n\n")

            # Email content (from emails_used if available)
            email_data = None
            if emails_used:
                for e in emails_used:
                    if e.get('email_id') == trial.email_id:
                        email_data = e
                        break

            if email_data:
                f.write("EMAIL CONTENT:\n")
                f.write(f"  Subject: {email_data.get('subject_line', email_data.get('subject', 'N/A'))}\n")
                f.write(f"  Sender: {email_data.get('sender_display', email_data.get('sender', 'N/A'))}\n")
                f.write(f"  Body:\n")
                body = email_data.get('body_text', email_data.get('body', 'N/A'))
                for line in body.split('\n'):
                    f.write(f"    {line}\n")
                f.write("\n")

            # Human response
            f.write("HUMAN (GROUND TRUTH):\n")
            f.write(f"  Action: {trial.human_action.upper()}\n")
            if trial.human_confidence:
                f.write(f"  Confidence: {trial.human_confidence}\n")
            f.write("\n")

            # LLM response
            f.write("LLM PREDICTION:\n")
            f.write(f"  Action: {trial.llm_action.upper()}\n")
            if trial.llm_confidence:
                f.write(f"  Confidence: {trial.llm_confidence}\n")
            if trial.llm_reasoning:
                f.write(f"  Reasoning: {trial.llm_reasoning}\n")
            f.write("\n")

            # Match result
            match_symbol = "CORRECT" if trial.is_correct else "INCORRECT"
            f.write(f"RESULT: {match_symbol}\n")

            if trial.error_message:
                f.write(f"ERROR: {trial.error_message}\n")

            # Full prompt used (if available)
            if prompts_used and idx < len(prompts_used):
                prompt_data = prompts_used[idx]
                f.write("\n" + "-" * 60 + "\n")
                f.write("FULL SYSTEM PROMPT USED:\n")
                f.write("-" * 60 + "\n")
                f.write(prompt_data.get('system_prompt', 'N/A'))
                f.write("\n")
                f.write("\n" + "-" * 60 + "\n")
                f.write("USER PROMPT (EMAIL PRESENTATION):\n")
                f.write("-" * 60 + "\n")
                f.write(prompt_data.get('user_prompt', 'N/A'))
                f.write("\n")

            # Raw LLM response
            if trial.raw_response:
                f.write("\n" + "-" * 60 + "\n")
                f.write("RAW LLM RESPONSE:\n")
                f.write("-" * 60 + "\n")
                f.write(trial.raw_response)
                f.write("\n")

            f.write("\n")

    def _z_to_level(self, z: float) -> str:
        """Convert z-score to human-readable level."""
        if z > 0.8:
            return "VERY HIGH"
        elif z > 0.3:
            return "HIGH"
        elif z > -0.3:
            return "MODERATE"
        elif z > -0.8:
            return "LOW"
        else:
            return "VERY LOW"


def create_calibration_log(
    persona: Any,
    model_id: str,
    prompt_config: str,
    split_result: Any,
    calibration_result: Any,
    prompts_used: List[Dict[str, Any]],
    emails_used: List[Dict[str, Any]],
    use_icl: bool = True,
    output_dir: str = None
) -> str:
    """
    Convenience function to create a calibration log.

    Returns:
        Path to the generated log file
    """
    logger = CalibrationLogger(output_dir)
    return logger.create_full_log(
        persona=persona,
        model_id=model_id,
        prompt_config=prompt_config,
        split_result=split_result,
        calibration_result=calibration_result,
        prompts_used=prompts_used,
        emails_used=emails_used,
        use_icl=use_icl
    )
