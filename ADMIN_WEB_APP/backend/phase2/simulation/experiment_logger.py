"""
CYPEARL Phase 2 - Experiment Logger

Generates detailed log files for experiment runs to help debug
and analyze LLM behavior across different combinations.

Creates comprehensive text files with:
- Experiment configuration
- Per-model statistics
- Per-persona breakdown
- Detailed trial logs grouped by model
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict


class ExperimentLogger:
    """
    Generates detailed experiment log files for analysis.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize the logger.

        Args:
            output_dir: Directory to save log files. Defaults to data/experiment_logs/
        """
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent  # backend/
            output_dir = base_dir / "data" / "experiment_logs"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_log_filename(self, experiment_name: str) -> str:
        """Generate a unique filename for the log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = experiment_name.replace(" ", "_").replace("/", "_")[:50]
        return f"experiment_{safe_name}_{timestamp}.txt"

    def create_full_log(
        self,
        experiment: Any,
        trials: List[Any],
        personas: Dict[str, Any],
        emails: Dict[str, Any]
    ) -> str:
        """
        Create a comprehensive experiment log file.

        Returns:
            Path to the generated log file
        """
        filename = self.generate_log_filename(experiment.name)
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("CYPEARL PHASE 2 - EXPERIMENT DETAILED LOG\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Log File: {filepath}\n\n")

            # Section 1: Experiment Configuration
            self._write_config_section(f, experiment, personas, emails)

            # Section 2: Overall Summary
            self._write_summary_section(f, trials)

            # Section 3: Model-by-Model Breakdown
            self._write_model_breakdown(f, trials, personas)

            # Section 4: Persona-by-Persona Breakdown
            self._write_persona_breakdown(f, trials, personas)

            # Section 5: Detailed Trial Logs (grouped by model)
            self._write_trials_section(f, trials, personas, emails)

            # Footer
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF EXPERIMENT LOG\n")
            f.write("=" * 100 + "\n")

        return str(filepath)

    def _write_config_section(
        self,
        f,
        experiment: Any,
        personas: Dict[str, Any],
        emails: Dict[str, Any]
    ):
        """Write experiment configuration section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 1: EXPERIMENT CONFIGURATION\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"Experiment ID: {experiment.experiment_id}\n")
        f.write(f"Experiment Name: {experiment.name}\n")
        f.write(f"Description: {experiment.description or 'N/A'}\n\n")

        f.write("DESIGN FACTORS:\n")
        f.write(f"  Personas: {len(experiment.persona_ids)}\n")
        for pid in experiment.persona_ids:
            p = personas.get(pid)
            f.write(f"    - {pid}: {p.name if p else 'Unknown'}\n")

        f.write(f"\n  Models: {len(experiment.model_ids)}\n")
        for mid in experiment.model_ids:
            f.write(f"    - {mid}\n")

        f.write(f"\n  Prompt Configs: {len(experiment.prompt_configs)}\n")
        for pc in experiment.prompt_configs:
            f.write(f"    - {pc.value if hasattr(pc, 'value') else pc}\n")

        f.write(f"\n  Emails: {len(experiment.email_ids)}\n")
        for eid in experiment.email_ids:
            email = emails.get(eid)
            if email:
                f.write(f"    - {eid}: {getattr(email, 'subject', 'N/A')[:50]}\n")
            else:
                f.write(f"    - {eid}\n")

        f.write(f"\n  Trials per Condition: {experiment.trials_per_condition}\n")
        f.write(f"  Total Trials: {experiment.total_trials}\n\n")

    def _write_summary_section(self, f, trials: List[Any]):
        """Write overall summary section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 2: OVERALL SUMMARY\n")
        f.write("-" * 100 + "\n\n")

        total = len(trials)
        successful = sum(1 for t in trials if t.parse_success)
        clicked = sum(1 for t in trials if t.ai_action == "click")
        reported = sum(1 for t in trials if t.ai_action == "report")
        ignored = sum(1 for t in trials if t.ai_action == "ignore")
        total_cost = sum(t.cost_usd for t in trials)
        latencies = [t.model_latency_ms for t in trials if t.model_latency_ms > 0]

        f.write(f"Total Trials: {total}\n")
        f.write(f"Parse Success Rate: {successful/total*100:.1f}% ({successful}/{total})\n\n")

        f.write("ACTION DISTRIBUTION:\n")
        f.write(f"  - Click: {clicked} ({clicked/total*100:.1f}%)\n")
        f.write(f"  - Report: {reported} ({reported/total*100:.1f}%)\n")
        f.write(f"  - Ignore: {ignored} ({ignored/total*100:.1f}%)\n\n")

        f.write("COST SUMMARY:\n")
        f.write(f"  - Total Cost: ${total_cost:.4f}\n")
        f.write(f"  - Avg Cost per Trial: ${total_cost/total:.6f}\n\n")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            f.write("LATENCY SUMMARY:\n")
            f.write(f"  - Avg Latency: {avg_latency:.0f}ms\n")
            f.write(f"  - Min Latency: {min(latencies):.0f}ms\n")
            f.write(f"  - Max Latency: {max(latencies):.0f}ms\n\n")

    def _write_model_breakdown(self, f, trials: List[Any], personas: Dict[str, Any]):
        """Write model-by-model breakdown section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 3: MODEL-BY-MODEL BREAKDOWN\n")
        f.write("-" * 100 + "\n\n")

        # Group by model
        by_model = defaultdict(list)
        for trial in trials:
            by_model[trial.model_id].append(trial)

        for model_id, model_trials in sorted(by_model.items()):
            f.write(f"\n{'='*60}\n")
            f.write(f"MODEL: {model_id}\n")
            f.write(f"{'='*60}\n\n")

            total = len(model_trials)
            clicked = sum(1 for t in model_trials if t.ai_action == "click")
            reported = sum(1 for t in model_trials if t.ai_action == "report")
            cost = sum(t.cost_usd for t in model_trials)
            latencies = [t.model_latency_ms for t in model_trials if t.model_latency_ms > 0]

            f.write(f"  Total Trials: {total}\n")
            f.write(f"  Click Rate: {clicked/total*100:.1f}%\n")
            f.write(f"  Report Rate: {reported/total*100:.1f}%\n")
            f.write(f"  Total Cost: ${cost:.4f}\n")
            f.write(f"  Cost/Trial: ${cost/total:.6f}\n")
            if latencies:
                f.write(f"  Avg Latency: {sum(latencies)/len(latencies):.0f}ms\n")

            # Breakdown by prompt config
            f.write("\n  By Prompt Config:\n")
            by_prompt = defaultdict(list)
            for t in model_trials:
                pc = t.prompt_config.value if hasattr(t.prompt_config, 'value') else str(t.prompt_config)
                by_prompt[pc].append(t)

            for prompt, prompt_trials in by_prompt.items():
                pt = len(prompt_trials)
                pc = sum(1 for t in prompt_trials if t.ai_action == "click")
                f.write(f"    {prompt}: {pt} trials, {pc/pt*100:.1f}% click rate\n")

            f.write("\n")

    def _write_persona_breakdown(self, f, trials: List[Any], personas: Dict[str, Any]):
        """Write persona-by-persona breakdown section."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 4: PERSONA-BY-PERSONA BREAKDOWN\n")
        f.write("-" * 100 + "\n\n")

        # Group by persona
        by_persona = defaultdict(list)
        for trial in trials:
            by_persona[trial.persona_id].append(trial)

        for persona_id, persona_trials in sorted(by_persona.items()):
            persona = personas.get(persona_id)
            persona_name = persona.name if persona else persona_id

            f.write(f"\n{'='*60}\n")
            f.write(f"PERSONA: {persona_name} ({persona_id})\n")
            f.write(f"{'='*60}\n\n")

            total = len(persona_trials)
            clicked = sum(1 for t in persona_trials if t.ai_action == "click")

            f.write(f"  Total Trials: {total}\n")
            f.write(f"  AI Click Rate: {clicked/total*100:.1f}%\n")

            if persona:
                f.write(f"  Human Click Rate (baseline): {persona.behavioral_statistics.phishing_click_rate*100:.1f}%\n")
                deviation = (clicked/total) - persona.behavioral_statistics.phishing_click_rate
                f.write(f"  Deviation: {'+' if deviation > 0 else ''}{deviation*100:.1f}%\n")

            # Breakdown by model
            f.write("\n  By Model:\n")
            by_model = defaultdict(list)
            for t in persona_trials:
                by_model[t.model_id].append(t)

            for model, model_trials in by_model.items():
                mt = len(model_trials)
                mc = sum(1 for t in model_trials if t.ai_action == "click")
                f.write(f"    {model}: {mt} trials, {mc/mt*100:.1f}% click rate\n")

            f.write("\n")

    def _write_trials_section(
        self,
        f,
        trials: List[Any],
        personas: Dict[str, Any],
        emails: Dict[str, Any]
    ):
        """Write detailed trial logs grouped by model."""
        f.write("-" * 100 + "\n")
        f.write("SECTION 5: DETAILED TRIAL LOGS (GROUPED BY MODEL)\n")
        f.write("-" * 100 + "\n\n")

        # Group by model for readability
        by_model = defaultdict(list)
        for trial in trials:
            by_model[trial.model_id].append(trial)

        for model_id, model_trials in sorted(by_model.items()):
            f.write(f"\n{'#'*100}\n")
            f.write(f"# MODEL: {model_id} ({len(model_trials)} trials)\n")
            f.write(f"{'#'*100}\n")

            for idx, trial in enumerate(model_trials[:50]):  # Limit to first 50 per model
                f.write(f"\n{'-'*60}\n")
                f.write(f"Trial {idx + 1} | ID: {trial.trial_id}\n")
                f.write(f"{'-'*60}\n\n")

                persona = personas.get(trial.persona_id)
                email = emails.get(trial.email_id)

                f.write(f"Persona: {persona.name if persona else trial.persona_id}\n")
                f.write(f"Email: {trial.email_id}")
                if email:
                    f.write(f" - {getattr(email, 'subject', 'N/A')[:40]}")
                f.write("\n")

                pc = trial.prompt_config.value if hasattr(trial.prompt_config, 'value') else str(trial.prompt_config)
                f.write(f"Prompt Config: {pc}\n\n")

                f.write(f"AI Action: {trial.ai_action.upper()}\n")
                if trial.ai_confidence:
                    f.write(f"Confidence: {trial.ai_confidence}\n")
                if trial.ai_reasoning:
                    f.write(f"Reasoning: {trial.ai_reasoning[:200]}...\n")

                f.write(f"\nLatency: {trial.model_latency_ms}ms\n")
                f.write(f"Cost: ${trial.cost_usd:.6f}\n")
                f.write(f"Parse Success: {trial.parse_success}\n")

                if trial.raw_response and len(trial.raw_response) < 500:
                    f.write(f"\nRaw Response:\n{trial.raw_response}\n")

            if len(model_trials) > 50:
                f.write(f"\n... and {len(model_trials) - 50} more trials for {model_id}\n")

        f.write("\n")


def create_experiment_log(
    experiment: Any,
    trials: List[Any],
    personas: Dict[str, Any],
    emails: Dict[str, Any],
    output_dir: str = None
) -> str:
    """
    Convenience function to create an experiment log.

    Returns:
        Path to the generated log file
    """
    logger = ExperimentLogger(output_dir)
    return logger.create_full_log(
        experiment=experiment,
        trials=trials,
        personas=personas,
        emails=emails
    )
