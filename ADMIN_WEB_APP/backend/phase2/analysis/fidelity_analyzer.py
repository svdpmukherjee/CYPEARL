"""
CYPEARL Phase 2 - Fidelity Analysis
Calculate behavioral fidelity metrics comparing AI to human personas.

FIXED: All numpy types converted to native Python types for JSON serialization.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from core.schemas import (
    Persona, SimulationTrial, FidelityMetrics,
    PromptConfiguration, ActionType, ModelTier
)


def to_python_float(val) -> float:
    """Convert numpy float to Python float for JSON serialization."""
    if val is None:
        return 0.0
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def to_python_int(val) -> int:
    """Convert numpy int to Python int for JSON serialization."""
    if val is None:
        return 0
    if isinstance(val, (np.floating, np.integer)):
        return int(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


@dataclass
class EmailLevelResult:
    """Results aggregated at email level."""
    email_id: str
    n_trials: int
    click_rate: float
    report_rate: float
    ignore_rate: float
    modal_action: ActionType


@dataclass
class ConditionResult:
    """Results for a persona-model-prompt condition."""
    persona_id: str
    model_id: str
    prompt_config: PromptConfiguration
    
    n_trials: int
    n_emails: int
    
    # Aggregated rates
    ai_click_rate: float
    ai_report_rate: float
    ai_ignore_rate: float
    
    # Per-email results
    email_results: Dict[str, EmailLevelResult]
    
    # Parse success rate
    parse_success_rate: float
    
    # Timing
    mean_latency_ms: float
    total_cost: float


class FidelityAnalyzer:
    """
    Analyzes AI persona fidelity against human baselines.
    
    Primary metrics (from proposal):
    - Normalized Accuracy >= 85%
    - Decision Agreement >= 80%
    - Effect Preservation r >= 0.80
    """
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def calculate_fidelity(
        self,
        trials: List[SimulationTrial],
        persona: Persona,
        model_id: str,
        prompt_config: PromptConfiguration
    ) -> FidelityMetrics:
        """
        Calculate fidelity metrics for a specific condition.
        """
        # Filter to successful trials
        valid_trials = [t for t in trials if t.parse_success and t.action != ActionType.ERROR]
        
        if len(valid_trials) == 0:
            return FidelityMetrics(
                persona_id=persona.persona_id,
                model_id=model_id,
                prompt_config=prompt_config,
                normalized_accuracy=0.0,
                decision_agreement=0.0,
                ai_click_rate=0.0,
                human_click_rate=to_python_float(persona.behavioral_statistics.phishing_click_rate),
                click_rate_diff=1.0,
                ai_report_rate=0.0,
                human_report_rate=to_python_float(persona.behavioral_statistics.report_rate),
                report_rate_diff=1.0,
                n_trials=len(trials),
                n_emails=0,
                ci_lower=0.0,
                ci_upper=0.0,
                meets_threshold=False,
                threshold_used=self.threshold
            )
        
        # Calculate AI rates
        n_click = sum(1 for t in valid_trials if t.action == ActionType.CLICK)
        n_report = sum(1 for t in valid_trials if t.action == ActionType.REPORT)
        n_total = len(valid_trials)
        
        ai_click_rate = n_click / n_total
        ai_report_rate = n_report / n_total
        
        # Human baselines from persona
        human_click_rate = to_python_float(persona.behavioral_statistics.phishing_click_rate)
        human_report_rate = to_python_float(persona.behavioral_statistics.report_rate)
        
        # Calculate differences
        click_rate_diff = abs(ai_click_rate - human_click_rate)
        report_rate_diff = abs(ai_report_rate - human_report_rate)
        
        # Normalized Accuracy: 1 - |AI_rate - Human_rate| / Human_rate
        if human_click_rate > 0:
            normalized_accuracy = max(0.0, 1.0 - (click_rate_diff / human_click_rate))
        else:
            normalized_accuracy = 1.0 if ai_click_rate < 0.05 else 0.0
        
        # Decision Agreement: % of trials where AI matches expected behavior
        expected_action = ActionType.CLICK if human_click_rate > 0.5 else ActionType.REPORT
        n_agree = sum(1 for t in valid_trials if t.action == expected_action)
        decision_agreement = n_agree / n_total
        
        # Confidence interval for click rate (95%) - CONVERT TO PYTHON FLOAT
        if n_total > 0:
            se = float(np.sqrt((ai_click_rate * (1 - ai_click_rate)) / n_total))
            ci_lower = max(0.0, ai_click_rate - 1.96 * se)
            ci_upper = min(1.0, ai_click_rate + 1.96 * se)
        else:
            ci_lower, ci_upper = 0.0, 0.0
        
        # Count unique emails
        n_emails = len(set(t.email_id for t in valid_trials))
        
        return FidelityMetrics(
            persona_id=persona.persona_id,
            model_id=model_id,
            prompt_config=prompt_config,
            normalized_accuracy=to_python_float(normalized_accuracy),
            decision_agreement=to_python_float(decision_agreement),
            ai_click_rate=to_python_float(ai_click_rate),
            human_click_rate=to_python_float(human_click_rate),
            click_rate_diff=to_python_float(click_rate_diff),
            ai_report_rate=to_python_float(ai_report_rate),
            human_report_rate=to_python_float(human_report_rate),
            report_rate_diff=to_python_float(report_rate_diff),
            n_trials=n_total,
            n_emails=n_emails,
            ci_lower=to_python_float(ci_lower),
            ci_upper=to_python_float(ci_upper),
            meets_threshold=normalized_accuracy >= self.threshold,
            threshold_used=self.threshold
        )
    
    def calculate_effect_preservation(
        self,
        trials: List[SimulationTrial],
        persona: Persona,
        emails: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate how well AI preserves email manipulation effects."""
        valid_trials = [t for t in trials if t.parse_success]
        
        if len(valid_trials) < 10:
            return {
                'urgency_r': None, 
                'familiarity_r': None, 
                'framing_r': None,
                'urgency_ai': None,
                'urgency_human': to_python_float(persona.email_interaction_effects.urgency_effect),
                'familiarity_ai': None,
                'familiarity_human': to_python_float(persona.email_interaction_effects.familiarity_effect),
            }
        
        # Group trials by email characteristics
        urgency_high_clicks = []
        urgency_low_clicks = []
        familiar_clicks = []
        unfamiliar_clicks = []
        
        for trial in valid_trials:
            email = emails.get(trial.email_id)
            if not email:
                continue
            
            is_click = 1 if trial.action == ActionType.CLICK else 0
            
            # Get urgency level - handle both dict and object
            if isinstance(email, dict):
                urgency = email.get('urgency_level', 'low')
                familiarity = email.get('sender_familiarity', 'unfamiliar')
            else:
                urgency = getattr(email, 'urgency_level', 'low')
                familiarity = getattr(email, 'sender_familiarity', 'unfamiliar')
            
            if urgency == 'high':
                urgency_high_clicks.append(is_click)
            else:
                urgency_low_clicks.append(is_click)
            
            if familiarity == 'familiar':
                familiar_clicks.append(is_click)
            else:
                unfamiliar_clicks.append(is_click)
        
        result = {}
        
        # Calculate AI urgency effect
        if urgency_high_clicks and urgency_low_clicks:
            ai_high_rate = float(np.mean(urgency_high_clicks))
            ai_low_rate = float(np.mean(urgency_low_clicks))
            ai_urgency_effect = ai_high_rate - ai_low_rate
            human_urgency_effect = to_python_float(persona.email_interaction_effects.urgency_effect)
            
            result['urgency_ai'] = round(ai_urgency_effect, 3)
            result['urgency_human'] = human_urgency_effect
            
            if abs(human_urgency_effect) > 0.01:
                urgency_preservation = max(0.0, 1.0 - abs(ai_urgency_effect - human_urgency_effect) / abs(human_urgency_effect))
            else:
                urgency_preservation = 1.0 if abs(ai_urgency_effect) < 0.05 else 0.5
            result['urgency_r'] = round(float(urgency_preservation), 3)
        else:
            result['urgency_ai'] = None
            result['urgency_human'] = to_python_float(persona.email_interaction_effects.urgency_effect)
            result['urgency_r'] = None
        
        # Calculate AI familiarity effect
        if familiar_clicks and unfamiliar_clicks:
            ai_fam_rate = float(np.mean(familiar_clicks))
            ai_unfam_rate = float(np.mean(unfamiliar_clicks))
            ai_fam_effect = ai_fam_rate - ai_unfam_rate
            human_fam_effect = to_python_float(persona.email_interaction_effects.familiarity_effect)
            
            result['familiarity_ai'] = round(ai_fam_effect, 3)
            result['familiarity_human'] = human_fam_effect
            
            if abs(human_fam_effect) > 0.01:
                fam_preservation = max(0.0, 1.0 - abs(ai_fam_effect - human_fam_effect) / abs(human_fam_effect))
            else:
                fam_preservation = 1.0 if abs(ai_fam_effect) < 0.05 else 0.5
            result['familiarity_r'] = round(float(fam_preservation), 3)
        else:
            result['familiarity_ai'] = None
            result['familiarity_human'] = to_python_float(persona.email_interaction_effects.familiarity_effect)
            result['familiarity_r'] = None
        
        result['framing_r'] = None  # Not implemented yet
        
        return result
    
    def aggregate_by_condition(
        self,
        trials: List[SimulationTrial],
        emails: Dict[str, Any]
    ) -> Dict[Tuple, ConditionResult]:
        """Aggregate trials by persona-model-prompt condition."""
        grouped = defaultdict(list)
        for trial in trials:
            key = (trial.persona_id, trial.model_id, trial.prompt_config)
            grouped[key].append(trial)
        
        results = {}
        
        for (persona_id, model_id, prompt_config), condition_trials in grouped.items():
            valid = [t for t in condition_trials if t.parse_success]
            
            if not valid:
                continue
            
            # Aggregate by email
            email_grouped = defaultdict(list)
            for t in valid:
                email_grouped[t.email_id].append(t)
            
            email_results = {}
            for email_id, email_trials in email_grouped.items():
                n_click = sum(1 for t in email_trials if t.action == ActionType.CLICK)
                n_report = sum(1 for t in email_trials if t.action == ActionType.REPORT)
                n_ignore = sum(1 for t in email_trials if t.action == ActionType.IGNORE)
                n_total = len(email_trials)
                
                counts = {
                    ActionType.CLICK: n_click,
                    ActionType.REPORT: n_report,
                    ActionType.IGNORE: n_ignore
                }
                modal = max(counts, key=counts.get)
                
                email_results[email_id] = EmailLevelResult(
                    email_id=email_id,
                    n_trials=n_total,
                    click_rate=n_click / n_total,
                    report_rate=n_report / n_total,
                    ignore_rate=n_ignore / n_total,
                    modal_action=modal
                )
            
            n_total = len(valid)
            n_click = sum(1 for t in valid if t.action == ActionType.CLICK)
            n_report = sum(1 for t in valid if t.action == ActionType.REPORT)
            n_ignore = sum(1 for t in valid if t.action == ActionType.IGNORE)
            
            # Convert numpy types to Python types
            latencies = [t.model_latency_ms for t in valid]
            mean_latency = to_python_float(np.mean(latencies)) if latencies else 0.0
            
            results[(persona_id, model_id, prompt_config)] = ConditionResult(
                persona_id=persona_id,
                model_id=model_id,
                prompt_config=prompt_config,
                n_trials=n_total,
                n_emails=len(email_results),
                ai_click_rate=n_click / n_total,
                ai_report_rate=n_report / n_total,
                ai_ignore_rate=n_ignore / n_total,
                email_results=email_results,
                parse_success_rate=len(valid) / len(condition_trials),
                mean_latency_ms=mean_latency,
                total_cost=sum(t.cost_usd for t in condition_trials)
            )
        
        return results
    
    def compare_models(
        self,
        trials: List[SimulationTrial],
        personas: Dict[str, Persona]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all models across all personas.
        Returns dict with Python native types for JSON serialization.
        """
        model_metrics = defaultdict(lambda: {
            'fidelity_scores': [],
            'click_rate_errors': [],
            'total_cost': 0.0,
            'total_trials': 0,
            'parse_success_count': 0,
            'latencies': []
        })
        
        # Group by model
        model_trials = defaultdict(list)
        for trial in trials:
            model_trials[trial.model_id].append(trial)
        
        for model_id, m_trials in model_trials.items():
            valid = [t for t in m_trials if t.parse_success]
            
            model_metrics[model_id]['total_trials'] = len(m_trials)
            model_metrics[model_id]['total_cost'] = sum(t.cost_usd for t in m_trials)
            model_metrics[model_id]['parse_success_count'] = len(valid)
            model_metrics[model_id]['latencies'] = [t.model_latency_ms for t in valid]
            
            # Calculate fidelity per persona
            persona_trials = defaultdict(list)
            for t in valid:
                persona_trials[t.persona_id].append(t)
            
            for persona_id, p_trials in persona_trials.items():
                persona = personas.get(persona_id)
                if not persona:
                    continue
                
                n_click = sum(1 for t in p_trials if t.action == ActionType.CLICK)
                ai_click_rate = n_click / len(p_trials)
                human_click_rate = to_python_float(persona.behavioral_statistics.phishing_click_rate)
                
                error = abs(ai_click_rate - human_click_rate)
                model_metrics[model_id]['click_rate_errors'].append(error)
                
                if human_click_rate > 0:
                    fidelity = max(0.0, 1.0 - (error / human_click_rate))
                    model_metrics[model_id]['fidelity_scores'].append(fidelity)
        
        # Summarize - CONVERT ALL NUMPY TYPES TO PYTHON TYPES
        summary = {}
        for model_id, metrics in model_metrics.items():
            latencies = metrics['latencies']
            fidelity_scores = metrics['fidelity_scores']
            click_rate_errors = metrics['click_rate_errors']
            total_trials = metrics['total_trials']
            parse_success_count = metrics['parse_success_count']
            
            summary[model_id] = {
                'mean_fidelity': to_python_float(np.mean(fidelity_scores)) if fidelity_scores else 0.0,
                'std_fidelity': to_python_float(np.std(fidelity_scores)) if len(fidelity_scores) > 1 else 0.0,
                'min_fidelity': to_python_float(np.min(fidelity_scores)) if fidelity_scores else 0.0,
                'max_fidelity': to_python_float(np.max(fidelity_scores)) if fidelity_scores else 0.0,
                'mean_click_error': to_python_float(np.mean(click_rate_errors)) if click_rate_errors else 1.0,
                'total_cost': to_python_float(metrics['total_cost']),
                'total_trials': to_python_int(total_trials),
                'parse_success_rate': to_python_float(parse_success_count / total_trials) if total_trials else 0.0,
                'mean_latency_ms': to_python_float(np.mean(latencies)) if latencies else 0.0,
                'p50_latency_ms': to_python_float(np.percentile(latencies, 50)) if latencies else 0.0,
                'p95_latency_ms': to_python_float(np.percentile(latencies, 95)) if latencies else 0.0,
                'p99_latency_ms': to_python_float(np.percentile(latencies, 99)) if latencies else 0.0,
                'cost_per_decision': to_python_float(metrics['total_cost'] / total_trials) if total_trials else 0.0,
                'n_personas_tested': len(fidelity_scores)
            }
        
        return summary
    
    def calculate_pareto_frontier(
        self,
        model_comparison: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Calculate the Pareto frontier of models optimizing fidelity vs cost."""
        points = []
        for model_id, metrics in model_comparison.items():
            points.append({
                'model_id': model_id,
                'fidelity': metrics['mean_fidelity'],
                'cost': metrics['cost_per_decision']
            })
        
        # Sort by fidelity descending
        points.sort(key=lambda x: x['fidelity'], reverse=True)
        
        pareto_frontier = []
        min_cost_so_far = float('inf')
        
        for point in points:
            if point['cost'] < min_cost_so_far:
                pareto_frontier.append(point['model_id'])
                min_cost_so_far = point['cost']
        
        return pareto_frontier
    
    def find_boundary_conditions(
        self,
        trials: List[SimulationTrial],
        personas: Dict[str, Persona],
        emails: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify conditions where AI systematically fails."""
        boundaries = []
        
        # Aggregate by persona
        persona_trials = defaultdict(list)
        for trial in trials:
            if trial.parse_success:
                persona_trials[trial.persona_id].append(trial)
        
        for persona_id, p_trials in persona_trials.items():
            persona = personas.get(persona_id)
            if not persona:
                continue
            
            n_click = sum(1 for t in p_trials if t.action == ActionType.CLICK)
            ai_click_rate = n_click / len(p_trials) if p_trials else 0
            human_click_rate = to_python_float(persona.behavioral_statistics.phishing_click_rate)
            
            # Check 1: Impulsive personas - AI under-clicking (over-deliberation)
            if persona.cognitive_style.value == 'impulsive':
                if human_click_rate - ai_click_rate > 0.15:
                    boundaries.append({
                        'type': 'over_deliberation',
                        'persona_id': persona_id,
                        'persona_name': persona.name,  # ADDED
                        'description': f'AI ({ai_click_rate:.1%}) clicks much less than impulsive human persona ({human_click_rate:.1%})',
                        'severity': 'high',
                        'human_pattern': f'{human_click_rate:.1%} click rate with fast decisions',
                        'ai_pattern': f'{ai_click_rate:.1%} click rate, appears to over-analyze',
                        'discrepancy': round(human_click_rate - ai_click_rate, 3),
                        'recommendation': 'Use shorter prompts or reduce reasoning traces for this persona'
                    })
            
            # Check 2: AI over-clicking (too trusting)
            if ai_click_rate - human_click_rate > 0.15:
                boundaries.append({
                    'type': 'over_clicking',
                    'persona_id': persona_id,
                    'persona_name': persona.name,  # ADDED
                    'description': f'AI ({ai_click_rate:.1%}) clicks much more than human persona ({human_click_rate:.1%})',
                    'severity': 'medium',
                    'human_pattern': f'{human_click_rate:.1%} click rate',
                    'ai_pattern': f'{ai_click_rate:.1%} click rate, too trusting',
                    'discrepancy': round(ai_click_rate - human_click_rate, 3),
                    'recommendation': 'Add more suspicion cues to persona prompt'
                })
            
            # Check 3: Urgency susceptibility not preserved
            urgency_effect = to_python_float(persona.email_interaction_effects.urgency_effect)
            if urgency_effect > 0.1:
                high_urgency = [t for t in p_trials 
                               if self._get_email_attr(emails.get(t.email_id), 'urgency_level') == 'high']
                low_urgency = [t for t in p_trials 
                              if self._get_email_attr(emails.get(t.email_id), 'urgency_level') == 'low']
                
                if high_urgency and low_urgency:
                    ai_high_click = sum(1 for t in high_urgency if t.action == ActionType.CLICK) / len(high_urgency)
                    ai_low_click = sum(1 for t in low_urgency if t.action == ActionType.CLICK) / len(low_urgency)
                    ai_effect = ai_high_click - ai_low_click
                    
                    if urgency_effect - ai_effect > 0.10:
                        boundaries.append({
                            'type': 'emotional_unresponsive',
                            'persona_id': persona_id,
                            'persona_name': persona.name,  # ADDED
                            'description': f'AI urgency effect ({ai_effect:.1%}) much weaker than human ({urgency_effect:.1%})',
                            'severity': 'medium',
                            'human_pattern': f'+{urgency_effect:.1%} click rate increase for urgent emails',
                            'ai_pattern': f'+{ai_effect:.1%} effect - AI doesn\'t feel urgency pressure',
                            'discrepancy': round(urgency_effect - ai_effect, 3),
                            'recommendation': 'Emphasize urgency susceptibility in persona prompt with examples'
                        })
            
            # Check 4: Trust/familiarity not preserved
            familiarity_effect = to_python_float(persona.email_interaction_effects.familiarity_effect)
            if familiarity_effect > 0.1:
                familiar = [t for t in p_trials 
                           if self._get_email_attr(emails.get(t.email_id), 'sender_familiarity') == 'familiar']
                unfamiliar = [t for t in p_trials 
                             if self._get_email_attr(emails.get(t.email_id), 'sender_familiarity') == 'unfamiliar']
                
                if familiar and unfamiliar:
                    ai_fam_click = sum(1 for t in familiar if t.action == ActionType.CLICK) / len(familiar)
                    ai_unfam_click = sum(1 for t in unfamiliar if t.action == ActionType.CLICK) / len(unfamiliar)
                    ai_effect = ai_fam_click - ai_unfam_click
                    
                    if familiarity_effect - ai_effect > 0.10:
                        boundaries.append({
                            'type': 'trust_calibration',
                            'persona_id': persona_id,
                            'persona_name': persona.name,  # ADDED
                            'description': f'AI familiarity effect ({ai_effect:.1%}) weaker than human ({familiarity_effect:.1%})',
                            'severity': 'low',
                            'human_pattern': f'+{familiarity_effect:.1%} trust for familiar senders',
                            'ai_pattern': f'+{ai_effect:.1%} effect',
                            'discrepancy': round(familiarity_effect - ai_effect, 3),
                            'recommendation': 'Add explicit trust propensity cues to prompt'
                        })
        
        return boundaries
    
    def _get_email_attr(self, email: Any, attr: str) -> Optional[str]:
        """Safely get email attribute from dict or object."""
        if email is None:
            return None
        if isinstance(email, dict):
            return email.get(attr)
        return getattr(email, attr, None)