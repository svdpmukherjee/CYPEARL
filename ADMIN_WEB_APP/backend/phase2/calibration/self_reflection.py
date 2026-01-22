"""
CYPEARL Phase 2 - Self-Reflection Engine

When calibration accuracy is below threshold, this engine asks the LLM
to analyze its failures and suggest prompt improvements.

The key insight: LLMs can reason about their own prompts and identify
what information is missing or misleading.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .calibration_engine import CalibrationResult, CalibrationTrial


@dataclass
class PromptSuggestion:
    """A suggested improvement to the prompt."""
    category: str  # behavioral_stats, trait_description, reasoning_examples, framing
    issue_identified: str
    suggested_change: str
    confidence: str  # high, medium, low
    affected_trials: int  # How many failures this might fix
    example_failure: Optional[str] = None


@dataclass
class ReflectionResult:
    """Result of self-reflection analysis."""
    persona_id: str
    model_id: str
    prompt_config: str

    # Analysis
    failure_patterns: Dict[str, Any] = field(default_factory=dict)
    root_cause_analysis: str = ""

    # Suggestions
    suggestions: List[PromptSuggestion] = field(default_factory=list)

    # New prompt (if generated)
    improved_prompt: Optional[str] = None

    # Metadata
    reflection_model: str = ""  # Which model did the reflection
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SelfReflectionEngine:
    """
    Engine for LLM self-reflection on calibration failures.

    When prompted correctly, LLMs can:
    1. Analyze patterns in their failures
    2. Identify what information was missing
    3. Suggest specific prompt improvements
    """

    # Reflection prompt template
    REFLECTION_PROMPT = """You are an expert in prompt engineering for behavioral simulation.

I'm trying to make LLMs simulate human personas for phishing susceptibility research.
The LLM was given a persona description and asked to predict how that person would respond to emails.

CALIBRATION RESULTS:
- Accuracy: {accuracy:.1%} (target: 80%+)
- The LLM predicted {n_correct}/{n_trials} human responses correctly

PERSONA USED:
Name: {persona_name}
Description: {persona_description}

BEHAVIORAL TARGETS (from real human data):
- Phishing click rate: {human_click_rate:.1%}
- Report rate: {human_report_rate:.1%}

LLM ACTUAL BEHAVIOR:
- Click rate: {llm_click_rate:.1%} (error: {click_error:.1%})
- Report rate: {llm_report_rate:.1%} (error: {report_error:.1%})

THE PROMPT USED:
```
{prompt_snippet}
```

SPECIFIC FAILURES (where LLM prediction != human action):
{failure_examples}

ANALYZE AND SUGGEST IMPROVEMENTS:

1. PATTERN ANALYSIS: What patterns do you see in the failures?
   - Is the LLM over-clicking or under-clicking?
   - Does it fail more on certain email types (phishing vs legitimate)?
   - Does urgency/familiarity affect accuracy?

2. ROOT CAUSE: Why do you think the prompt failed to produce human-like behavior?
   - Is the persona description unclear?
   - Are behavioral statistics not being followed?
   - Is there conflicting information?

3. SPECIFIC SUGGESTIONS: What changes to the prompt would help?
   For each suggestion, provide:
   - CATEGORY: (behavioral_stats, trait_description, reasoning_examples, framing, temperature)
   - ISSUE: What's wrong?
   - CHANGE: What specific text should be added/modified?
   - CONFIDENCE: How confident are you this will help? (high/medium/low)

4. IMPROVED PROMPT SECTION: Write an improved version of the key prompt section that would better capture this persona's behavior.

Format your response as:
## Pattern Analysis
[your analysis]

## Root Cause
[your analysis]

## Suggestions
### Suggestion 1
- Category: [category]
- Issue: [what's wrong]
- Change: [specific change]
- Confidence: [high/medium/low]

[repeat for each suggestion]

## Improved Prompt Section
```
[your improved prompt text]
```
"""

    FAILURE_EXAMPLE_TEMPLATE = """
Trial {i}:
- Email type: {email_type}, Urgency: {urgency}, Familiarity: {familiarity}
- Human action: {human_action}
- LLM predicted: {llm_action}
- LLM reasoning: "{llm_reasoning}"
"""

    def __init__(self, router):
        """
        Initialize reflection engine.

        Args:
            router: ProviderRouter for making LLM calls
        """
        self.router = router

    async def reflect(
        self,
        calibration_result: CalibrationResult,
        original_prompt: str,
        persona_description: str,
        reflection_model: str = "claude-sonnet-4"
    ) -> ReflectionResult:
        """
        Analyze calibration failures and suggest improvements.

        Args:
            calibration_result: The failed calibration result
            original_prompt: The prompt that was used
            persona_description: The persona's description
            reflection_model: Which model to use for reflection (default: Claude)

        Returns:
            ReflectionResult with analysis and suggestions
        """
        from phase2.providers.base import LLMRequest

        result = ReflectionResult(
            persona_id=calibration_result.persona_id,
            model_id=calibration_result.model_id,
            prompt_config=calibration_result.prompt_config,
            reflection_model=reflection_model
        )

        # Build failure examples
        failure_examples = self._format_failure_examples(
            calibration_result.failed_trials[:10]  # Limit to 10 examples
        )

        # Build reflection prompt
        prompt = self.REFLECTION_PROMPT.format(
            accuracy=calibration_result.accuracy,
            n_correct=calibration_result.n_correct,
            n_trials=calibration_result.n_trials,
            persona_name=calibration_result.persona_name,
            persona_description=persona_description,
            human_click_rate=calibration_result.human_click_rate,
            human_report_rate=calibration_result.human_report_rate,
            llm_click_rate=calibration_result.llm_click_rate,
            llm_report_rate=calibration_result.llm_report_rate,
            click_error=calibration_result.click_rate_error,
            report_error=calibration_result.report_rate_error,
            prompt_snippet=original_prompt[:2000],  # Truncate for token limits
            failure_examples=failure_examples
        )

        try:
            # Make reflection request (use high-capability model)
            request = LLMRequest(
                system_prompt="You are an expert prompt engineer specializing in behavioral simulation. Analyze failures carefully and provide actionable suggestions.",
                user_prompt=prompt,
                temperature=0.3,  # Lower temperature for analysis
                max_tokens=2000
            )

            response = await self.router.complete(reflection_model, request)

            if response.success:
                # Parse the reflection response
                parsed = self._parse_reflection_response(response.content)
                result.failure_patterns = parsed.get('patterns', {})
                result.root_cause_analysis = parsed.get('root_cause', '')
                result.suggestions = parsed.get('suggestions', [])
                result.improved_prompt = parsed.get('improved_prompt')

        except Exception as e:
            result.root_cause_analysis = f"Reflection failed: {str(e)}"

        return result

    def _format_failure_examples(self, failures: List[CalibrationTrial]) -> str:
        """Format failure examples for the reflection prompt."""
        examples = []
        for i, trial in enumerate(failures, 1):
            examples.append(self.FAILURE_EXAMPLE_TEMPLATE.format(
                i=i,
                email_type=trial.email_type,
                urgency=trial.urgency_level,
                familiarity=trial.sender_familiarity,
                human_action=trial.human_action,
                llm_action=trial.llm_action,
                llm_reasoning=trial.llm_reasoning[:200] if trial.llm_reasoning else "No reasoning provided"
            ))
        return "\n".join(examples)

    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured reflection response."""
        result = {
            'patterns': {},
            'root_cause': '',
            'suggestions': [],
            'improved_prompt': None
        }

        # Extract pattern analysis
        if '## Pattern Analysis' in response:
            pattern_section = response.split('## Pattern Analysis')[1]
            if '## Root Cause' in pattern_section:
                pattern_section = pattern_section.split('## Root Cause')[0]
            result['patterns'] = {'analysis': pattern_section.strip()}

        # Extract root cause
        if '## Root Cause' in response:
            root_section = response.split('## Root Cause')[1]
            if '## Suggestions' in root_section:
                root_section = root_section.split('## Suggestions')[0]
            result['root_cause'] = root_section.strip()

        # Extract suggestions
        if '## Suggestions' in response:
            suggestions_section = response.split('## Suggestions')[1]
            if '## Improved Prompt' in suggestions_section:
                suggestions_section = suggestions_section.split('## Improved Prompt')[0]

            # Parse individual suggestions
            result['suggestions'] = self._parse_suggestions(suggestions_section)

        # Extract improved prompt
        if '## Improved Prompt' in response:
            prompt_section = response.split('## Improved Prompt')[1]
            # Extract code block
            if '```' in prompt_section:
                code_start = prompt_section.find('```') + 3
                code_end = prompt_section.find('```', code_start)
                if code_end > code_start:
                    result['improved_prompt'] = prompt_section[code_start:code_end].strip()

        return result

    def _parse_suggestions(self, section: str) -> List[PromptSuggestion]:
        """Parse suggestion blocks from the response."""
        suggestions = []

        # Split by "### Suggestion"
        parts = section.split('### Suggestion')

        for part in parts[1:]:  # Skip first empty part
            suggestion = PromptSuggestion(
                category='unknown',
                issue_identified='',
                suggested_change='',
                confidence='medium',
                affected_trials=0
            )

            lines = part.strip().split('\n')
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('- category:'):
                    suggestion.category = line.split(':', 1)[1].strip()
                elif line_lower.startswith('- issue:'):
                    suggestion.issue_identified = line.split(':', 1)[1].strip()
                elif line_lower.startswith('- change:'):
                    suggestion.suggested_change = line.split(':', 1)[1].strip()
                elif line_lower.startswith('- confidence:'):
                    conf = line.split(':', 1)[1].strip().lower()
                    if 'high' in conf:
                        suggestion.confidence = 'high'
                    elif 'low' in conf:
                        suggestion.confidence = 'low'
                    else:
                        suggestion.confidence = 'medium'

            if suggestion.issue_identified or suggestion.suggested_change:
                suggestions.append(suggestion)

        return suggestions

    async def generate_improved_prompt(
        self,
        calibration_result: CalibrationResult,
        original_prompt: str,
        suggestions: List[PromptSuggestion],
        generation_model: str = "claude-sonnet-4"
    ) -> str:
        """
        Generate an improved prompt incorporating suggestions.

        Args:
            calibration_result: The calibration result
            original_prompt: The original prompt
            suggestions: List of suggestions to incorporate
            generation_model: Model to use for generation

        Returns:
            Improved prompt text
        """
        from phase2.providers.base import LLMRequest

        suggestions_text = "\n".join([
            f"- {s.category}: {s.suggested_change}"
            for s in suggestions
            if s.confidence in ['high', 'medium']
        ])

        prompt = f"""Rewrite this persona simulation prompt incorporating these improvements:

ORIGINAL PROMPT:
```
{original_prompt}
```

IMPROVEMENTS TO INCORPORATE:
{suggestions_text}

REQUIREMENTS:
1. Keep the same overall structure
2. Incorporate ALL the suggested improvements
3. Make the behavioral expectations clearer
4. Ensure the persona's click rate target ({calibration_result.human_click_rate:.1%}) is clearly communicated
5. Add situational variation guidance

Write the improved prompt:
"""

        request = LLMRequest(
            system_prompt="You are a prompt engineer. Rewrite prompts to improve LLM behavioral simulation accuracy.",
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=2000
        )

        response = await self.router.complete(generation_model, request)

        if response.success:
            return response.content
        else:
            return original_prompt  # Return original if generation fails

    async def iterative_improvement(
        self,
        persona,
        split_result,
        prompt_config,
        model_id: str,
        calibration_engine,
        max_iterations: int = 3,
        target_accuracy: float = 0.80
    ) -> Dict[str, Any]:
        """
        Iteratively improve prompt until accuracy threshold is met.

        Args:
            persona: The persona to calibrate
            split_result: Train/test split data
            prompt_config: Starting prompt configuration
            model_id: Model to calibrate
            calibration_engine: CalibrationEngine instance
            max_iterations: Max improvement iterations
            target_accuracy: Accuracy threshold to achieve

        Returns:
            Dict with history of improvements and final result
        """
        history = []
        current_prompt = None  # Will use default prompt builder initially

        for iteration in range(max_iterations):
            # Run calibration
            cal_result = await calibration_engine.run_calibration(
                persona, split_result, prompt_config, model_id
            )

            history.append({
                'iteration': iteration,
                'accuracy': cal_result.accuracy,
                'click_rate_error': cal_result.click_rate_error,
                'n_failures': len(cal_result.failed_trials)
            })

            # Check if we've met threshold
            if cal_result.accuracy >= target_accuracy:
                return {
                    'success': True,
                    'iterations': iteration + 1,
                    'final_accuracy': cal_result.accuracy,
                    'history': history,
                    'final_prompt': current_prompt
                }

            # Get reflection and suggestions
            reflection = await self.reflect(
                cal_result,
                current_prompt or "Default prompt builder",
                persona.description,
                model_id  # Use same model for reflection
            )

            # Generate improved prompt
            if reflection.suggestions:
                current_prompt = await self.generate_improved_prompt(
                    cal_result,
                    current_prompt or "Default prompt",
                    reflection.suggestions,
                    model_id
                )

        # Return final state even if threshold not met
        return {
            'success': False,
            'iterations': max_iterations,
            'final_accuracy': history[-1]['accuracy'] if history else 0,
            'history': history,
            'final_prompt': current_prompt,
            'message': f"Did not reach {target_accuracy:.0%} accuracy after {max_iterations} iterations"
        }
