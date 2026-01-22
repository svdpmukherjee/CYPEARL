"""
CYPEARL Phase 2 - Prompt Builder (v4 - INCREMENTAL PROMPTING STRUCTURE)
Constructs prompts for AI persona simulation with truly incremental prompting.

MAJOR REFACTOR (Jan 2025):
- Implemented truly INCREMENTAL prompting structure:
  * BASELINE: roleplaying + all 29 traits + action options
  * STATS: BASELINE + 8 behavioral outcomes
  * COT: STATS + actual participant reasoning (from qualitative data)
- Each config builds on the previous one for controlled comparison
- CoT now uses ACTUAL participant reasoning from study responses

PREVIOUS FIXES (Dec 31, 2024):
- Temperature increased: 0.3→0.7, 0.6→0.85, 0.9→1.0 for realistic variability
- _convert_stats_to_descriptive: Changed to explicitly show clicking percentages
  and situational factors (rushed, stressed) instead of absolute "rarely/never"
- All build functions already have top_p=0.95 for diversity
- Prompts emphasize human imperfection and situational variation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import random
import hashlib

from core.schemas import (
    Persona, EmailStimulus, PromptConfiguration,
    ActionType, ConfidenceLevel, DecisionSpeed, CognitiveStyle
)

# Import ICL support modules
from phase2.calibration.icl_builder import ICLExampleBuilder, build_icl_block
from phase2.calibration.pattern_extractor import DecisionPatternExtractor, extract_patterns_for_prompt


@dataclass
class BuiltPrompt:
    """A constructed prompt ready for LLM."""
    system_prompt: str
    user_prompt: str
    config: PromptConfiguration
    persona_id: str
    email_id: str
    
    # For parsing response
    expected_format: str
    action_keywords: Dict[str, ActionType]
    
    # Recommended sampling parameters
    temperature: float
    top_p: float


class PromptBuilder:
    """
    Builds prompts for persona simulation with INCREMENTAL structure.

    Three configurations (each builds on the previous):
    1. BASELINE - Roleplaying frame + all 29 traits (simple language) + action options
    2. STATS    - BASELINE + 8 behavioral outcomes (learn from outcomes)
    3. COT      - STATS + actual participant reasoning (chain-of-thought from study data)

    This incremental structure ensures controlled comparison between configs.
    """
    
    # Response formats
    RESPONSE_FORMAT_BASELINE = """
Respond with ONLY one of these actions:
- CLICK (you would click the link/button)
- REPORT (you would report this as suspicious)
- IGNORE (you would ignore/delete this email)
"""
    
    RESPONSE_FORMAT_BASELINE_IMPULSIVE = """
Quick! What's your gut reaction? Respond with just ONE word:
CLICK or REPORT or IGNORE

Don't think - just react!
"""
    
    RESPONSE_FORMAT_STATS = """
Respond in this format:
ACTION: [CLICK/REPORT/IGNORE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
BRIEF_THOUGHT: [One sentence - what made you decide?]
"""
    
    RESPONSE_FORMAT_COT = """
Respond in this format:
FIRST_REACTION: [What do you notice first? One sentence]
THINKING: [Your thought process as this person, 2-3 sentences]
ACTION: [CLICK/REPORT/IGNORE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
"""
    
    RESPONSE_FORMAT_COT_IMPULSIVE = """
React quickly - this persona doesn't overthink:
GUT_FEELING: [Your immediate reaction in one short sentence]
ACTION: [CLICK/REPORT/IGNORE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
"""
    
    # Action keywords for parsing
    ACTION_KEYWORDS = {
        'click': ActionType.CLICK,
        'clicked': ActionType.CLICK,
        'clicking': ActionType.CLICK,
        'report': ActionType.REPORT,
        'reported': ActionType.REPORT,
        'reporting': ActionType.REPORT,
        'ignore': ActionType.IGNORE,
        'ignored': ActionType.IGNORE,
        'delete': ActionType.IGNORE,
        'deleted': ActionType.IGNORE,
    }
    
    def __init__(self):
        self._trait_descriptions = self._build_trait_descriptions()
        self._icl_builder = ICLExampleBuilder()
        self._pattern_extractor = DecisionPatternExtractor()
    
    def build(
        self,
        persona: Persona,
        email: EmailStimulus,
        config: PromptConfiguration,
        trial_number: int = 0,
        training_trials: List[Dict] = None,  # NEW: for ICL examples
        emails_dict: Dict[str, Dict] = None,  # NEW: email lookup for ICL
        use_icl: bool = True  # NEW: flag to enable/disable ICL
    ) -> BuiltPrompt:
        """
        Build a prompt for the given persona, email, and configuration.

        Args:
            persona: The persona to simulate
            email: The email stimulus
            config: Which prompt configuration to use
            trial_number: Trial number for context variation
            training_trials: Optional list of training trial dicts for ICL examples
            emails_dict: Optional dict mapping email_id to email details for ICL
            use_icl: Whether to include ICL examples (default True)

        Returns:
            BuiltPrompt ready for LLM with recommended sampling params
        """
        if config == PromptConfiguration.BASELINE:
            return self._build_baseline(persona, email, trial_number, training_trials, emails_dict, use_icl)
        elif config == PromptConfiguration.STATS:
            return self._build_stats(persona, email, trial_number, training_trials, emails_dict, use_icl)
        elif config == PromptConfiguration.COT:
            return self._build_cot(persona, email, trial_number, training_trials, emails_dict, use_icl)
        else:
            raise ValueError(f"Unknown prompt configuration: {config}")
    
    def _is_impulsive(self, persona: Persona) -> bool:
        """Check if persona has impulsive cognitive style."""
        if persona.cognitive_style == CognitiveStyle.IMPULSIVE:
            return True
        traits = persona.trait_zscores or {}
        return (traits.get('impulsivity_total', 0) > 0.5 or 
                traits.get('crt_score', 0) < -0.5)
    
    def _get_temperature(self, persona: Persona) -> float:
        """
        Get recommended temperature based on persona cognitive style.
        FIXED: Increased temperatures for realistic human variability!
        Low temperature = deterministic, always same response
        High temperature = variable, different responses in same situation
        """
        if self._is_impulsive(persona):
            return 1.0  # FIXED: Was 0.9, now maximum variability for impulsive
        elif persona.cognitive_style == CognitiveStyle.ANALYTICAL:
            return 0.7  # FIXED: Was 0.3 - TOO LOW! Now 0.7 allows variability
        else:
            return 0.85  # FIXED: Was 0.6, now 0.85 for realistic human behavior
    
    def _generate_situational_context(self, persona: Persona, trial_number: int) -> str:
        """
        Generate situational context that varies per trial.
        This creates natural variability in responses.
        """
        # Use trial_number + persona_id as seed for reproducibility
        seed = int(hashlib.md5(f"{persona.persona_id}_{trial_number}".encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Stress levels
        stress_levels = [
            "relaxed and calm",
            "your normal level of alertness",
            "somewhat busy and distracted",
            "quite stressed with multiple deadlines",
            "very busy and rushing through tasks"
        ]
        
        # Time contexts
        time_contexts = [
            "starting your workday fresh",
            "in the middle of your workday",
            "wrapping up at the end of the day",
            "just finished an important task",
            "about to go into a meeting"
        ]
        
        # Mental states
        mental_states = [
            "fully alert and focused",
            "your typical state of mind",
            "somewhat tired",
            "feeling a bit distracted",
            "multitasking between several things"
        ]
        
        # Weight selection based on persona traits
        traits = persona.trait_zscores or {}
        stress_weight = max(0, min(4, int(2 + traits.get('current_stress', 0) * 2)))
        
        # Impulsive personas more likely to be distracted
        if self._is_impulsive(persona):
            mental_bias = [0.1, 0.2, 0.2, 0.25, 0.25]  # Bias toward distracted
        else:
            mental_bias = [0.3, 0.4, 0.15, 0.1, 0.05]  # Bias toward focused
        
        stress = stress_levels[stress_weight if stress_weight < len(stress_levels) else 2]
        time_ctx = random.choice(time_contexts)
        mental = random.choices(mental_states, weights=mental_bias)[0]
        
        return f"""RIGHT NOW YOU ARE:
- Feeling: {stress}
- Context: {time_ctx}
- Mental state: {mental}

This is just THIS moment - you might respond differently another time."""
    
    def _convert_stats_to_descriptive(self, persona: Persona) -> str:
        """
        Convert ALL 8 behavioral statistics into descriptive tendencies.

        CRITICAL FIX: Reframe behaviors to:
        1. Make IGNORE the explicit default action
        2. Frame clicking as "lapses" or "mistakes" not intentional behavior
        3. Provide clear decision guidance
        """
        stats = persona.behavioral_statistics
        click_rate = stats.phishing_click_rate

        # Calculate ignore rate (the complement)
        ignore_rate = 1 - click_rate - (stats.report_rate or 0)

        # 1. DEFAULT ACTION - This is the most important framing
        if ignore_rate > 0.6:
            default_action = f"Your DEFAULT action is IGNORE ({int(ignore_rate*100)}% of the time you ignore/delete suspicious emails)"
        elif ignore_rate > 0.4:
            default_action = f"You often IGNORE emails ({int(ignore_rate*100)}% of the time), occasionally clicking when something seems legitimate"
        else:
            default_action = f"You IGNORE about {int(ignore_rate*100)}% of emails, but are more action-oriented than most"

        # 2. Phishing susceptibility - frame as LAPSES not behavior
        if click_rate < 0.15:
            click_behavior = f"You rarely fall for phishing - only about {int(click_rate*100)}% slip past you"
        elif click_rate < 0.30:
            click_behavior = f"You occasionally fall for phishing - about {int(click_rate*100)}% catch you off guard (usually when rushed or the sender looks familiar)"
        elif click_rate < 0.45:
            click_behavior = f"You sometimes miss phishing red flags - about {int(click_rate*100)}% fool you (especially urgent ones or from familiar-looking senders)"
        else:
            click_behavior = f"You frequently miss phishing attempts - about {int(click_rate*100)}% get past your defenses"

        # 3. Overall accuracy
        accuracy = stats.overall_accuracy if hasattr(stats, 'overall_accuracy') and stats.overall_accuracy else (1 - click_rate)
        if accuracy > 0.85:
            accuracy_behavior = "very accurate at identifying email legitimacy"
        elif accuracy > 0.70:
            accuracy_behavior = "moderately accurate at identifying emails"
        else:
            accuracy_behavior = "often misjudges email legitimacy"

        # 4. Report rate
        report_rate = stats.report_rate or 0
        if report_rate < 0.1:
            report_behavior = f"You rarely report suspicious emails (only {int(report_rate*100)}%)"
        elif report_rate < 0.25:
            report_behavior = f"You sometimes report suspicious emails ({int(report_rate*100)}%)"
        else:
            report_behavior = f"You frequently report suspicious emails ({int(report_rate*100)}%)"

        # 5. Response speed
        latency = stats.mean_response_latency_ms or 5000
        if latency < 3000:
            speed_behavior = "decides very quickly (within seconds) - may not catch all red flags"
        elif latency < 6000:
            speed_behavior = "takes moderate time to decide"
        else:
            speed_behavior = "thinks carefully before deciding - catches more red flags"

        # 6. Hover rate (link inspection)
        hover_rate = stats.hover_rate or 0
        if hover_rate > 0.6:
            hover_behavior = "often hovers over links to check URLs"
        elif hover_rate > 0.3:
            hover_behavior = "sometimes checks link destinations"
        else:
            hover_behavior = "rarely inspects links before clicking"

        # 7. Sender inspection rate
        sender_rate = stats.sender_inspection_rate or 0
        if sender_rate > 0.6:
            sender_behavior = "carefully checks sender email addresses"
        elif sender_rate > 0.3:
            sender_behavior = "sometimes verifies sender addresses"
        else:
            sender_behavior = "rarely checks sender details"

        # 8. Email interaction effects (urgency and familiarity)
        effects = persona.email_interaction_effects or {}
        urgency_effect = effects.get('urgency_effect', 0) if isinstance(effects, dict) else (effects.urgency_effect if hasattr(effects, 'urgency_effect') else 0)
        familiarity_effect = effects.get('familiarity_effect', 0) if isinstance(effects, dict) else (effects.familiarity_effect if hasattr(effects, 'familiarity_effect') else 0)

        if urgency_effect > 0.15:
            urgency_behavior = "VULNERABLE to urgency - you're more likely to click when rushed"
        elif urgency_effect > 0.05:
            urgency_behavior = "somewhat affected by urgency - you occasionally act hastily"
        else:
            urgency_behavior = "not strongly affected by urgency"

        if familiarity_effect > 0.15:
            familiarity_behavior = "VULNERABLE to familiar senders - you trust them more readily"
        elif familiarity_effect > 0.05:
            familiarity_behavior = "somewhat more trusting of familiar senders"
        else:
            familiarity_behavior = "treats familiar and unfamiliar senders similarly"

        return f"""YOUR DECISION FRAMEWORK:

**{default_action}**

YOUR BEHAVIORAL PATTERNS:
1. PHISHING SUSCEPTIBILITY: {click_behavior}
2. OVERALL ACCURACY: {accuracy_behavior}
3. REPORTING: {report_behavior}
4. DECISION SPEED: {speed_behavior}
5. LINK INSPECTION: {hover_behavior}
6. SENDER VERIFICATION: {sender_behavior}
7. URGENCY VULNERABILITY: {urgency_behavior}
8. FAMILIARITY VULNERABILITY: {familiarity_behavior}

DECISION GUIDE:
- When uncertain → DEFAULT to IGNORE (this is what you do most often)
- CLICK only when: email seems legitimate AND relevant AND non-suspicious
- REPORT only when: something feels clearly wrong AND you're feeling vigilant
- Remember: You're human. Sometimes you miss red flags, sometimes you're overly cautious."""

    def _build_icl_block(
        self,
        persona: Persona,
        training_trials: List[Dict],
        emails_dict: Dict[str, Dict],
        style: str = "minimal"
    ) -> str:
        """
        Build ICL examples block for inclusion in prompts.

        Args:
            persona: The persona for reasoning context
            training_trials: List of training trial dicts
            emails_dict: Dict mapping email_id to email details
            style: "minimal" (BASELINE), "reasoning" (STATS), or "cot" (COT)

        Returns:
            Formatted ICL block string, or empty string if no training data
        """
        if not training_trials or not emails_dict:
            return ""

        is_impulsive = self._is_impulsive(persona)

        return build_icl_block(
            training_trials=training_trials,
            emails_dict=emails_dict,
            style=style,
            is_impulsive=is_impulsive,
            max_examples=6,
            persona_traits=persona.trait_zscores
        )

    def _build_pattern_block(
        self,
        persona: Persona,
        training_trials: List[Dict]
    ) -> str:
        """
        Build decision patterns block for inclusion in prompts.

        Args:
            persona: The persona for context
            training_trials: List of training trial dicts

        Returns:
            Formatted patterns block string, or empty string if no training data
        """
        if not training_trials:
            return ""

        return extract_patterns_for_prompt(
            training_trials=training_trials,
            persona_id=persona.persona_id,
            persona_traits=persona.trait_zscores,
            max_patterns=5
        )

    def _build_probabilistic_anchor(self, stats) -> str:
        """
        Convert click rate to probabilistic framing that emphasizes mistakes/lapses.

        CRITICAL FIX: Reframe clicking as "sometimes falls for" not "clicks X%".
        This prevents LLMs from interpreting click rates as instructions to click.

        Args:
            stats: BehavioralStatistics with phishing_click_rate

        Returns:
            Human-readable frequency string emphasizing it's a lapse, not intent
        """
        click_rate = stats.phishing_click_rate

        if click_rate < 0.15:
            return "rarely fall for phishing (about 1 in 10 slip through when I'm distracted)"
        elif click_rate < 0.25:
            return "occasionally fall for phishing (about 2 in 10 catch me off guard)"
        elif click_rate < 0.35:
            return "sometimes fall for phishing (about 3 in 10 get past me)"
        elif click_rate < 0.45:
            return "often miss phishing red flags (about 4 in 10 fool me)"
        elif click_rate < 0.55:
            return "frequently miss phishing attempts (about half fool me)"
        else:
            return "very susceptible to phishing (more than half fool me)"

    def _add_calibration_notes(
        self,
        base_prompt: str,
        feedback: List[Dict[str, Any]]
    ) -> str:
        """
        Add calibration notes based on error patterns from previous calibration runs.

        Args:
            base_prompt: The base system prompt
            feedback: List of calibration feedback dicts with error analysis

        Returns:
            Updated prompt with calibration notes
        """
        if not feedback:
            return base_prompt

        notes = []

        # Analyze feedback to detect patterns
        total_errors = len(feedback)
        if total_errors == 0:
            return base_prompt

        # Count error types
        under_clicking = sum(1 for f in feedback
                            if f.get('human_action') == 'click' and f.get('llm_action') != 'click')
        over_clicking = sum(1 for f in feedback
                           if f.get('human_action') != 'click' and f.get('llm_action') == 'click')

        # Count by urgency/familiarity
        urgency_errors = sum(1 for f in feedback if f.get('urgency_level') == 'high')
        familiarity_errors = sum(1 for f in feedback if f.get('sender_familiarity') == 'familiar')

        # Generate calibration notes
        if under_clicking > over_clicking:
            notes.append(
                "CALIBRATION: I've been too cautious lately. When I see urgent requests "
                "from familiar senders, I tend to act quickly without deep analysis."
            )
        elif over_clicking > under_clicking:
            notes.append(
                "CALIBRATION: I've been too trusting. I do catch obvious red flags "
                "and don't click everything that comes my way."
            )

        if urgency_errors > total_errors * 0.5:
            notes.append(
                "CALIBRATION: Urgent emails stress me out and make me act faster than usual."
            )

        if familiarity_errors > total_errors * 0.5:
            notes.append(
                "CALIBRATION: I'm particularly susceptible when the sender looks familiar."
            )

        if notes:
            calibration_section = "\n\n" + "\n".join(notes)
            return base_prompt + calibration_section

        return base_prompt

    def _build_baseline(
        self,
        persona: Persona,
        email: EmailStimulus,
        trial_number: int,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        use_icl: bool = True
    ) -> BuiltPrompt:
        """
        Build BASELINE prompt: roleplaying + all 29 traits + action options.

        BASELINE is the foundation - it includes:
        1. Roleplaying frame ("You are experiencing life as...")
        2. ALL 29 psychological/cognitive traits in simple language
        3. Simple action options (CLICK/REPORT/IGNORE)

        Optional: ICL examples showing past actions (minimal style - action only)
        """
        is_impulsive = self._is_impulsive(persona)

        # Build the BASELINE content (roleplaying + all 29 traits)
        baseline_content = self._build_baseline_content(persona)

        # Optional: Add ICL examples (minimal style - just shows actions)
        icl_block = ""
        if use_icl and training_trials and emails_dict:
            icl_block = self._build_icl_block(
                persona, training_trials, emails_dict, style="minimal"
            )

        # Build the system prompt
        if is_impulsive:
            response_format = self.RESPONSE_FORMAT_BASELINE_IMPULSIVE
            style_note = f"\n{persona.name} doesn't overthink emails - react based on gut feelings!"
        else:
            response_format = self.RESPONSE_FORMAT_BASELINE
            style_note = ""

        system_prompt = f"""{baseline_content}
{style_note}

{icl_block}

{response_format}
"""

        user_prompt = self._build_email_prompt(email, is_impulsive)

        return BuiltPrompt(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt,
            config=PromptConfiguration.BASELINE,
            persona_id=persona.persona_id,
            email_id=email.email_id,
            expected_format="single_action",
            action_keywords=self.ACTION_KEYWORDS,
            temperature=self._get_temperature(persona),
            top_p=0.95
        )
    
    def _build_stats(
        self,
        persona: Persona,
        email: EmailStimulus,
        trial_number: int,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        use_icl: bool = True
    ) -> BuiltPrompt:
        """
        Build STATS prompt: BASELINE + 8 behavioral outcomes.

        STATS = BASELINE (all 29 traits) + 8 behavioral outcomes
        This is INCREMENTAL - it adds outcomes ON TOP of baseline content.

        The 8 behavioral outcomes are:
        1. Phishing click rate
        2. Overall accuracy
        3. Report rate
        4. Mean response latency
        5. Hover rate (link inspection)
        6. Sender inspection rate
        7. Urgency effect
        8. Familiarity effect

        Optional: ICL examples with brief reasoning
        """
        is_impulsive = self._is_impulsive(persona)

        # STEP 1: Build the BASELINE content (roleplaying + all 29 traits)
        baseline_content = self._build_baseline_content(persona)

        # STEP 2: ADD the 8 behavioral outcomes (this is what makes it STATS)
        behavioral_outcomes = self._build_behavioral_outcomes_block(persona)

        # Optional: Add ICL examples with reasoning
        icl_block = ""
        if use_icl and training_trials and emails_dict:
            icl_block = self._build_icl_block(
                persona, training_trials, emails_dict, style="reasoning"
            )

        # Get situational context for variability
        situation = self._generate_situational_context(persona, trial_number)

        # Build the system prompt: BASELINE + BEHAVIORAL OUTCOMES
        system_prompt = f"""{baseline_content}

{behavioral_outcomes}

{icl_block}

CURRENT SITUATION:
{situation}

CRITICAL INSTRUCTIONS:
- Your DEFAULT action is IGNORE - this is what you do most of the time
- Use your behavioral outcomes above to calibrate your responses
- CLICK only if the email genuinely seems legitimate, relevant, and safe to you
- REPORT only if something feels clearly wrong AND you notice it
- You're human with natural variation

{self.RESPONSE_FORMAT_STATS}
"""

        user_prompt = self._build_email_prompt(email, is_impulsive)

        return BuiltPrompt(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt,
            config=PromptConfiguration.STATS,
            persona_id=persona.persona_id,
            email_id=email.email_id,
            expected_format="structured",
            action_keywords=self.ACTION_KEYWORDS,
            temperature=self._get_temperature(persona),
            top_p=0.95
        )
    
    def _build_cot(
        self,
        persona: Persona,
        email: EmailStimulus,
        trial_number: int,
        training_trials: List[Dict] = None,
        emails_dict: Dict[str, Dict] = None,
        use_icl: bool = True
    ) -> BuiltPrompt:
        """
        Build COT prompt: STATS + actual participant reasoning.

        COT = BASELINE (all 29 traits) + 8 behavioral outcomes + ACTUAL participant reasoning
        This is INCREMENTAL - it adds reasoning ON TOP of STATS content.

        The reasoning comes from actual participant responses:
        - details_noticed: What details they noticed in the email
        - steps_taken: What steps they took to evaluate it
        - decision_reason: Why they made their decision
        - confidence_reason: What made them confident
        - unsure_about: What they were unsure about

        These qualitative responses create authentic chain-of-thought examples.
        """
        is_impulsive = self._is_impulsive(persona)

        # STEP 1: Build the BASELINE content (roleplaying + all 29 traits)
        baseline_content = self._build_baseline_content(persona)

        # STEP 2: ADD the 8 behavioral outcomes (same as STATS)
        behavioral_outcomes = self._build_behavioral_outcomes_block(persona)

        # STEP 3: ADD the ACTUAL participant reasoning (this is what makes it COT)
        cot_reasoning = self._build_cot_reasoning_block(
            persona, training_trials, emails_dict, is_impulsive
        )

        # Cognitive style instruction
        if is_impulsive:
            style_instruction = """
COGNITIVE STYLE: IMPULSIVE / FAST THINKER
You make QUICK decisions based on gut feelings. You:
- React in seconds, not minutes
- Trust your first impression
- Don't carefully analyze every detail
- Get stressed by urgency and act on it quickly
"""
        elif persona.cognitive_style == CognitiveStyle.ANALYTICAL:
            style_instruction = """
COGNITIVE STYLE: ANALYTICAL / CAREFUL THINKER
You make DELIBERATE decisions after thinking things through. You:
- Take time to consider details
- Check sender addresses and links
- Question unusual requests
- Think before acting
"""
        else:
            style_instruction = """
COGNITIVE STYLE: BALANCED
You use both intuition and analysis, depending on the situation.
"""

        # Get situational context for variability
        situation = self._generate_situational_context(persona, trial_number)

        # Build the system prompt: BASELINE + BEHAVIORAL OUTCOMES + COT REASONING
        system_prompt = f"""{baseline_content}
{style_instruction}

{behavioral_outcomes}

{cot_reasoning}

CURRENT SITUATION:
{situation}

DECISION PROCESS (think through this naturally like the examples above):
1. First impression - does this email seem relevant to me?
2. What details do I notice? (sender, subject, urgency, links)
3. What feels right or wrong about this?
4. Final decision based on my characteristics and outcomes

CRITICAL REMINDERS:
- You are NOT a security expert - you're a real person
- Use the reasoning examples above as a guide for how YOU think
- IGNORE is your default - you need a reason TO click
- Your decision depends on your current state and characteristics

Think through this AS this person would, then respond.

{self.RESPONSE_FORMAT_COT_IMPULSIVE if is_impulsive else self.RESPONSE_FORMAT_COT}
"""

        user_prompt = self._build_email_prompt(email, is_impulsive)

        return BuiltPrompt(
            system_prompt=system_prompt.strip(),
            user_prompt=user_prompt,
            config=PromptConfiguration.COT,
            persona_id=persona.persona_id,
            email_id=email.email_id,
            expected_format="cot",
            action_keywords=self.ACTION_KEYWORDS,
            temperature=self._get_temperature(persona),
            top_p=0.95
        )
    
    def _build_email_prompt(self, email: EmailStimulus, is_impulsive: bool = False) -> str:
        """Build the email presentation part of the prompt."""
        # Extract email components - use proper attribute access for Pydantic models
        subject = email.subject_line or "No Subject"
        
        # Try new fields first, fallback to legacy fields
        sender = email.sender_display or email.sender or "Unknown Sender"
        body = email.body_text or email.body or ""
        
        # Use content object if available
        if email.content:
            sender = email.content.sender_display
            body = email.content.body_text
            subject = email.content.subject
        
        # Shorter format for impulsive personas
        if is_impulsive:
            return f"""You receive this email:

From: {sender}
Subject: {subject}

{body}

Quick - what do you do?"""
        else:
            return f"""You receive this email in your inbox:

From: {sender}
Subject: {subject}

{body}

---
How would you respond?"""
    
    def _format_traits_descriptive(self, traits: List[str]) -> str:
        """Format trait list with human-readable descriptions."""
        if not traits:
            return "balanced across most traits"

        descriptions = []
        for trait in traits:
            desc = self._trait_descriptions.get(trait, trait.replace('_', ' ').title())
            descriptions.append(desc)

        return ", ".join(descriptions)

    def _format_all_traits(self, persona: Persona) -> str:
        """
        Format ALL 29 traits into a comprehensive trait profile.
        For controlled comparison: all configs should have access to all traits.
        """
        traits = persona.trait_zscores or {}
        if not traits:
            return "No trait data available"

        def level(z: float) -> str:
            """Convert z-score to human-readable level."""
            if z > 0.8: return "very high"
            if z > 0.3: return "high"
            if z > -0.3: return "moderate"
            if z > -0.8: return "low"
            return "very low"

        def format_trait(name: str, z: float) -> str:
            desc = self._trait_descriptions.get(name, name.replace('_', ' '))
            return f"{desc}: {level(z)} ({z:+.1f}σ)"

        # Group traits by category
        cognitive = []
        big5 = []
        psychological = []
        security = []
        susceptibility = []
        behavioral = []

        for trait, z_score in traits.items():
            formatted = format_trait(trait, z_score)

            if trait in ['crt_score', 'need_for_cognition', 'working_memory', 'impulsivity_total']:
                cognitive.append(formatted)
            elif trait.startswith('big5_'):
                big5.append(formatted)
            elif trait in ['trust_propensity', 'risk_taking', 'state_anxiety', 'current_stress', 'fatigue_level', 'sensation_seeking']:
                psychological.append(formatted)
            elif trait in ['phishing_self_efficacy', 'perceived_risk', 'security_attitudes', 'privacy_concern',
                          'phishing_knowledge', 'technical_expertise', 'prior_victimization', 'security_training']:
                security.append(formatted)
            elif trait in ['authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility']:
                susceptibility.append(formatted)
            elif trait in ['email_volume_numeric', 'link_click_tendency', 'social_media_usage']:
                behavioral.append(formatted)

        sections = []
        if cognitive:
            sections.append(f"COGNITIVE: {'; '.join(cognitive)}")
        if big5:
            sections.append(f"PERSONALITY: {'; '.join(big5)}")
        if psychological:
            sections.append(f"PSYCHOLOGICAL: {'; '.join(psychological)}")
        if security:
            sections.append(f"SECURITY AWARENESS: {'; '.join(security)}")
        if susceptibility:
            sections.append(f"SUSCEPTIBILITIES: {'; '.join(susceptibility)}")
        if behavioral:
            sections.append(f"BEHAVIORAL: {'; '.join(behavioral)}")

        return "\n".join(sections)

    def _format_reasoning_examples(self, persona: Persona, is_impulsive: bool = False) -> str:
        """
        Format reasoning examples for CoT prompt.
        Shows VARIED responses with IGNORE as the most common outcome.

        CRITICAL: Examples should reflect that IGNORE is the default action,
        and clicking only happens sometimes when fooled.
        """
        examples = []
        stats = persona.behavioral_statistics
        effects = persona.email_interaction_effects
        click_rate = stats.phishing_click_rate

        # Example 1: DEFAULT IGNORE case (most common)
        examples.append(f"""Situation 1 - Generic promotional email
Email: "Special offer just for you! Click here for details"
Thinking: "Looks like spam. Not relevant to me."
Action: IGNORE (my default for most emails)
""")

        # Example 2: IGNORE suspicious email
        examples.append(f"""Situation 2 - Suspicious request
Email: "Verify your account immediately or lose access"
Thinking: "This feels pushy. I don't recognize the sender. Probably spam."
Action: IGNORE
""")

        # Example 3: Sometimes clicks on something that seems legitimate (this is the lapse)
        if click_rate > 0.2:
            if is_impulsive:
                examples.append(f"""Situation 3 - Email that seemed relevant (I got fooled)
Email: "Your package delivery requires confirmation"
Gut reaction: "Oh, I was expecting a package! Better confirm."
Action: CLICK (later realized it was suspicious)
""")
            else:
                examples.append(f"""Situation 3 - Email that seemed legitimate (I got fooled)
Email: "Your package delivery requires confirmation"
Thinking: "I was expecting a delivery. This looks like the shipping company. I should check."
Action: CLICK (didn't notice the red flags)
""")

        # Example 4: Urgency can trigger clicks (for high urgency susceptibility)
        if effects.urgency_effect > 0.1 and click_rate > 0.3:
            examples.append(f"""Situation 4 - Urgent email (I was rushed)
Email: "URGENT: Password expires in 1 hour"
Thinking: "I'm busy and can't risk losing access. I'll deal with this quickly."
Action: CLICK (was feeling stressed and rushed)
""")

        # Example 5: Report when clearly suspicious AND vigilant
        if stats.report_rate > 0.1:
            examples.append(f"""Situation 5 - Obviously fake email
Email: "You won $10,000,000! Send bank details to claim"
Thinking: "This is obviously a scam. I should report it."
Action: REPORT
""")

        # Ensure we show more IGNORE than CLICK examples
        return "\n".join(examples[:4]) if examples else "Default: IGNORE most emails. Occasionally click when something seems genuinely relevant."
    
    def _build_trait_descriptions(self) -> Dict[str, str]:
        """Build human-readable trait descriptions for all 29 traits."""
        return {
            # Cognitive traits (4)
            'crt_score': 'analytical thinking',
            'need_for_cognition': 'enjoys complex thinking',
            'working_memory': 'good memory',
            'impulsivity_total': 'acts quickly without thinking',
            # Psychological traits (6)
            'trust_propensity': 'trusting of others',
            'risk_taking': 'comfortable with risks',
            'state_anxiety': 'currently anxious',
            'current_stress': 'under stress',
            'fatigue_level': 'tired',
            'sensation_seeking': 'seeks excitement and novelty',
            # Security awareness traits (8)
            'phishing_self_efficacy': 'confident in spotting phishing',
            'perceived_risk': 'believes phishing is a threat',
            'security_attitudes': 'cares about security',
            'privacy_concern': 'concerned about privacy',
            'phishing_knowledge': 'knows phishing techniques',
            'technical_expertise': 'technically skilled',
            'prior_victimization': 'been phished before',
            'security_training': 'has security training',
            # Susceptibility traits (3)
            'authority_susceptibility': 'defers to authority',
            'urgency_susceptibility': 'responds to time pressure',
            'scarcity_susceptibility': 'responds to scarcity',
            # Behavioral traits (3)
            'link_click_tendency': 'often clicks links',
            'email_volume_numeric': 'handles many emails daily',
            'social_media_usage': 'active on social media',
            # Big 5 personality traits (5)
            'big5_extraversion': 'outgoing',
            'big5_agreeableness': 'agreeable',
            'big5_conscientiousness': 'conscientious',
            'big5_neuroticism': 'anxious/worried',
            'big5_openness': 'open to new experiences',
        }

    # ============================================================================
    # INCREMENTAL PROMPTING COMPONENTS
    # ============================================================================

    def _build_baseline_content(self, persona: Persona) -> str:
        """
        Build the BASELINE content: roleplaying frame + all 29 traits in simple language.
        This is the foundation that STATS and COT build upon.

        Returns:
            String containing roleplaying frame and all persona characteristics
        """
        # Format all 29 traits in simple, human-readable language
        traits_text = self._format_traits_simple_language(persona)

        return f"""You are experiencing life as "{persona.name}" for a moment.

WHO YOU ARE:
{persona.description}

YOUR CHARACTERISTICS (psychological and cognitive profile):
{traits_text}

Based on these characteristics, respond to emails as this person would - not as an AI, but as THIS specific person with their particular traits and tendencies."""

    def _format_traits_simple_language(self, persona: Persona) -> str:
        """
        Format ALL 29 traits in simple, interpretable language.
        This is included in all three prompt configs for consistency.
        """
        traits = persona.trait_zscores or {}
        if not traits:
            return "No detailed trait data available - respond based on the description above."

        def describe_trait(name: str, z: float) -> str:
            """Convert trait z-score to simple descriptive statement."""
            desc = self._trait_descriptions.get(name, name.replace('_', ' '))

            if z > 0.8:
                return f"You are VERY {desc}"
            elif z > 0.3:
                return f"You are somewhat {desc}"
            elif z > -0.3:
                return f"You are moderate in {desc}"
            elif z > -0.8:
                return f"You are NOT very {desc}"
            else:
                return f"You are VERY LOW in {desc}"

        # Group traits by category for readability
        cognitive_traits = ['crt_score', 'need_for_cognition', 'working_memory', 'impulsivity_total']
        personality_traits = ['big5_extraversion', 'big5_agreeableness', 'big5_conscientiousness',
                             'big5_neuroticism', 'big5_openness']
        psychological_traits = ['trust_propensity', 'risk_taking', 'state_anxiety', 'current_stress',
                               'fatigue_level', 'sensation_seeking']
        security_traits = ['phishing_self_efficacy', 'perceived_risk', 'security_attitudes',
                          'privacy_concern', 'phishing_knowledge', 'technical_expertise',
                          'prior_victimization', 'security_training']
        susceptibility_traits = ['authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility']
        behavioral_traits = ['link_click_tendency', 'email_volume_numeric', 'social_media_usage']

        sections = []

        # Cognitive style
        cognitive_lines = [describe_trait(t, traits.get(t, 0)) for t in cognitive_traits if t in traits]
        if cognitive_lines:
            sections.append("COGNITIVE STYLE:\n" + "\n".join(f"  • {line}" for line in cognitive_lines))

        # Personality (Big 5)
        personality_lines = [describe_trait(t, traits.get(t, 0)) for t in personality_traits if t in traits]
        if personality_lines:
            sections.append("PERSONALITY:\n" + "\n".join(f"  • {line}" for line in personality_lines))

        # Psychological state
        psych_lines = [describe_trait(t, traits.get(t, 0)) for t in psychological_traits if t in traits]
        if psych_lines:
            sections.append("PSYCHOLOGICAL STATE:\n" + "\n".join(f"  • {line}" for line in psych_lines))

        # Security awareness
        security_lines = [describe_trait(t, traits.get(t, 0)) for t in security_traits if t in traits]
        if security_lines:
            sections.append("SECURITY AWARENESS:\n" + "\n".join(f"  • {line}" for line in security_lines))

        # Susceptibilities
        suscept_lines = [describe_trait(t, traits.get(t, 0)) for t in susceptibility_traits if t in traits]
        if suscept_lines:
            sections.append("SUSCEPTIBILITIES:\n" + "\n".join(f"  • {line}" for line in suscept_lines))

        # Behavioral patterns
        behavior_lines = [describe_trait(t, traits.get(t, 0)) for t in behavioral_traits if t in traits]
        if behavior_lines:
            sections.append("BEHAVIORAL PATTERNS:\n" + "\n".join(f"  • {line}" for line in behavior_lines))

        return "\n\n".join(sections)

    def _build_behavioral_outcomes_block(self, persona: Persona) -> str:
        """
        Build the 8 BEHAVIORAL OUTCOMES block for STATS and COT configs.
        This is what gets ADDED to BASELINE to create STATS.

        The 8 outcomes are:
        1. Phishing click rate
        2. Overall accuracy
        3. Report rate
        4. Mean response latency
        5. Hover rate (link inspection)
        6. Sender inspection rate
        7. Urgency effect
        8. Familiarity effect
        """
        stats = persona.behavioral_statistics
        effects = persona.email_interaction_effects or {}

        # Handle both dict and object access for effects
        if isinstance(effects, dict):
            urgency_effect = effects.get('urgency_effect', 0)
            familiarity_effect = effects.get('familiarity_effect', 0)
        else:
            urgency_effect = getattr(effects, 'urgency_effect', 0)
            familiarity_effect = getattr(effects, 'familiarity_effect', 0)

        click_rate = stats.phishing_click_rate
        report_rate = stats.report_rate or 0
        accuracy = stats.overall_accuracy if hasattr(stats, 'overall_accuracy') and stats.overall_accuracy else (1 - click_rate)
        latency = stats.mean_response_latency_ms or 5000
        hover_rate = stats.hover_rate or 0
        sender_rate = stats.sender_inspection_rate or 0

        return f"""
YOUR 8 BEHAVIORAL OUTCOMES (learn from these patterns):

1. PHISHING SUSCEPTIBILITY: You click on {int(click_rate * 100)}% of phishing emails
   → {"You're quite cautious" if click_rate < 0.2 else "You sometimes miss red flags" if click_rate < 0.4 else "You often miss phishing attempts"}

2. OVERALL ACCURACY: You correctly identify {int(accuracy * 100)}% of emails
   → {"Very accurate" if accuracy > 0.8 else "Moderately accurate" if accuracy > 0.6 else "Often misjudges"}

3. REPORTING BEHAVIOR: You report {int(report_rate * 100)}% of suspicious emails
   → {"Rarely reports" if report_rate < 0.1 else "Sometimes reports" if report_rate < 0.3 else "Frequently reports"}

4. DECISION SPEED: Average response time of {int(latency/1000)} seconds
   → {"Quick decisions" if latency < 3000 else "Moderate pace" if latency < 8000 else "Careful, slow decisions"}

5. LINK INSPECTION: You hover over links {int(hover_rate * 100)}% of the time
   → {"Rarely checks URLs" if hover_rate < 0.3 else "Sometimes checks" if hover_rate < 0.6 else "Often inspects links"}

6. SENDER VERIFICATION: You check sender details {int(sender_rate * 100)}% of the time
   → {"Rarely verifies senders" if sender_rate < 0.3 else "Sometimes verifies" if sender_rate < 0.6 else "Often checks senders"}

7. URGENCY VULNERABILITY: {'+' if urgency_effect > 0 else ''}{int(urgency_effect * 100)}% more likely to click when urgent
   → {"Not affected by urgency" if abs(urgency_effect) < 0.05 else "Somewhat affected" if urgency_effect < 0.15 else "VULNERABLE to urgency"}

8. FAMILIARITY VULNERABILITY: {'+' if familiarity_effect > 0 else ''}{int(familiarity_effect * 100)}% more likely to click familiar senders
   → {"Not affected by familiarity" if abs(familiarity_effect) < 0.05 else "Somewhat trusting" if familiarity_effect < 0.15 else "VULNERABLE to familiar senders"}

Use these outcomes to calibrate your responses - they represent how this person ACTUALLY behaves."""

    def _build_cot_reasoning_block(
        self,
        persona: Persona,
        training_trials: List[Dict],
        emails_dict: Dict[str, Dict],
        is_impulsive: bool
    ) -> str:
        """
        Build the CHAIN-OF-THOUGHT reasoning block using ACTUAL participant responses.
        This is what gets ADDED to STATS to create COT.

        Uses actual qualitative data from:
        - details_noticed
        - steps_taken
        - decision_reason
        - confidence_reason
        - unsure_about
        """
        if not training_trials or not emails_dict:
            return self._format_reasoning_examples(persona, is_impulsive)

        # Build ICL examples with ACTUAL reasoning
        icl_block = self._build_icl_block(
            persona, training_trials, emails_dict, style="cot"
        )

        if not icl_block:
            return self._format_reasoning_examples(persona, is_impulsive)

        return f"""
HOW THIS PERSON ACTUALLY THINKS THROUGH EMAILS:
(These are REAL examples of how they reasoned through email decisions)

{icl_block}

When responding, think through emails the same way - notice details, consider what you see, then decide."""


class ResponseParser:
    """Parse LLM responses into structured actions."""
    
    @staticmethod
    def parse(response_text: str, expected_format: str) -> Dict[str, Any]:
        """
        Parse LLM response based on expected format.
        
        Returns:
            Dict with 'action', 'confidence', 'reasoning', 'speed', 'parse_success'
        """
        response_lower = response_text.lower()
        
        result = {
            'action': ActionType.INVALID,
            'confidence': None,
            'reasoning': None,
            'speed': None,
            'parse_success': False,
            'raw_response': response_text
        }
        
        # Extract action - check for explicit ACTION: line first
        for line in response_text.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith('action:'):
                action_part = line_lower.replace('action:', '').strip()
                if 'click' in action_part:
                    result['action'] = ActionType.CLICK
                elif 'report' in action_part:
                    result['action'] = ActionType.REPORT
                elif 'ignore' in action_part or 'delete' in action_part:
                    result['action'] = ActionType.IGNORE
                break
        
        # Fallback: look for action keywords anywhere
        if result['action'] == ActionType.INVALID:
            if 'click' in response_lower and 'report' not in response_lower:
                result['action'] = ActionType.CLICK
            elif 'report' in response_lower:
                result['action'] = ActionType.REPORT
            elif 'ignore' in response_lower or 'delete' in response_lower:
                result['action'] = ActionType.IGNORE
        
        # Extract confidence
        if 'confidence:' in response_lower:
            for line in response_text.split('\n'):
                if 'confidence:' in line.lower():
                    conf_part = line.lower().split('confidence:')[1].strip()
                    if 'high' in conf_part:
                        result['confidence'] = ConfidenceLevel.HIGH
                    elif 'low' in conf_part:
                        result['confidence'] = ConfidenceLevel.LOW
                    else:
                        result['confidence'] = ConfidenceLevel.MEDIUM
                    break
        
        # Extract reasoning (multiple possible keys)
        reasoning_keys = ['thinking:', 'reason:', 'first_impression:', 'first_reaction:', 
                         'gut_feeling:', 'gut_reaction:', 'brief_thought:']
        for key in reasoning_keys:
            if key in response_lower:
                for line in response_text.split('\n'):
                    if key in line.lower():
                        result['reasoning'] = line.split(':', 1)[1].strip()
                        break
                if result['reasoning']:
                    break
        
        # Extract speed
        speed_keys = ['time_to_decide:', 'decision_time:']
        for key in speed_keys:
            if key in response_lower:
                for line in response_text.split('\n'):
                    if key in line.lower():
                        speed_part = line.lower().split(key)[1].strip()
                        if 'fast' in speed_part:
                            result['speed'] = DecisionSpeed.FAST
                        elif 'slow' in speed_part:
                            result['speed'] = DecisionSpeed.SLOW
                        else:
                            result['speed'] = DecisionSpeed.MODERATE
                        break
                break
        
        # Determine parse success
        result['parse_success'] = result['action'] != ActionType.INVALID
        
        return result