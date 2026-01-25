"""
CYPEARL Phase 2 - Behavioral Signature Matching

Goes beyond simple action matching (click/report/ignore) to capture
the CAUSAL FACTORS behind those actions:
- Response time patterns
- Confidence levels
- Attention patterns (what they noticed)
- Reasoning depth
- Hesitation (almost chose differently)
- Influence factors (urgency, familiarity)

These are the causal mechanisms: Traits → Cognitive Process → Behavioral Signature → Action

Research contribution: "Beyond Action Matching: Behavioral Signatures for Persona Validation"
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import cosine, euclidean
import re

from core.schemas import Persona


class ReasoningDepth(str, Enum):
    """How deeply the persona analyzed the email."""
    NONE = "none"           # No reasoning provided
    SHALLOW = "shallow"     # Quick gut reaction
    MODERATE = "moderate"   # Some analysis
    DEEP = "deep"          # Careful deliberation


class AttentionFocus(str, Enum):
    """What the persona paid attention to."""
    SENDER = "sender"
    SUBJECT = "subject"
    URGENCY = "urgency"
    LINKS = "links"
    CONTENT = "content"
    FORMATTING = "formatting"
    AUTHORITY = "authority"
    NOTHING = "nothing"  # Didn't inspect carefully


class DecisionSpeed(str, Enum):
    """How quickly the persona decided."""
    INSTANT = "instant"      # < 1 second (gut reaction)
    FAST = "fast"            # 1-3 seconds
    MODERATE = "moderate"    # 3-8 seconds
    SLOW = "slow"           # 8+ seconds (deliberate)


class HesitationLevel(str, Enum):
    """Whether the persona showed signs of hesitation."""
    NONE = "none"            # Confident, no hesitation
    SLIGHT = "slight"        # Minor second-guessing
    MODERATE = "moderate"    # Considered alternatives
    HIGH = "high"           # Almost chose differently


@dataclass
class BehavioralSignature:
    """
    Complete behavioral signature for a single decision.

    This captures not just WHAT action was taken, but HOW and WHY.
    These are the causal factors that explain the action.
    """

    # The action (outcome)
    action: str  # click/report/ignore

    # Causal Factor 1: Confidence
    confidence_level: str = "medium"  # high/medium/low
    confidence_score: float = 0.5  # Numeric [0, 1]

    # Causal Factor 2: Response Time Pattern
    decision_speed: DecisionSpeed = DecisionSpeed.MODERATE
    estimated_response_time_ms: float = 5000.0

    # Causal Factor 3: Reasoning Depth
    reasoning_depth: ReasoningDepth = ReasoningDepth.MODERATE
    reasoning_word_count: int = 0
    reasoning_text: str = ""

    # Causal Factor 4: Attention Pattern (what they noticed)
    attention_pattern: List[AttentionFocus] = field(default_factory=list)
    details_mentioned: List[str] = field(default_factory=list)

    # Causal Factor 5: Hesitation
    hesitation_level: HesitationLevel = HesitationLevel.NONE
    alternative_considered: Optional[str] = None

    # Causal Factor 6: Influence Factors (situational effects)
    urgency_influence: float = 0.0  # How much urgency affected decision [-1, 1]
    familiarity_influence: float = 0.0  # How much sender familiarity affected [-1, 1]
    authority_influence: float = 0.0  # How much authority affected [-1, 1]

    # Metadata
    email_type: str = ""
    email_urgency: str = ""
    email_familiarity: str = ""

    def to_vector(self) -> np.ndarray:
        """
        Convert signature to numeric vector for distance calculations.

        Returns 12-dimensional vector:
        [action_encoded, confidence, speed, depth, n_attention_items,
         hesitation, urgency_inf, familiarity_inf, authority_inf,
         word_count_normalized, details_count, reasoning_depth_score]
        """
        # Action encoding
        action_map = {'click': 1.0, 'report': 0.0, 'ignore': 0.5}
        action_encoded = action_map.get(self.action.lower(), 0.5)

        # Speed encoding
        speed_map = {
            DecisionSpeed.INSTANT: 0.0,
            DecisionSpeed.FAST: 0.25,
            DecisionSpeed.MODERATE: 0.5,
            DecisionSpeed.SLOW: 1.0
        }
        speed_encoded = speed_map.get(self.decision_speed, 0.5)

        # Depth encoding
        depth_map = {
            ReasoningDepth.NONE: 0.0,
            ReasoningDepth.SHALLOW: 0.33,
            ReasoningDepth.MODERATE: 0.67,
            ReasoningDepth.DEEP: 1.0
        }
        depth_encoded = depth_map.get(self.reasoning_depth, 0.5)

        # Hesitation encoding
        hesitation_map = {
            HesitationLevel.NONE: 0.0,
            HesitationLevel.SLIGHT: 0.33,
            HesitationLevel.MODERATE: 0.67,
            HesitationLevel.HIGH: 1.0
        }
        hesitation_encoded = hesitation_map.get(self.hesitation_level, 0.0)

        return np.array([
            action_encoded,
            self.confidence_score,
            speed_encoded,
            depth_encoded,
            len(self.attention_pattern) / 6.0,  # Normalized by max possible
            hesitation_encoded,
            (self.urgency_influence + 1) / 2,  # Normalize from [-1,1] to [0,1]
            (self.familiarity_influence + 1) / 2,
            (self.authority_influence + 1) / 2,
            min(self.reasoning_word_count / 100, 1.0),  # Cap at 100 words
            len(self.details_mentioned) / 5.0,  # Normalize by expected max
            depth_encoded  # Include again for weighting
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence_level': self.confidence_level,
            'confidence_score': self.confidence_score,
            'decision_speed': self.decision_speed.value,
            'estimated_response_time_ms': self.estimated_response_time_ms,
            'reasoning_depth': self.reasoning_depth.value,
            'reasoning_word_count': self.reasoning_word_count,
            'reasoning_text': self.reasoning_text,
            'attention_pattern': [a.value for a in self.attention_pattern],
            'details_mentioned': self.details_mentioned,
            'hesitation_level': self.hesitation_level.value,
            'alternative_considered': self.alternative_considered,
            'urgency_influence': self.urgency_influence,
            'familiarity_influence': self.familiarity_influence,
            'authority_influence': self.authority_influence
        }


@dataclass
class SignatureComparisonResult:
    """Result of comparing two behavioral signatures."""

    # Overall similarity score [0, 1]
    overall_similarity: float = 0.0

    # Component-wise similarity
    action_match: bool = False
    confidence_similarity: float = 0.0
    speed_similarity: float = 0.0
    depth_similarity: float = 0.0
    attention_overlap: float = 0.0
    hesitation_similarity: float = 0.0
    influence_similarity: float = 0.0

    # Distance metrics
    euclidean_distance: float = 0.0
    cosine_similarity: float = 0.0

    # Diagnosis of mismatch
    mismatch_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_similarity': self.overall_similarity,
            'action_match': self.action_match,
            'confidence_similarity': self.confidence_similarity,
            'speed_similarity': self.speed_similarity,
            'depth_similarity': self.depth_similarity,
            'attention_overlap': self.attention_overlap,
            'hesitation_similarity': self.hesitation_similarity,
            'influence_similarity': self.influence_similarity,
            'euclidean_distance': self.euclidean_distance,
            'cosine_similarity': self.cosine_similarity,
            'mismatch_reasons': self.mismatch_reasons
        }


@dataclass
class SignatureMatchingResult:
    """Result of signature matching analysis across multiple trials."""

    persona_id: str
    persona_name: str
    model_id: str

    # Trial-level signatures
    ai_signatures: List[BehavioralSignature] = field(default_factory=list)
    human_signatures: List[BehavioralSignature] = field(default_factory=list)
    comparisons: List[SignatureComparisonResult] = field(default_factory=list)

    # Aggregate metrics
    mean_overall_similarity: float = 0.0
    action_accuracy: float = 0.0
    mean_confidence_similarity: float = 0.0
    mean_speed_similarity: float = 0.0
    mean_depth_similarity: float = 0.0
    mean_attention_overlap: float = 0.0

    # Aggregate influence preservation
    urgency_effect_correlation: float = 0.0
    familiarity_effect_correlation: float = 0.0

    # Systematic biases detected
    systematic_biases: Dict[str, str] = field(default_factory=dict)

    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)

    # Metadata
    n_trials: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'persona_id': self.persona_id,
            'persona_name': self.persona_name,
            'model_id': self.model_id,
            'n_trials': self.n_trials,
            'mean_overall_similarity': self.mean_overall_similarity,
            'action_accuracy': self.action_accuracy,
            'mean_confidence_similarity': self.mean_confidence_similarity,
            'mean_speed_similarity': self.mean_speed_similarity,
            'mean_depth_similarity': self.mean_depth_similarity,
            'mean_attention_overlap': self.mean_attention_overlap,
            'urgency_effect_correlation': self.urgency_effect_correlation,
            'familiarity_effect_correlation': self.familiarity_effect_correlation,
            'systematic_biases': self.systematic_biases,
            'improvement_suggestions': self.improvement_suggestions
        }


class BehavioralSignatureExtractor:
    """
    Extracts behavioral signatures from LLM responses.

    Parses the reasoning, confidence, and other factors from LLM output
    to construct a complete behavioral signature.
    """

    # Keywords for attention pattern detection
    ATTENTION_KEYWORDS = {
        AttentionFocus.SENDER: ['sender', 'from', 'email address', 'who sent', 'source'],
        AttentionFocus.SUBJECT: ['subject', 'title', 'heading'],
        AttentionFocus.URGENCY: ['urgent', 'immediately', 'asap', 'deadline', 'expire', 'hurry', 'quick'],
        AttentionFocus.LINKS: ['link', 'url', 'click here', 'button', 'href'],
        AttentionFocus.CONTENT: ['body', 'message', 'text', 'content', 'says'],
        AttentionFocus.FORMATTING: ['format', 'spelling', 'grammar', 'layout', 'design', 'logo'],
        AttentionFocus.AUTHORITY: ['boss', 'manager', 'ceo', 'official', 'company', 'hr', 'it department']
    }

    # Keywords for hesitation detection
    HESITATION_KEYWORDS = [
        'unsure', 'not sure', 'uncertain', 'maybe', 'might', 'could be',
        'on one hand', 'on the other', 'but then', 'however', 'although',
        'considered', 'thought about', 'almost', 'nearly', 'hesitant'
    ]

    def extract_from_llm_response(
        self,
        action: str,
        confidence: Optional[str],
        reasoning: Optional[str],
        email_context: Dict[str, str],
        response_time_ms: Optional[float] = None
    ) -> BehavioralSignature:
        """
        Extract behavioral signature from LLM response.

        Args:
            action: The action taken (click/report/ignore)
            confidence: Self-reported confidence (high/medium/low)
            reasoning: The reasoning text from LLM
            email_context: Dict with email_type, urgency_level, sender_familiarity
            response_time_ms: Optional response latency

        Returns:
            BehavioralSignature with all causal factors extracted
        """
        signature = BehavioralSignature(
            action=action.lower(),
            email_type=email_context.get('email_type', ''),
            email_urgency=email_context.get('urgency_level', ''),
            email_familiarity=email_context.get('sender_familiarity', '')
        )

        # Extract confidence
        if confidence:
            signature.confidence_level = confidence.lower()
            conf_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
            signature.confidence_score = conf_map.get(confidence.lower(), 0.5)

        # Extract from reasoning text
        if reasoning:
            signature.reasoning_text = reasoning
            signature.reasoning_word_count = len(reasoning.split())

            # Extract reasoning depth
            signature.reasoning_depth = self._extract_reasoning_depth(reasoning)

            # Extract attention pattern
            signature.attention_pattern = self._extract_attention_pattern(reasoning)
            signature.details_mentioned = self._extract_details_mentioned(reasoning)

            # Extract hesitation
            signature.hesitation_level, signature.alternative_considered = \
                self._extract_hesitation(reasoning)

            # Extract influence factors from reasoning
            signature.urgency_influence = self._extract_urgency_influence(
                reasoning, email_context.get('urgency_level', '')
            )
            signature.familiarity_influence = self._extract_familiarity_influence(
                reasoning, email_context.get('sender_familiarity', '')
            )
            signature.authority_influence = self._extract_authority_influence(reasoning)

        # Response time / decision speed
        if response_time_ms:
            signature.estimated_response_time_ms = response_time_ms
            signature.decision_speed = self._classify_decision_speed(response_time_ms)
        else:
            # Infer from reasoning length
            signature.decision_speed = self._infer_speed_from_reasoning(reasoning)

        return signature

    def _extract_reasoning_depth(self, reasoning: str) -> ReasoningDepth:
        """Classify reasoning depth based on text analysis."""
        if not reasoning:
            return ReasoningDepth.NONE

        word_count = len(reasoning.split())
        sentences = reasoning.count('.') + reasoning.count('!') + reasoning.count('?')

        # Quick heuristics
        if word_count < 10:
            return ReasoningDepth.SHALLOW
        elif word_count < 30 and sentences <= 2:
            return ReasoningDepth.MODERATE
        elif word_count >= 30 or sentences >= 3:
            return ReasoningDepth.DEEP
        else:
            return ReasoningDepth.MODERATE

    def _extract_attention_pattern(self, reasoning: str) -> List[AttentionFocus]:
        """Extract what the persona paid attention to."""
        if not reasoning:
            return [AttentionFocus.NOTHING]

        reasoning_lower = reasoning.lower()
        attention_items = []

        for focus, keywords in self.ATTENTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in reasoning_lower:
                    attention_items.append(focus)
                    break  # Only add each focus once

        return attention_items if attention_items else [AttentionFocus.NOTHING]

    def _extract_details_mentioned(self, reasoning: str) -> List[str]:
        """Extract specific details mentioned in reasoning."""
        if not reasoning:
            return []

        details = []

        # Look for quoted content
        quoted = re.findall(r'"([^"]+)"', reasoning)
        details.extend(quoted[:3])  # Limit to 3

        # Look for specific mentions
        if 'sender' in reasoning.lower() or 'from' in reasoning.lower():
            details.append('sender_address')
        if 'link' in reasoning.lower() or 'url' in reasoning.lower():
            details.append('links')
        if any(word in reasoning.lower() for word in ['urgent', 'deadline', 'expire']):
            details.append('urgency_cues')

        return details[:5]  # Limit to 5 details

    def _extract_hesitation(self, reasoning: str) -> Tuple[HesitationLevel, Optional[str]]:
        """Extract hesitation level and alternative considered."""
        if not reasoning:
            return HesitationLevel.NONE, None

        reasoning_lower = reasoning.lower()

        # Count hesitation indicators
        hesitation_count = sum(1 for kw in self.HESITATION_KEYWORDS if kw in reasoning_lower)

        # Determine level
        if hesitation_count == 0:
            level = HesitationLevel.NONE
        elif hesitation_count == 1:
            level = HesitationLevel.SLIGHT
        elif hesitation_count <= 3:
            level = HesitationLevel.MODERATE
        else:
            level = HesitationLevel.HIGH

        # Try to extract alternative considered
        alternative = None
        alternative_patterns = [
            r'almost (clicked|reported|ignored)',
            r'considered (clicking|reporting|ignoring)',
            r'thought about (clicking|reporting|ignoring)'
        ]
        for pattern in alternative_patterns:
            match = re.search(pattern, reasoning_lower)
            if match:
                action_map = {
                    'click': 'click', 'clicked': 'click', 'clicking': 'click',
                    'report': 'report', 'reported': 'report', 'reporting': 'report',
                    'ignore': 'ignore', 'ignored': 'ignore', 'ignoring': 'ignore'
                }
                for word in match.group(0).split():
                    if word in action_map:
                        alternative = action_map[word]
                        break
                break

        return level, alternative

    def _extract_urgency_influence(self, reasoning: str, urgency_level: str) -> float:
        """
        Extract how much urgency influenced the decision.

        Returns value in [-1, 1]:
        - Positive: urgency pushed toward clicking
        - Negative: urgency was recognized as red flag
        - Near zero: urgency not mentioned
        """
        if not reasoning:
            return 0.0

        reasoning_lower = reasoning.lower()

        # Count urgency mentions
        urgency_keywords = ['urgent', 'immediately', 'asap', 'deadline', 'expire', 'hurry']
        urgency_mentions = sum(1 for kw in urgency_keywords if kw in reasoning_lower)

        if urgency_mentions == 0:
            return 0.0

        # Check if urgency was seen as red flag or motivator
        red_flag_phrases = ['suspicious', 'red flag', 'tactic', 'pressure', 'scam']
        motivator_phrases = ['need to act', 'should respond', 'better hurry', 'cant wait']

        red_flag_count = sum(1 for p in red_flag_phrases if p in reasoning_lower)
        motivator_count = sum(1 for p in motivator_phrases if p in reasoning_lower)

        # Calculate influence
        if red_flag_count > motivator_count:
            return -0.3 * urgency_mentions  # Urgency recognized as red flag
        elif motivator_count > red_flag_count:
            return 0.3 * urgency_mentions  # Urgency motivated action
        else:
            return 0.1 * urgency_mentions  # Neutral mention

    def _extract_familiarity_influence(self, reasoning: str, familiarity: str) -> float:
        """Extract how much sender familiarity influenced the decision."""
        if not reasoning:
            return 0.0

        reasoning_lower = reasoning.lower()

        familiarity_keywords = ['familiar', 'know', 'recognize', 'trust', 'sender']
        familiarity_mentions = sum(1 for kw in familiarity_keywords if kw in reasoning_lower)

        if familiarity_mentions == 0:
            return 0.0

        # Check if familiarity increased or decreased trust
        trust_phrases = ['can trust', 'seems legit', 'know them', 'familiar sender']
        distrust_phrases = ['pretending', 'spoofed', 'fake', 'impersonat']

        trust_count = sum(1 for p in trust_phrases if p in reasoning_lower)
        distrust_count = sum(1 for p in distrust_phrases if p in reasoning_lower)

        if trust_count > distrust_count:
            return 0.3 * familiarity_mentions
        elif distrust_count > trust_count:
            return -0.3 * familiarity_mentions
        else:
            return 0.1 * familiarity_mentions

    def _extract_authority_influence(self, reasoning: str) -> float:
        """Extract how much authority influenced the decision."""
        if not reasoning:
            return 0.0

        reasoning_lower = reasoning.lower()

        authority_keywords = ['boss', 'manager', 'ceo', 'director', 'official', 'company']
        authority_mentions = sum(1 for kw in authority_keywords if kw in reasoning_lower)

        if authority_mentions == 0:
            return 0.0

        # Check if authority was trusted or questioned
        trust_phrases = ['from management', 'official request', 'must comply']
        question_phrases = ['would they', 'unusual for', 'verify', 'confirm']

        trust_count = sum(1 for p in trust_phrases if p in reasoning_lower)
        question_count = sum(1 for p in question_phrases if p in reasoning_lower)

        if trust_count > question_count:
            return 0.3 * authority_mentions
        elif question_count > trust_count:
            return -0.2 * authority_mentions
        else:
            return 0.1 * authority_mentions

    def _classify_decision_speed(self, response_time_ms: float) -> DecisionSpeed:
        """Classify response time into decision speed category."""
        if response_time_ms < 1000:
            return DecisionSpeed.INSTANT
        elif response_time_ms < 3000:
            return DecisionSpeed.FAST
        elif response_time_ms < 8000:
            return DecisionSpeed.MODERATE
        else:
            return DecisionSpeed.SLOW

    def _infer_speed_from_reasoning(self, reasoning: Optional[str]) -> DecisionSpeed:
        """Infer decision speed from reasoning text length."""
        if not reasoning:
            return DecisionSpeed.INSTANT

        word_count = len(reasoning.split())

        if word_count < 5:
            return DecisionSpeed.INSTANT
        elif word_count < 15:
            return DecisionSpeed.FAST
        elif word_count < 40:
            return DecisionSpeed.MODERATE
        else:
            return DecisionSpeed.SLOW


class BehavioralSignatureMatcher:
    """
    Compares behavioral signatures between AI and human responses.

    Goes beyond simple action matching to assess full behavioral fidelity.
    """

    # Weights for different signature components
    COMPONENT_WEIGHTS = {
        'action': 0.30,           # Action is still important
        'confidence': 0.15,       # Confidence matching
        'speed': 0.10,            # Decision speed
        'depth': 0.10,            # Reasoning depth
        'attention': 0.15,        # Attention pattern overlap
        'hesitation': 0.05,       # Hesitation matching
        'influence': 0.15         # Influence factors
    }

    def __init__(self, extractor: BehavioralSignatureExtractor = None):
        self.extractor = extractor or BehavioralSignatureExtractor()

    def compare_signatures(
        self,
        ai_signature: BehavioralSignature,
        human_signature: BehavioralSignature
    ) -> SignatureComparisonResult:
        """
        Compare AI and human behavioral signatures.

        Returns detailed comparison with component-wise similarities.
        """
        result = SignatureComparisonResult()

        # Action match (exact match required)
        result.action_match = (
            ai_signature.action.lower() == human_signature.action.lower()
        )

        # Confidence similarity
        result.confidence_similarity = 1 - abs(
            ai_signature.confidence_score - human_signature.confidence_score
        )

        # Speed similarity
        speed_order = [DecisionSpeed.INSTANT, DecisionSpeed.FAST,
                      DecisionSpeed.MODERATE, DecisionSpeed.SLOW]
        ai_speed_idx = speed_order.index(ai_signature.decision_speed) if ai_signature.decision_speed in speed_order else 2
        human_speed_idx = speed_order.index(human_signature.decision_speed) if human_signature.decision_speed in speed_order else 2
        result.speed_similarity = 1 - abs(ai_speed_idx - human_speed_idx) / 3

        # Depth similarity
        depth_order = [ReasoningDepth.NONE, ReasoningDepth.SHALLOW,
                      ReasoningDepth.MODERATE, ReasoningDepth.DEEP]
        ai_depth_idx = depth_order.index(ai_signature.reasoning_depth) if ai_signature.reasoning_depth in depth_order else 2
        human_depth_idx = depth_order.index(human_signature.reasoning_depth) if human_signature.reasoning_depth in depth_order else 2
        result.depth_similarity = 1 - abs(ai_depth_idx - human_depth_idx) / 3

        # Attention overlap (Jaccard similarity)
        ai_attention = set(ai_signature.attention_pattern)
        human_attention = set(human_signature.attention_pattern)
        if ai_attention or human_attention:
            intersection = len(ai_attention & human_attention)
            union = len(ai_attention | human_attention)
            result.attention_overlap = intersection / union if union > 0 else 0
        else:
            result.attention_overlap = 1.0  # Both paid no attention

        # Hesitation similarity
        hesitation_order = [HesitationLevel.NONE, HesitationLevel.SLIGHT,
                          HesitationLevel.MODERATE, HesitationLevel.HIGH]
        ai_hes_idx = hesitation_order.index(ai_signature.hesitation_level) if ai_signature.hesitation_level in hesitation_order else 0
        human_hes_idx = hesitation_order.index(human_signature.hesitation_level) if human_signature.hesitation_level in hesitation_order else 0
        result.hesitation_similarity = 1 - abs(ai_hes_idx - human_hes_idx) / 3

        # Influence factor similarity
        ai_influences = np.array([
            ai_signature.urgency_influence,
            ai_signature.familiarity_influence,
            ai_signature.authority_influence
        ])
        human_influences = np.array([
            human_signature.urgency_influence,
            human_signature.familiarity_influence,
            human_signature.authority_influence
        ])
        influence_diff = np.abs(ai_influences - human_influences)
        result.influence_similarity = 1 - np.mean(influence_diff) / 2  # Normalize by max diff

        # Vector-based distances
        ai_vector = ai_signature.to_vector()
        human_vector = human_signature.to_vector()
        result.euclidean_distance = float(euclidean(ai_vector, human_vector))
        result.cosine_similarity = float(1 - cosine(ai_vector, human_vector)) if np.any(ai_vector) and np.any(human_vector) else 0

        # Calculate weighted overall similarity
        result.overall_similarity = (
            self.COMPONENT_WEIGHTS['action'] * (1.0 if result.action_match else 0.0) +
            self.COMPONENT_WEIGHTS['confidence'] * result.confidence_similarity +
            self.COMPONENT_WEIGHTS['speed'] * result.speed_similarity +
            self.COMPONENT_WEIGHTS['depth'] * result.depth_similarity +
            self.COMPONENT_WEIGHTS['attention'] * result.attention_overlap +
            self.COMPONENT_WEIGHTS['hesitation'] * result.hesitation_similarity +
            self.COMPONENT_WEIGHTS['influence'] * result.influence_similarity
        )

        # Diagnose mismatches
        result.mismatch_reasons = self._diagnose_mismatches(
            ai_signature, human_signature, result
        )

        return result

    def _diagnose_mismatches(
        self,
        ai_sig: BehavioralSignature,
        human_sig: BehavioralSignature,
        comparison: SignatureComparisonResult
    ) -> List[str]:
        """Identify reasons for behavioral mismatch."""
        reasons = []

        if not comparison.action_match:
            reasons.append(f"Action mismatch: AI={ai_sig.action}, Human={human_sig.action}")

        if comparison.confidence_similarity < 0.5:
            reasons.append(f"Confidence mismatch: AI={ai_sig.confidence_level}, Human={human_sig.confidence_level}")

        if comparison.speed_similarity < 0.5:
            reasons.append(f"Speed mismatch: AI={ai_sig.decision_speed.value}, Human={human_sig.decision_speed.value}")

        if comparison.depth_similarity < 0.5:
            reasons.append(f"Reasoning depth mismatch: AI={ai_sig.reasoning_depth.value}, Human={human_sig.reasoning_depth.value}")

        if comparison.attention_overlap < 0.3:
            ai_focus = [a.value for a in ai_sig.attention_pattern]
            human_focus = [a.value for a in human_sig.attention_pattern]
            reasons.append(f"Attention pattern mismatch: AI focused on {ai_focus}, Human on {human_focus}")

        # Check influence factor mismatches
        if abs(ai_sig.urgency_influence - human_sig.urgency_influence) > 0.3:
            direction = "more" if ai_sig.urgency_influence > human_sig.urgency_influence else "less"
            reasons.append(f"AI {direction} influenced by urgency than human")

        if abs(ai_sig.familiarity_influence - human_sig.familiarity_influence) > 0.3:
            direction = "more" if ai_sig.familiarity_influence > human_sig.familiarity_influence else "less"
            reasons.append(f"AI {direction} influenced by sender familiarity than human")

        return reasons

    async def run_signature_matching(
        self,
        persona: Persona,
        ai_responses: List[Dict],
        human_responses: List[Dict],
        model_id: str
    ) -> SignatureMatchingResult:
        """
        Run full signature matching analysis.

        Args:
            persona: Persona being simulated
            ai_responses: List of AI response dicts with action, confidence, reasoning
            human_responses: List of human response dicts (ground truth)
            model_id: Model used for AI responses

        Returns:
            SignatureMatchingResult with aggregate analysis
        """
        result = SignatureMatchingResult(
            persona_id=persona.persona_id,
            persona_name=persona.name,
            model_id=model_id,
            started_at=datetime.now().isoformat()
        )

        # Extract signatures and compare
        for ai_resp, human_resp in zip(ai_responses, human_responses):
            email_context = {
                'email_type': human_resp.get('email_type', ''),
                'urgency_level': human_resp.get('urgency_level', ''),
                'sender_familiarity': human_resp.get('sender_familiarity', '')
            }

            # Extract AI signature
            ai_sig = self.extractor.extract_from_llm_response(
                action=ai_resp.get('action', 'ignore'),
                confidence=ai_resp.get('confidence'),
                reasoning=ai_resp.get('reasoning'),
                email_context=email_context,
                response_time_ms=ai_resp.get('response_time_ms')
            )
            result.ai_signatures.append(ai_sig)

            # Extract human signature (from study data)
            human_sig = self.extractor.extract_from_llm_response(
                action=human_resp.get('action', 'ignore'),
                confidence=human_resp.get('confidence'),
                reasoning=human_resp.get('reasoning'),
                email_context=email_context,
                response_time_ms=human_resp.get('response_time_ms')
            )
            result.human_signatures.append(human_sig)

            # Compare
            comparison = self.compare_signatures(ai_sig, human_sig)
            result.comparisons.append(comparison)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(result)

        result.n_trials = len(result.comparisons)
        result.completed_at = datetime.now().isoformat()

        return result

    def _calculate_aggregate_metrics(self, result: SignatureMatchingResult):
        """Calculate aggregate metrics from all comparisons."""
        if not result.comparisons:
            return

        comparisons = result.comparisons

        # Mean similarities
        result.mean_overall_similarity = np.mean([c.overall_similarity for c in comparisons])
        result.action_accuracy = np.mean([1.0 if c.action_match else 0.0 for c in comparisons])
        result.mean_confidence_similarity = np.mean([c.confidence_similarity for c in comparisons])
        result.mean_speed_similarity = np.mean([c.speed_similarity for c in comparisons])
        result.mean_depth_similarity = np.mean([c.depth_similarity for c in comparisons])
        result.mean_attention_overlap = np.mean([c.attention_overlap for c in comparisons])

        # Influence correlations
        if len(result.ai_signatures) > 2:
            ai_urgency = [s.urgency_influence for s in result.ai_signatures]
            human_urgency = [s.urgency_influence for s in result.human_signatures]
            if np.std(ai_urgency) > 0 and np.std(human_urgency) > 0:
                result.urgency_effect_correlation = float(np.corrcoef(ai_urgency, human_urgency)[0, 1])

            ai_familiarity = [s.familiarity_influence for s in result.ai_signatures]
            human_familiarity = [s.familiarity_influence for s in result.human_signatures]
            if np.std(ai_familiarity) > 0 and np.std(human_familiarity) > 0:
                result.familiarity_effect_correlation = float(np.corrcoef(ai_familiarity, human_familiarity)[0, 1])

        # Detect systematic biases
        result.systematic_biases = self._detect_systematic_biases(result)

        # Generate improvement suggestions
        result.improvement_suggestions = self._generate_suggestions(result)

    def _detect_systematic_biases(self, result: SignatureMatchingResult) -> Dict[str, str]:
        """Detect systematic biases in AI behavior."""
        biases = {}

        if not result.ai_signatures or not result.human_signatures:
            return biases

        # Speed bias
        ai_speeds = [s.decision_speed for s in result.ai_signatures]
        human_speeds = [s.decision_speed for s in result.human_signatures]
        speed_order = [DecisionSpeed.INSTANT, DecisionSpeed.FAST,
                      DecisionSpeed.MODERATE, DecisionSpeed.SLOW]
        ai_speed_mean = np.mean([speed_order.index(s) for s in ai_speeds])
        human_speed_mean = np.mean([speed_order.index(s) for s in human_speeds])
        if ai_speed_mean - human_speed_mean > 0.5:
            biases['speed'] = "AI decides slower than human"
        elif human_speed_mean - ai_speed_mean > 0.5:
            biases['speed'] = "AI decides faster than human"

        # Depth bias
        ai_depths = [s.reasoning_depth for s in result.ai_signatures]
        human_depths = [s.reasoning_depth for s in result.human_signatures]
        depth_order = [ReasoningDepth.NONE, ReasoningDepth.SHALLOW,
                      ReasoningDepth.MODERATE, ReasoningDepth.DEEP]
        ai_depth_mean = np.mean([depth_order.index(d) for d in ai_depths])
        human_depth_mean = np.mean([depth_order.index(d) for d in human_depths])
        if ai_depth_mean - human_depth_mean > 0.5:
            biases['depth'] = "AI over-analyzes (too much reasoning)"
        elif human_depth_mean - ai_depth_mean > 0.5:
            biases['depth'] = "AI under-analyzes (too little reasoning)"

        # Confidence bias
        ai_conf = np.mean([s.confidence_score for s in result.ai_signatures])
        human_conf = np.mean([s.confidence_score for s in result.human_signatures])
        if ai_conf - human_conf > 0.2:
            biases['confidence'] = "AI is overconfident"
        elif human_conf - ai_conf > 0.2:
            biases['confidence'] = "AI is underconfident"

        return biases

    def _generate_suggestions(self, result: SignatureMatchingResult) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        biases = result.systematic_biases

        if 'speed' in biases:
            if 'slower' in biases['speed']:
                suggestions.append("Consider adding impulsivity cues to prompt for faster decisions")
            else:
                suggestions.append("Add deliberation instructions to slow down decision making")

        if 'depth' in biases:
            if 'over-analyzes' in biases['depth']:
                suggestions.append("Simplify prompt structure - remove detailed analysis requirements")
            else:
                suggestions.append("Add chain-of-thought requirements to increase reasoning depth")

        if 'confidence' in biases:
            if 'overconfident' in biases['confidence']:
                suggestions.append("Add uncertainty acknowledgment to persona description")
            else:
                suggestions.append("Strengthen persona conviction in prompt")

        if result.mean_attention_overlap < 0.5:
            suggestions.append("Add attention guidance - specify what this persona typically notices")

        if result.action_accuracy < 0.7:
            suggestions.append("Action accuracy low - consider recalibrating behavioral statistics")

        return suggestions

    def get_matching_summary(self, result: SignatureMatchingResult) -> str:
        """Generate human-readable summary of signature matching."""
        lines = [
            f"\n{'='*60}",
            f"BEHAVIORAL SIGNATURE MATCHING: {result.persona_name}",
            f"{'='*60}",
            "",
            f"Trials analyzed: {result.n_trials}",
            "",
            "SIMILARITY SCORES:",
            "-" * 40,
            f"  Overall Similarity:    {result.mean_overall_similarity:.1%}",
            f"  Action Accuracy:       {result.action_accuracy:.1%}",
            f"  Confidence Match:      {result.mean_confidence_similarity:.1%}",
            f"  Speed Match:           {result.mean_speed_similarity:.1%}",
            f"  Reasoning Depth Match: {result.mean_depth_similarity:.1%}",
            f"  Attention Overlap:     {result.mean_attention_overlap:.1%}",
            "",
            "INFLUENCE CORRELATIONS:",
            "-" * 40,
            f"  Urgency Effect:        r={result.urgency_effect_correlation:.2f}",
            f"  Familiarity Effect:    r={result.familiarity_effect_correlation:.2f}",
        ]

        if result.systematic_biases:
            lines.extend([
                "",
                "SYSTEMATIC BIASES DETECTED:",
                "-" * 40
            ])
            for bias_type, description in result.systematic_biases.items():
                lines.append(f"  {bias_type.upper()}: {description}")

        if result.improvement_suggestions:
            lines.extend([
                "",
                "IMPROVEMENT SUGGESTIONS:",
                "-" * 40
            ])
            for i, suggestion in enumerate(result.improvement_suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        lines.append(f"\n{'='*60}")

        return "\n".join(lines)
