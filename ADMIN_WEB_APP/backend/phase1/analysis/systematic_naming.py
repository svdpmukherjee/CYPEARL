"""
Systematic Persona Naming for CYPEARL Phase 1

HYBRID APPROACH:
- Systematic Code: Deterministic, taxonomy-encoded (e.g., CR-INT-IMP-CLK)
- LLM Creative Name: Memorable, human-friendly (e.g., "Impulsive Phish Prone")
- Combined: "CR-INT-IMP-CLK: Impulsive Phish Prone"

The systematic code reveals persona attributes at a glance:
- Position 1: Risk Level (CR/HI/MD/LO)
- Position 2: Cognitive Style (INT/ANL/BAL)
- Position 3: Dominant Trait (IMP/TRU/SKP/etc.)
- Position 4: Behavior Pattern (CLK/RPT/IGN/etc.)

The LLM name adds memorability for human communication.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class CognitiveStyle(Enum):
    INTUITIVE = "Intuitive"
    ANALYTICAL = "Analytical"
    BALANCED = "Balanced"


class DominantTrait(Enum):
    IMPULSIVE = "Impulsive"
    TRUSTING = "Trusting"
    SKEPTICAL = "Skeptical"
    ANXIOUS = "Anxious"
    CONFIDENT = "Confident"
    OVERWHELMED = "Overwhelmed"
    CAUTIOUS = "Cautious"
    CURIOUS = "Curious"
    COMPLIANT = "Compliant"
    DISTRACTED = "Distracted"


class BehaviorPattern(Enum):
    CLICKER = "Clicker"
    REPORTER = "Reporter"
    IGNORER = "Ignorer"
    HESITANT = "Hesitant"
    INSPECTOR = "Inspector"


@dataclass
class SystematicName:
    """Structured persona name with all components"""
    risk: RiskLevel
    cognitive_style: CognitiveStyle
    dominant_trait: DominantTrait
    behavior: BehaviorPattern
    cluster_id: int
    llm_name: Optional[str] = None  # LLM-generated creative name

    @property
    def short_code(self) -> str:
        """Short code: 'CR-INT-IMP-CLK'"""
        risk_codes = {'Critical': 'CR', 'High': 'HI', 'Medium': 'MD', 'Low': 'LO'}
        style_codes = {'Intuitive': 'INT', 'Analytical': 'ANL', 'Balanced': 'BAL'}
        trait_codes = {
            'Impulsive': 'IMP', 'Trusting': 'TRU', 'Skeptical': 'SKP',
            'Anxious': 'ANX', 'Confident': 'CON', 'Overwhelmed': 'OVR',
            'Cautious': 'CAU', 'Curious': 'CUR', 'Compliant': 'CMP',
            'Distracted': 'DST'
        }
        behavior_codes = {
            'Clicker': 'CLK', 'Reporter': 'RPT', 'Ignorer': 'IGN',
            'Hesitant': 'HES', 'Inspector': 'INS'
        }

        return f"{risk_codes[self.risk.value]}-{style_codes[self.cognitive_style.value]}-{trait_codes[self.dominant_trait.value]}-{behavior_codes[self.behavior.value]}"

    @property
    def readable_code(self) -> str:
        """Readable version: 'Critical Intuitive Impulsive Clicker'"""
        return f"{self.risk.value} {self.cognitive_style.value} {self.dominant_trait.value} {self.behavior.value}"

    @property
    def hybrid_name(self) -> str:
        """Hybrid name: 'CR-INT-IMP-CLK: Impulsive Phish Prone'"""
        if self.llm_name:
            return f"{self.short_code}: {self.llm_name}"
        return self.short_code

    @property
    def display_name(self) -> str:
        """Display name for UI"""
        return self.hybrid_name

    def to_dict(self) -> Dict:
        return {
            'name': self.hybrid_name,
            'short_code': self.short_code,
            'readable_code': self.readable_code,
            'llm_name': self.llm_name,
            'display_name': self.display_name,
            'components': {
                'risk': self.risk.value,
                'cognitive_style': self.cognitive_style.value,
                'dominant_trait': self.dominant_trait.value,
                'behavior': self.behavior.value
            },
            'cluster_id': self.cluster_id
        }


class SystematicPersonaNamer:
    """
    Generates systematic, taxonomy-encoded persona names.

    The naming follows a fixed schema:
    [Risk Level] [Cognitive Style] [Dominant Trait] [Behavior Pattern]

    Each component is derived algorithmically from the persona's data:
    - Risk Level: Based on phishing click rate thresholds
    - Cognitive Style: Based on CRT, need_for_cognition, impulsivity
    - Dominant Trait: Based on highest z-score trait
    - Behavior Pattern: Based on behavioral outcomes (click, report, ignore rates)
    """

    # Risk level thresholds (click rate)
    RISK_THRESHOLDS = {
        'critical': 0.35,
        'high': 0.28,
        'medium': 0.22,
    }

    # Cognitive style thresholds
    ANALYTICAL_THRESHOLD = 0.3
    INTUITIVE_THRESHOLD = -0.3

    # Trait to dominant trait mapping
    TRAIT_MAPPINGS = {
        # Impulsive indicators
        'impulsivity_total': ('high', DominantTrait.IMPULSIVE),
        # Trust indicators
        'trust_propensity': ('high', DominantTrait.TRUSTING),
        'agreeableness': ('high', DominantTrait.COMPLIANT),
        # Skeptical indicators
        'phishing_self_efficacy': ('high', DominantTrait.SKEPTICAL),
        'security_attitudes': ('high', DominantTrait.SKEPTICAL),
        # Anxious indicators
        'state_anxiety': ('high', DominantTrait.ANXIOUS),
        'neuroticism': ('high', DominantTrait.ANXIOUS),
        # Confident indicators
        'phishing_knowledge': ('high', DominantTrait.CONFIDENT),
        'technical_expertise': ('high', DominantTrait.CONFIDENT),
        # Overwhelmed indicators
        'current_stress': ('high', DominantTrait.OVERWHELMED),
        'fatigue_level': ('high', DominantTrait.OVERWHELMED),
        # Cautious indicators
        'perceived_risk': ('high', DominantTrait.CAUTIOUS),
        'conscientiousness': ('high', DominantTrait.CAUTIOUS),
        # Curious indicators
        'sensation_seeking': ('high', DominantTrait.CURIOUS),
        'openness': ('high', DominantTrait.CURIOUS),
        # Distracted indicators
        'working_memory': ('low', DominantTrait.DISTRACTED),
        'need_for_cognition': ('low', DominantTrait.DISTRACTED),
    }

    def __init__(self):
        self.names: Dict[int, SystematicName] = {}

    def _determine_risk_level(self, persona: Dict) -> RiskLevel:
        """Determine risk level from click rate"""
        click_rate = persona.get('phishing_click_rate', 0.3)

        if click_rate >= self.RISK_THRESHOLDS['critical']:
            return RiskLevel.CRITICAL
        elif click_rate >= self.RISK_THRESHOLDS['high']:
            return RiskLevel.HIGH
        elif click_rate >= self.RISK_THRESHOLDS['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _determine_cognitive_style(self, persona: Dict) -> CognitiveStyle:
        """Determine cognitive style from cognitive traits"""
        traits = persona.get('trait_zscores', {})

        crt = traits.get('crt_score', 0)
        nfc = traits.get('need_for_cognition', 0)
        impulsivity = traits.get('impulsivity_total', 0)
        working_memory = traits.get('working_memory', 0)

        # Compute analytical score
        analytical_score = (
            crt * 0.4 +
            nfc * 0.25 +
            working_memory * 0.15 +
            (-impulsivity) * 0.2
        )

        if analytical_score > self.ANALYTICAL_THRESHOLD:
            return CognitiveStyle.ANALYTICAL
        elif analytical_score < self.INTUITIVE_THRESHOLD:
            return CognitiveStyle.INTUITIVE
        else:
            return CognitiveStyle.BALANCED

    def _determine_dominant_trait(self, persona: Dict) -> DominantTrait:
        """Determine dominant trait from highest impact z-score"""
        traits = persona.get('trait_zscores', {})
        top_high = persona.get('top_high_traits', [])
        top_low = persona.get('top_low_traits', [])

        # Score each potential dominant trait
        trait_scores: Dict[DominantTrait, float] = {}

        for trait_name, (direction, dominant_trait) in self.TRAIT_MAPPINGS.items():
            z_score = traits.get(trait_name, 0)

            if direction == 'high' and z_score > 0.5:
                current_score = trait_scores.get(dominant_trait, 0)
                trait_scores[dominant_trait] = max(current_score, z_score)
            elif direction == 'low' and z_score < -0.5:
                current_score = trait_scores.get(dominant_trait, 0)
                trait_scores[dominant_trait] = max(current_score, abs(z_score))

        # Also check top_high_traits and top_low_traits
        for trait_name, z_score in top_high[:3]:
            if trait_name in self.TRAIT_MAPPINGS:
                direction, dominant_trait = self.TRAIT_MAPPINGS[trait_name]
                if direction == 'high':
                    current_score = trait_scores.get(dominant_trait, 0)
                    trait_scores[dominant_trait] = max(current_score, abs(z_score) * 1.2)

        for trait_name, z_score in top_low[:3]:
            if trait_name in self.TRAIT_MAPPINGS:
                direction, dominant_trait = self.TRAIT_MAPPINGS[trait_name]
                if direction == 'low':
                    current_score = trait_scores.get(dominant_trait, 0)
                    trait_scores[dominant_trait] = max(current_score, abs(z_score) * 1.2)

        # Return highest scoring trait, or default based on risk
        if trait_scores:
            return max(trait_scores.keys(), key=lambda t: trait_scores[t])

        # Default based on click rate
        click_rate = persona.get('phishing_click_rate', 0.3)
        if click_rate >= 0.35:
            return DominantTrait.IMPULSIVE
        elif click_rate <= 0.22:
            return DominantTrait.CAUTIOUS
        else:
            return DominantTrait.COMPLIANT

    def _determine_behavior_pattern(self, persona: Dict) -> BehaviorPattern:
        """Determine behavior pattern from behavioral outcomes"""
        click_rate = persona.get('phishing_click_rate', 0.3)
        behavioral = persona.get('behavioral_outcomes', {})

        report_rate = behavioral.get('report_rate', {}).get('mean', 0)
        hover_rate = behavioral.get('hover_rate', {}).get('mean', 0)
        sender_inspection = behavioral.get('sender_inspection_rate', {}).get('mean', 0)

        # Determine primary behavior
        if click_rate >= 0.35:
            return BehaviorPattern.CLICKER
        elif report_rate >= 0.15:
            return BehaviorPattern.REPORTER
        elif sender_inspection >= 0.3 or hover_rate >= 0.4:
            return BehaviorPattern.INSPECTOR
        elif click_rate <= 0.22 and report_rate <= 0.10:
            return BehaviorPattern.IGNORER
        else:
            return BehaviorPattern.HESITANT

    def generate_name(self, persona: Dict, cluster_id: int) -> SystematicName:
        """Generate systematic name for a single persona"""
        risk = self._determine_risk_level(persona)
        cognitive_style = self._determine_cognitive_style(persona)
        dominant_trait = self._determine_dominant_trait(persona)
        behavior = self._determine_behavior_pattern(persona)

        name = SystematicName(
            risk=risk,
            cognitive_style=cognitive_style,
            dominant_trait=dominant_trait,
            behavior=behavior,
            cluster_id=cluster_id
        )

        self.names[cluster_id] = name
        return name

    def generate_all_names(self, clusters: Dict[int, Dict],
                           llm_names: Optional[Dict[int, str]] = None) -> Dict[int, Dict]:
        """
        Generate systematic names for all clusters with optional LLM creative names.

        HYBRID MODE:
        - If llm_names provided: "CR-INT-IMP-CLK: Impulsive Phish Prone"
        - If no llm_names: "CR-INT-IMP-CLK" (systematic code only)

        Args:
            clusters: Dict of cluster_id -> cluster data
            llm_names: Optional dict of cluster_id -> LLM-generated creative name

        Returns dict in the same format as LLM naming for compatibility:
        {
            cluster_id: {
                'name': 'CR-INT-IMP-CLK: Impulsive Phish Prone',
                'archetype': 'CR-INT-IMP-CLK',
                'description': 'High-risk persona with intuitive decision-making...'
            }
        }
        """
        labels = {}
        llm_names = llm_names or {}

        for cluster_id, cluster_data in clusters.items():
            cluster_id = int(cluster_id)

            # Get LLM creative name if available
            llm_creative_name = llm_names.get(cluster_id)

            # Generate systematic name with optional LLM name
            name = self.generate_name(cluster_data, cluster_id)
            name.llm_name = llm_creative_name

            # Generate description from components
            description = self._generate_description(name, cluster_data)

            labels[cluster_id] = {
                'name': name.hybrid_name,  # "CR-INT-IMP-CLK: Creative Name" or just code
                'archetype': name.short_code,  # Always the systematic code
                'readable_code': name.readable_code,  # "Critical Intuitive Impulsive Clicker"
                'llm_name': llm_creative_name,  # Just the LLM part
                'description': description,
                'components': name.to_dict()['components']
            }

        return labels

    def _generate_description(self, name: SystematicName, persona: Dict) -> str:
        """Generate a brief description from the systematic name components"""
        risk_desc = {
            RiskLevel.CRITICAL: "Very high phishing susceptibility (35%+ click rate)",
            RiskLevel.HIGH: "High phishing susceptibility (28-35% click rate)",
            RiskLevel.MEDIUM: "Moderate phishing susceptibility (22-28% click rate)",
            RiskLevel.LOW: "Low phishing susceptibility (<22% click rate)"
        }

        style_desc = {
            CognitiveStyle.INTUITIVE: "fast, gut-feel decision maker",
            CognitiveStyle.ANALYTICAL: "deliberate, logical decision maker",
            CognitiveStyle.BALANCED: "context-adaptive decision maker"
        }

        trait_desc = {
            DominantTrait.IMPULSIVE: "acts quickly without deliberation",
            DominantTrait.TRUSTING: "tends to trust others easily",
            DominantTrait.SKEPTICAL: "naturally suspicious of requests",
            DominantTrait.ANXIOUS: "prone to worry and stress",
            DominantTrait.CONFIDENT: "high self-efficacy in security",
            DominantTrait.OVERWHELMED: "experiences cognitive overload",
            DominantTrait.CAUTIOUS: "careful and risk-averse",
            DominantTrait.CURIOUS: "drawn to explore and investigate",
            DominantTrait.COMPLIANT: "tends to follow requests",
            DominantTrait.DISTRACTED: "limited attention resources"
        }

        behavior_desc = {
            BehaviorPattern.CLICKER: "frequently clicks on links",
            BehaviorPattern.REPORTER: "actively reports suspicious emails",
            BehaviorPattern.IGNORER: "tends to ignore emails",
            BehaviorPattern.HESITANT: "shows uncertainty in responses",
            BehaviorPattern.INSPECTOR: "examines sender details carefully"
        }

        click_rate = persona.get('phishing_click_rate', 0) * 100
        n_participants = persona.get('n_participants', 0)

        return (
            f"{risk_desc[name.risk]}. "
            f"A {style_desc[name.cognitive_style]} who {trait_desc[name.dominant_trait]} and {behavior_desc[name.behavior]}. "
            f"Click rate: {click_rate:.1f}%, N={n_participants}."
        )


def generate_systematic_names(clusters: Dict[int, Dict],
                              llm_names: Optional[Dict[int, str]] = None) -> Dict[int, Dict]:
    """
    Generate systematic names for all clusters.

    HYBRID MODE (recommended):
    - Pass llm_names dict to get: "CR-INT-IMP-CLK: Impulsive Phish Prone"

    SYSTEMATIC ONLY MODE:
    - Don't pass llm_names to get: "CR-INT-IMP-CLK"

    Args:
        clusters: Dict of cluster_id -> cluster data
        llm_names: Optional dict of cluster_id -> LLM creative name

    Returns:
        Dict of cluster_id -> naming info compatible with existing persona labels format
    """
    namer = SystematicPersonaNamer()
    return namer.generate_all_names(clusters, llm_names)


def generate_systematic_codes(clusters: Dict[int, Dict]) -> Dict[int, str]:
    """
    Generate ONLY the systematic codes for all clusters (no LLM).

    Returns:
        Dict of cluster_id -> systematic code (e.g., "CR-INT-IMP-CLK")
    """
    namer = SystematicPersonaNamer()
    codes = {}

    for cluster_id, cluster_data in clusters.items():
        cluster_id = int(cluster_id)
        name = namer.generate_name(cluster_data, cluster_id)
        codes[cluster_id] = name.short_code

    return codes


def merge_with_llm_names(clusters: Dict[int, Dict],
                         llm_labels: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Merge systematic codes with existing LLM-generated labels.

    Takes existing LLM labels and adds systematic codes to create hybrid names.

    Args:
        clusters: Dict of cluster_id -> cluster data
        llm_labels: Existing labels from LLM naming (has 'name' key)

    Returns:
        Updated labels with hybrid names: "CR-INT-IMP-CLK: LLM Name"
    """
    namer = SystematicPersonaNamer()

    # Extract just the LLM names
    llm_names = {}
    for cluster_id, label_data in llm_labels.items():
        cluster_id = int(cluster_id)
        if isinstance(label_data, dict) and 'name' in label_data:
            llm_names[cluster_id] = label_data['name']
        elif isinstance(label_data, str):
            llm_names[cluster_id] = label_data

    return namer.generate_all_names(clusters, llm_names)
