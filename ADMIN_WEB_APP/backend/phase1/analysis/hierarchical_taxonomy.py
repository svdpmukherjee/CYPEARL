"""
Hierarchical Persona Taxonomy for CYPEARL Phase 1

Creates a multi-level taxonomy of discovered personas:
- Level 1: Meta-types based on cognitive style (Analytical vs Intuitive)
- Level 2: Risk profiles within each meta-type (Critical/High/Medium/Low)
- Level 3: Individual personas with their characteristics

This helps business managers:
1. See the "big picture" of persona landscape
2. Make strategic decisions at different granularities
3. Communicate findings more effectively to stakeholders
4. Target interventions at the right level of abstraction
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class CognitiveStyle(Enum):
    """Meta-types based on dual-process theory (System 1 vs System 2)"""
    INTUITIVE = "intuitive"  # Fast, automatic, emotional (System 1 dominant)
    ANALYTICAL = "analytical"  # Slow, deliberate, logical (System 2 dominant)
    BALANCED = "balanced"  # Mixed cognitive style


class RiskLevel(Enum):
    """Risk stratification within each meta-type"""
    CRITICAL = "critical"  # 35%+ phishing click rate
    HIGH = "high"  # 28-35% click rate
    MEDIUM = "medium"  # 22-28% click rate
    LOW = "low"  # <22% click rate


@dataclass
class TaxonomyNode:
    """A node in the hierarchical taxonomy"""
    id: str
    name: str
    level: int  # 0=root, 1=meta-type, 2=risk level, 3=persona
    description: str
    children: List['TaxonomyNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level,
            'description': self.description,
            'children': [c.to_dict() for c in self.children],
            'metadata': self.metadata,
            'n_children': len(self.children)
        }


@dataclass
class TaxonomyMetrics:
    """Metrics for a taxonomy level or node"""
    n_personas: int
    n_participants: int
    pct_of_population: float
    mean_click_rate: float
    mean_report_rate: float
    mean_response_latency: float
    risk_distribution: Dict[str, int]  # Count of each risk level

    def to_dict(self) -> Dict:
        return {
            'n_personas': self.n_personas,
            'n_participants': self.n_participants,
            'pct_of_population': self.pct_of_population,
            'mean_click_rate': self.mean_click_rate,
            'mean_report_rate': self.mean_report_rate,
            'mean_response_latency': self.mean_response_latency,
            'risk_distribution': self.risk_distribution
        }


class HierarchicalPersonaTaxonomy:
    """
    Builds a hierarchical taxonomy from flat persona clusters.

    The taxonomy has 3 levels:
    1. Meta-types: Cognitive style (Analytical vs Intuitive vs Balanced)
    2. Risk profiles: Within each meta-type, grouped by susceptibility level
    3. Personas: Individual discovered personas

    Example taxonomy:

    Root: All Personas
    ├── Intuitive Decision-Makers
    │   ├── Critical Risk
    │   │   └── Impulsive Clicker (Persona 3)
    │   ├── High Risk
    │   │   └── Trusting Intuitive (Persona 1)
    │   └── Medium Risk
    │       └── Busy Overwhelmed (Persona 5)
    ├── Analytical Decision-Makers
    │   ├── Low Risk
    │   │   └── Vigilant Skeptic (Persona 2)
    │   └── Medium Risk
    │       └── Cautious Verifier (Persona 4)
    └── Balanced Decision-Makers
        └── Medium Risk
            └── Adaptive Pragmatist (Persona 6)
    """

    # Thresholds for cognitive style classification
    ANALYTICAL_THRESHOLD = 0.3  # CRT z-score above this = more analytical
    INTUITIVE_THRESHOLD = -0.3  # CRT z-score below this = more intuitive
    IMPULSIVITY_THRESHOLD = 0.5  # High impulsivity suggests intuitive

    # Risk level thresholds (based on phishing click rate)
    RISK_THRESHOLDS = {
        'critical': 0.35,
        'high': 0.28,
        'medium': 0.22,
    }

    def __init__(self):
        self.taxonomy: Optional[TaxonomyNode] = None
        self.personas: List[Dict] = []
        self.meta_type_labels = {
            CognitiveStyle.INTUITIVE: {
                'name': 'Intuitive Decision-Makers',
                'description': 'Fast, automatic, gut-feel decision makers (System 1 dominant). '
                              'Tend to react quickly to emotional cues and urgency signals.'
            },
            CognitiveStyle.ANALYTICAL: {
                'name': 'Analytical Decision-Makers',
                'description': 'Deliberate, logical, reflective decision makers (System 2 dominant). '
                              'Tend to scrutinize details and verify before acting.'
            },
            CognitiveStyle.BALANCED: {
                'name': 'Balanced Decision-Makers',
                'description': 'Mixed cognitive style, adapts based on context. '
                              'Shows both intuitive and analytical tendencies.'
            }
        }

        self.risk_labels = {
            RiskLevel.CRITICAL: {
                'name': 'Critical Risk',
                'description': 'Very high susceptibility (35%+ click rate). Requires immediate, intensive intervention.'
            },
            RiskLevel.HIGH: {
                'name': 'High Risk',
                'description': 'High susceptibility (28-35% click rate). Needs targeted security awareness training.'
            },
            RiskLevel.MEDIUM: {
                'name': 'Medium Risk',
                'description': 'Moderate susceptibility (22-28% click rate). Standard training with reinforcement.'
            },
            RiskLevel.LOW: {
                'name': 'Low Risk',
                'description': 'Low susceptibility (<22% click rate). Security-aware, can be peer advocates.'
            }
        }

    def _determine_cognitive_style(self, persona: Dict) -> CognitiveStyle:
        """
        Determine meta-type based on cognitive traits.

        Uses:
        - CRT score (Cognitive Reflection Test) - higher = more analytical
        - Need for cognition - higher = enjoys thinking
        - Impulsivity - higher = more intuitive/reactive
        - Working memory - higher = can handle complex analysis
        """
        traits = persona.get('trait_zscores', {})

        crt = traits.get('crt_score', 0)
        nfc = traits.get('need_for_cognition', 0)
        impulsivity = traits.get('impulsivity_total', 0)
        working_memory = traits.get('working_memory', 0)

        # Compute analytical score (positive = more analytical)
        analytical_score = (
            crt * 0.4 +  # CRT is primary indicator
            nfc * 0.25 +  # Need for cognition
            working_memory * 0.15 +  # Working memory capacity
            (-impulsivity) * 0.2  # Low impulsivity = more analytical
        )

        if analytical_score > self.ANALYTICAL_THRESHOLD:
            return CognitiveStyle.ANALYTICAL
        elif analytical_score < self.INTUITIVE_THRESHOLD:
            return CognitiveStyle.INTUITIVE
        else:
            return CognitiveStyle.BALANCED

    def _determine_risk_level(self, persona: Dict) -> RiskLevel:
        """Determine risk level based on phishing click rate"""
        click_rate = persona.get('phishing_click_rate', 0.3)

        if click_rate >= self.RISK_THRESHOLDS['critical']:
            return RiskLevel.CRITICAL
        elif click_rate >= self.RISK_THRESHOLDS['high']:
            return RiskLevel.HIGH
        elif click_rate >= self.RISK_THRESHOLDS['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _compute_group_metrics(self, personas: List[Dict]) -> TaxonomyMetrics:
        """Compute aggregate metrics for a group of personas"""
        if not personas:
            return TaxonomyMetrics(
                n_personas=0, n_participants=0, pct_of_population=0,
                mean_click_rate=0, mean_report_rate=0, mean_response_latency=0,
                risk_distribution={}
            )

        total_participants = sum(p.get('n_participants', 0) for p in personas)
        total_pct = sum(p.get('pct_of_population', 0) for p in personas)

        # Weighted averages by participant count
        weights = [p.get('n_participants', 1) for p in personas]
        weight_sum = sum(weights) or 1

        mean_click = sum(
            p.get('phishing_click_rate', 0) * w
            for p, w in zip(personas, weights)
        ) / weight_sum

        mean_report = sum(
            p.get('behavioral_outcomes', {}).get('report_rate', {}).get('mean', 0) * w
            for p, w in zip(personas, weights)
        ) / weight_sum

        mean_latency = sum(
            p.get('behavioral_outcomes', {}).get('mean_response_latency', {}).get('mean', 0) * w
            for p, w in zip(personas, weights)
        ) / weight_sum

        # Risk distribution
        risk_dist = {}
        for p in personas:
            risk = p.get('risk_level', 'MEDIUM')
            risk_dist[risk] = risk_dist.get(risk, 0) + 1

        return TaxonomyMetrics(
            n_personas=len(personas),
            n_participants=total_participants,
            pct_of_population=total_pct,
            mean_click_rate=mean_click,
            mean_report_rate=mean_report,
            mean_response_latency=mean_latency,
            risk_distribution=risk_dist
        )

    def build_taxonomy(self, clusters: Dict[int, Dict],
                       persona_labels: Optional[Dict[int, Dict]] = None) -> TaxonomyNode:
        """
        Build the hierarchical taxonomy from flat clusters.

        Args:
            clusters: Dict of cluster_id -> cluster characterization data
            persona_labels: Optional dict of cluster_id -> {name, archetype, description}

        Returns:
            Root node of the taxonomy tree
        """
        self.personas = []
        persona_labels = persona_labels or {}

        # Convert clusters dict to list with IDs
        for cluster_id, cluster_data in clusters.items():
            persona = {
                **cluster_data,
                'cluster_id': int(cluster_id),
                'display_id': int(cluster_id) + 1,  # 1-indexed for display
                'persona_name': persona_labels.get(int(cluster_id), {}).get('name') or
                               f"Persona {int(cluster_id) + 1}",
                'archetype': persona_labels.get(int(cluster_id), {}).get('archetype', ''),
            }
            # Determine cognitive style and ensure risk level is set
            persona['cognitive_style'] = self._determine_cognitive_style(persona)
            if 'risk_level' not in persona:
                persona['risk_level'] = self._determine_risk_level(persona).value.upper()

            self.personas.append(persona)

        # Group by cognitive style (Level 1)
        by_cognitive_style: Dict[CognitiveStyle, List[Dict]] = {
            CognitiveStyle.INTUITIVE: [],
            CognitiveStyle.ANALYTICAL: [],
            CognitiveStyle.BALANCED: []
        }

        for persona in self.personas:
            style = persona['cognitive_style']
            by_cognitive_style[style].append(persona)

        # Build root node
        root_metrics = self._compute_group_metrics(self.personas)
        root = TaxonomyNode(
            id='root',
            name='All Behavioral Personas',
            level=0,
            description=f'Complete taxonomy of {len(self.personas)} discovered personas',
            metadata={
                'metrics': root_metrics.to_dict(),
                'total_personas': len(self.personas)
            }
        )

        # Build meta-type nodes (Level 1)
        for style, style_personas in by_cognitive_style.items():
            if not style_personas:
                continue

            style_info = self.meta_type_labels[style]
            style_metrics = self._compute_group_metrics(style_personas)

            style_node = TaxonomyNode(
                id=f'meta_{style.value}',
                name=style_info['name'],
                level=1,
                description=style_info['description'],
                metadata={
                    'cognitive_style': style.value,
                    'metrics': style_metrics.to_dict(),
                    'key_traits': self._get_style_key_traits(style)
                }
            )

            # Group by risk level within this meta-type (Level 2)
            by_risk: Dict[RiskLevel, List[Dict]] = {
                RiskLevel.CRITICAL: [],
                RiskLevel.HIGH: [],
                RiskLevel.MEDIUM: [],
                RiskLevel.LOW: []
            }

            for persona in style_personas:
                risk_str = persona.get('risk_level', 'MEDIUM').lower()
                try:
                    risk = RiskLevel(risk_str)
                except ValueError:
                    risk = RiskLevel.MEDIUM
                by_risk[risk].append(persona)

            # Build risk level nodes (Level 2)
            for risk, risk_personas in by_risk.items():
                if not risk_personas:
                    continue

                risk_info = self.risk_labels[risk]
                risk_metrics = self._compute_group_metrics(risk_personas)

                risk_node = TaxonomyNode(
                    id=f'{style.value}_{risk.value}',
                    name=risk_info['name'],
                    level=2,
                    description=risk_info['description'],
                    metadata={
                        'cognitive_style': style.value,
                        'risk_level': risk.value,
                        'metrics': risk_metrics.to_dict(),
                        'intervention_priority': self._get_intervention_priority(style, risk)
                    }
                )

                # Add individual personas (Level 3)
                for persona in sorted(risk_personas, key=lambda p: p.get('phishing_click_rate', 0), reverse=True):
                    persona_node = TaxonomyNode(
                        id=f"persona_{persona['cluster_id']}",
                        name=persona['persona_name'],
                        level=3,
                        description=persona.get('description', '') or persona.get('archetype', ''),
                        metadata={
                            'cluster_id': persona['cluster_id'],
                            'display_id': persona['display_id'],
                            'cognitive_style': style.value,
                            'risk_level': persona['risk_level'],
                            'n_participants': persona.get('n_participants', 0),
                            'pct_of_population': persona.get('pct_of_population', 0),
                            'phishing_click_rate': persona.get('phishing_click_rate', 0),
                            'top_high_traits': persona.get('top_high_traits', [])[:3],
                            'top_low_traits': persona.get('top_low_traits', [])[:3],
                            'archetype': persona.get('archetype', ''),
                            'behavioral_statistics': {
                                'click_rate': persona.get('phishing_click_rate', 0),
                                'report_rate': persona.get('behavioral_outcomes', {}).get('report_rate', {}).get('mean', 0),
                                'hover_rate': persona.get('behavioral_outcomes', {}).get('hover_rate', {}).get('mean', 0),
                                'sender_inspection': persona.get('behavioral_outcomes', {}).get('sender_inspection_rate', {}).get('mean', 0),
                            }
                        }
                    )
                    risk_node.children.append(persona_node)

                style_node.children.append(risk_node)

            # Sort risk nodes: Critical > High > Medium > Low
            risk_order = ['critical', 'high', 'medium', 'low']
            style_node.children.sort(
                key=lambda n: risk_order.index(n.metadata.get('risk_level', 'medium'))
            )

            root.children.append(style_node)

        # Sort meta-type nodes: put highest risk first
        def meta_type_priority(node):
            metrics = node.metadata.get('metrics', {})
            return -metrics.get('mean_click_rate', 0)

        root.children.sort(key=meta_type_priority)

        self.taxonomy = root
        return root

    def _get_style_key_traits(self, style: CognitiveStyle) -> List[str]:
        """Get key traits that define this cognitive style"""
        if style == CognitiveStyle.ANALYTICAL:
            return ['high_crt_score', 'high_need_for_cognition', 'low_impulsivity', 'high_working_memory']
        elif style == CognitiveStyle.INTUITIVE:
            return ['low_crt_score', 'high_impulsivity', 'high_trust_propensity', 'high_urgency_susceptibility']
        else:
            return ['balanced_crt', 'moderate_impulsivity', 'context_dependent']

    def _get_intervention_priority(self, style: CognitiveStyle, risk: RiskLevel) -> Dict:
        """
        Get recommended intervention priority for this combination.

        Business managers can use this to prioritize security awareness efforts.
        """
        base_priority = {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4
        }[risk]

        # Intuitive high-risk is hardest to train (they don't naturally deliberate)
        if style == CognitiveStyle.INTUITIVE and risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            intervention = {
                'priority': 1,
                'urgency': 'immediate',
                'recommended_approach': 'habit-based interventions',
                'training_notes': 'Focus on automatic recognition cues, not analysis. '
                                 'Use gamification, repeated exposure, and just-in-time warnings.'
            }
        elif style == CognitiveStyle.ANALYTICAL and risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            intervention = {
                'priority': 2,
                'urgency': 'high',
                'recommended_approach': 'knowledge-based training',
                'training_notes': 'Provide detailed threat information and verification procedures. '
                                 'They respond well to data and logic-based explanations.'
            }
        elif risk == RiskLevel.LOW:
            intervention = {
                'priority': 4,
                'urgency': 'low',
                'recommended_approach': 'peer advocacy programs',
                'training_notes': 'Low-risk employees can be champions for security culture. '
                                 'Consider them for security ambassador roles.'
            }
        else:
            intervention = {
                'priority': base_priority,
                'urgency': 'moderate',
                'recommended_approach': 'standard awareness training',
                'training_notes': 'Regular security awareness training with periodic reinforcement.'
            }

        return intervention

    def get_summary(self) -> Dict:
        """Get a summary of the taxonomy for quick overview"""
        if not self.taxonomy:
            return {'error': 'Taxonomy not built. Call build_taxonomy() first.'}

        summary = {
            'total_personas': len(self.personas),
            'meta_types': {},
            'risk_distribution': {},
            'highest_risk_personas': [],
            'recommended_priorities': []
        }

        # Count by meta-type and risk
        for persona in self.personas:
            style = persona['cognitive_style'].value
            risk = persona.get('risk_level', 'MEDIUM')

            summary['meta_types'][style] = summary['meta_types'].get(style, 0) + 1
            summary['risk_distribution'][risk] = summary['risk_distribution'].get(risk, 0) + 1

        # Top 3 highest risk personas
        sorted_by_risk = sorted(
            self.personas,
            key=lambda p: p.get('phishing_click_rate', 0),
            reverse=True
        )
        summary['highest_risk_personas'] = [
            {
                'name': p['persona_name'],
                'click_rate': f"{p.get('phishing_click_rate', 0) * 100:.1f}%",
                'cognitive_style': p['cognitive_style'].value,
                'risk_level': p.get('risk_level', 'MEDIUM'),
                'n_participants': p.get('n_participants', 0)
            }
            for p in sorted_by_risk[:3]
        ]

        # Priority interventions
        critical_intuitive = [
            p for p in self.personas
            if p['cognitive_style'] == CognitiveStyle.INTUITIVE
            and p.get('risk_level', '').lower() in ['critical', 'high']
        ]
        if critical_intuitive:
            summary['recommended_priorities'].append({
                'priority': 1,
                'group': 'High-Risk Intuitive Decision-Makers',
                'n_personas': len(critical_intuitive),
                'action': 'Implement habit-based interventions immediately'
            })

        return summary

    def to_dict(self) -> Dict:
        """Convert entire taxonomy to dictionary"""
        if not self.taxonomy:
            return {'error': 'Taxonomy not built. Call build_taxonomy() first.'}

        return {
            'taxonomy': self.taxonomy.to_dict(),
            'summary': self.get_summary(),
            'meta': {
                'total_personas': len(self.personas),
                'levels': {
                    1: 'Meta-types (Cognitive Style)',
                    2: 'Risk Profiles',
                    3: 'Individual Personas'
                },
                'cognitive_styles': [s.value for s in CognitiveStyle],
                'risk_levels': [r.value for r in RiskLevel]
            }
        }

    def get_flat_tree(self) -> List[Dict]:
        """
        Get a flattened tree representation for UI rendering.

        Each item includes depth and parent info for tree visualization.
        """
        if not self.taxonomy:
            return []

        flat = []

        def traverse(node: TaxonomyNode, depth: int = 0, parent_id: Optional[str] = None):
            item = {
                'id': node.id,
                'name': node.name,
                'level': node.level,
                'depth': depth,
                'parent_id': parent_id,
                'description': node.description,
                'has_children': len(node.children) > 0,
                'n_children': len(node.children),
                **node.metadata
            }
            flat.append(item)

            for child in node.children:
                traverse(child, depth + 1, node.id)

        traverse(self.taxonomy)
        return flat
