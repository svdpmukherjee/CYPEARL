"""
CYPEARL Phase 1 API

New endpoints added:
- /data-quality - Verify proposal requirements
- /industry-analysis - Cross-domain analysis
- /process-metrics - Response time, hover, inspection by cluster
- /expert/* - Delphi method validation
- /persona/* - Naming and labeling
- /export/ai-personas - Phase 2 ready exports

Existing endpoints preserved:
- /summary, /features, /run, /optimize, /analyze/interactions, etc.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
import traceback
import json
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core logic imports
from core.data_loader import DataLoader
from core.preprocessor import DataPreprocessor
from core.s3_loader import s3_loader
from phase1.algorithms.clustering import ALGORITHMS
from phase1.metrics.calculator import MetricsCalculator
from phase1.characterization.characterizer import ClusterCharacterizer
from phase1.constants import OUTCOME_FEATURES, GET_ALL_CLUSTERING_FEATURES

router = APIRouter()

# =============================================================================
# STATE MANAGEMENT - Extended
# =============================================================================

class AppState:
    """Application state with extended support for new features."""
    participants: pd.DataFrame = None
    responses: pd.DataFrame = None
    email_stimuli: pd.DataFrame = None
    preprocessor: DataPreprocessor = None
    last_run_result: Dict = None
    optimization_cache: Dict = None
    
    # NEW: Expert validation state
    expert_ratings: List[Dict] = []
    delphi_round: int = 1
    
    # NEW: Persona labels
    persona_labels: Dict[int, Dict] = {}  # cluster_id -> {name, label, description}

state = AppState()

# Data paths - local fallback for development
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PARTICIPANTS_PATH = DATA_DIR / "phishing_study_participants.csv"
RESPONSES_PATH = DATA_DIR / "phishing_study_responses.csv"
EMAIL_STIMULI_PATH = DATA_DIR / "email_stimuli.csv"

# =============================================================================
# STARTUP
# =============================================================================

@router.on_event("startup")
async def startup_event():
    """Load data on startup from S3 or local files."""
    try:
        # Try S3 first, fallback to local files
        # S3 loader automatically handles fallback
        state.participants, state.responses = s3_loader.load_study_data()
        state.email_stimuli = s3_loader.load_email_stimuli()
        state.preprocessor = DataPreprocessor()

        # If S3 failed and local DataLoader is needed
        if state.participants is None or len(state.participants) == 0:
            print("Trying local DataLoader as fallback...")
            loader = DataLoader(str(PARTICIPANTS_PATH), str(RESPONSES_PATH))
            state.participants, state.responses = loader.load()

        # Merge email attributes into responses if both exist
        if state.responses is not None and state.email_stimuli is not None and len(state.email_stimuli) > 0:
            # Check which columns from email_stimuli are missing in responses
            email_cols_to_merge = []
            base_cols = ['email_type', 'sender_familiarity', 'urgency_level',
                         'framing_type', 'ground_truth', 'phishing_quality', 'subject_line']

            for col in base_cols:
                if col not in state.responses.columns and col in state.email_stimuli.columns:
                    email_cols_to_merge.append(col)

            if email_cols_to_merge:
                merge_cols = ['email_id'] + email_cols_to_merge
                email_attrs = state.email_stimuli[merge_cols].drop_duplicates('email_id')
                state.responses = state.responses.merge(email_attrs, on='email_id', how='left')
                print(f"✓ Merged {len(email_cols_to_merge)} email attributes: {', '.join(email_cols_to_merge)}")
            else:
                print(f"✓ Email attributes already present in responses")

        print(f"✓ Data source: {s3_loader.get_data_source()}")
        print(f"✓ Loaded {len(state.participants) if state.participants is not None else 0} participants")
        if state.responses is not None:
            print(f"✓ Loaded {len(state.responses)} responses")
        if state.email_stimuli is not None:
            print(f"✓ Loaded {len(state.email_stimuli)} email stimuli")
    except Exception as e:
        print(f"⚠ Warning: Could not load default data on startup: {e}")
        traceback.print_exc()

# =============================================================================
# DATA REFRESH ENDPOINT
# =============================================================================

@router.post("/refresh-data")
async def refresh_data():
    """
    Refresh data from S3 (or local fallback).
    Call this after new participants complete the experiment to reload latest data.
    """
    try:
        state.participants, state.responses = s3_loader.refresh()
        state.email_stimuli = s3_loader.load_email_stimuli()

        # Re-merge email attributes if needed
        if state.responses is not None and state.email_stimuli is not None and len(state.email_stimuli) > 0:
            base_cols = ['email_type', 'sender_familiarity', 'urgency_level',
                         'framing_type', 'ground_truth', 'phishing_quality', 'subject_line']
            email_cols_to_merge = [col for col in base_cols
                                   if col not in state.responses.columns
                                   and col in state.email_stimuli.columns]
            if email_cols_to_merge:
                merge_cols = ['email_id'] + email_cols_to_merge
                email_attrs = state.email_stimuli[merge_cols].drop_duplicates('email_id')
                state.responses = state.responses.merge(email_attrs, on='email_id', how='left')

        return {
            "status": "success",
            "data_source": s3_loader.get_data_source(),
            "participants": len(state.participants) if state.participants is not None else 0,
            "responses": len(state.responses) if state.responses is not None else 0,
            "email_stimuli": len(state.email_stimuli) if state.email_stimuli is not None else 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/data-source")
async def get_data_source():
    """
    Get detailed information about the current data source (S3 or local).
    Useful for debugging and verifying S3 connectivity.
    """
    loader_status = s3_loader.get_status()

    return {
        **loader_status,
        "data_loaded": {
            "participants": len(state.participants) if state.participants is not None else 0,
            "responses": len(state.responses) if state.responses is not None else 0,
            "email_stimuli": len(state.email_stimuli) if state.email_stimuli is not None else 0
        }
    }


# =============================================================================
# REQUEST/RESPONSE MODELS - Extended
# =============================================================================

class ClusteringRequest(BaseModel):
    algorithm: str = "kmeans"
    k: int = 5
    features: Optional[List[str]] = None
    use_pca: bool = True
    pca_variance: float = 0.90
    random_state: int = 42
    min_cluster_size: int = 30
    industry_filter: Optional[List[str]] = None  # NEW: filter by industry

class OptimizationRequest(BaseModel):
    algorithm: str = "all"
    k_min: int = 2
    k_max: int = 15
    use_pca: bool = True
    pca_variance: float = 0.90
    weights: Optional[Dict[str, float]] = None
    min_cluster_size: int = 30

# NEW: Expert validation models
class ExpertRating(BaseModel):
    expert_id: str
    expert_name: str
    expert_role: str
    cluster_id: int
    realism: int  # 1-7
    distinctiveness: int  # 1-7
    actionability: int  # 1-7
    comments: Optional[str] = None

class PersonaLabel(BaseModel):
    cluster_id: int
    name: str  # e.g., "Trusting Novice"
    archetype: Optional[str] = None  # e.g., "High-Risk Impulsive"
    description: Optional[str] = None

# =============================================================================
# NEW ENDPOINT: DATA QUALITY CHECK
# =============================================================================

@router.get("/data-quality")
async def check_data_quality():
    """
    Check if data meets ARA proposal requirements.
    
    Requirements:
    - 700+ participants
    - 3+ industries (healthcare, education, finance)
    - 16 emails with factorial design
    - Behavioral tracking (click, report, hover, inspect, response time)
    """
    if state.participants is None:
        return {"status": "no_data", "checks": [], "all_passed": False}
    
    checks = []
    
    # 1. Participant count
    n_participants = len(state.participants)
    checks.append({
        "name": "Minimum Participants",
        "target": "≥ 700",
        "actual": n_participants,
        "passed": n_participants >= 700,
        "severity": "critical"
    })
    
    # 2. Industry coverage
    industries = []
    if 'industry' in state.participants.columns:
        industries = state.participants['industry'].unique().tolist()
    target_industries = ['healthcare', 'education', 'finance']
    covered = [ind for ind in target_industries if any(ind.lower() in str(i).lower() for i in industries)]
    checks.append({
        "name": "Industry Coverage",
        "target": "healthcare, education, finance",
        "actual": industries,
        "covered": covered,
        "passed": len(industries) >= 3,
        "severity": "critical"
    })
    
    # 3. Email stimuli count
    n_emails = 0
    if state.responses is not None and 'email_id' in state.responses.columns:
        n_emails = state.responses['email_id'].nunique()
    checks.append({
        "name": "Email Stimuli",
        "target": "≥ 16 emails",
        "actual": n_emails,
        "passed": n_emails >= 16,
        "severity": "critical"
    })
    
    # 4. Email factorial design
    email_factors = {}
    if state.responses is not None:
        for factor in ['urgency_level', 'sender_familiarity', 'framing_type', 'email_type']:
            if factor in state.responses.columns:
                email_factors[factor] = state.responses[factor].unique().tolist()
    checks.append({
        "name": "Factorial Design",
        "target": "urgency × familiarity × framing × type",
        "actual": email_factors,
        "passed": len(email_factors) >= 3,
        "severity": "high"
    })
    
    # 5. Behavioral tracking columns
    required_behaviors = ['clicked', 'reported', 'response_latency_ms', 'hovered_link', 'inspected_sender']
    available_behaviors = []
    if state.responses is not None:
        available_behaviors = [b for b in required_behaviors if b in state.responses.columns]
    checks.append({
        "name": "Behavioral Tracking",
        "target": required_behaviors,
        "actual": available_behaviors,
        "missing": [b for b in required_behaviors if b not in available_behaviors],
        "passed": len(available_behaviors) >= 4,
        "severity": "high"
    })
    
    # 6. Psychological traits
    all_clustering_features = GET_ALL_CLUSTERING_FEATURES()
    available_features = [f for f in all_clustering_features if f in state.participants.columns]
    checks.append({
        "name": "Psychological Traits",
        "target": f"{len(all_clustering_features)} traits",
        "actual": len(available_features),
        "coverage": f"{len(available_features)}/{len(all_clustering_features)}",
        "passed": len(available_features) >= 20,
        "severity": "medium"
    })
    
    all_passed = all(c['passed'] for c in checks)
    
    return {
        "status": "complete",
        "checks": checks,
        "all_passed": all_passed,
        "summary": {
            "n_participants": n_participants,
            "n_industries": len(industries),
            "n_emails": n_emails,
            "n_features": len(available_features)
        }
    }

# =============================================================================
# NEW ENDPOINT: INDUSTRY ANALYSIS (Cross-Domain)
# =============================================================================

@router.get("/industry-analysis")
async def get_industry_analysis():
    """Analyze cluster distribution across industries."""
    try:
        if state.participants is None:
            raise HTTPException(status_code=400, detail="Data not loaded")
        
        if state.last_run_result is None:
            raise HTTPException(status_code=400, detail="Run clustering first")
        
        labels = state.last_run_result.get('_labels')
        if labels is None:
            raise HTTPException(status_code=500, detail="Labels not found")
        
        df = state.participants.copy()
        
        if 'industry' not in df.columns:
            return {"error": "Industry column not found", "domain_transferable": True}
        
        if len(labels) != len(df):
            df = df.iloc[:len(labels)].copy()
        
        df['cluster'] = labels
        industries = df['industry'].unique().tolist()
        n_clusters = len(np.unique(labels))
        
        # Cluster distribution by industry
        distribution = {}
        for ind in industries:
            ind_data = df[df['industry'] == ind]
            if len(ind_data) == 0:
                continue
            cluster_counts = ind_data['cluster'].value_counts().to_dict()
            cluster_pcts = {k: v / len(ind_data) * 100 for k, v in cluster_counts.items()}
            distribution[str(ind)] = {
                "n": int(len(ind_data)),
                "counts": {int(k): int(v) for k, v in cluster_counts.items()},
                "percentages": {int(k): float(v) for k, v in cluster_pcts.items()}
            }
        
        # Chi-squared test
        try:
            contingency = pd.crosstab(df['industry'], df['cluster'])
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (len(df) * min_dim)) if min_dim > 0 else 0
        except:
            chi2, p_value, dof, cramers_v = 0.0, 1.0, 0, 0.0
        
        # Convert numpy types to Python native types
        chi2 = float(chi2)
        p_value = float(p_value)
        dof = int(dof)
        cramers_v = float(cramers_v)
        
        if cramers_v < 0.1:
            interpretation = "Weak association: Personas are industry-independent"
        elif cramers_v < 0.3:
            interpretation = "Moderate association: Some industry patterns exist"
        else:
            interpretation = "Strong association: Consider industry-specific models"
        
        # IMPORTANT: Convert numpy.bool to Python bool!
        is_transferable = bool(cramers_v < 0.2 and p_value > 0.05)
        
        return {
            "industries": [str(i) for i in industries],
            "n_clusters": int(n_clusters),
            "distribution": distribution,
            "statistical_tests": {
                "chi_squared": chi2,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "degrees_of_freedom": dof
            },
            "interpretation": interpretation,
            "domain_transferable": is_transferable  # Now it's a Python bool, not numpy.bool
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in industry-analysis: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# =============================================================================
# NEW ENDPOINT: PROCESS METRICS BY CLUSTER
# =============================================================================

@router.get("/process-metrics")
async def get_process_metrics():
    """
    Get behavioral process metrics (response time, hover, inspection) by cluster.
    
    Proposal requirement: "Track behaviors (click/report rates, response time, 
    hover over link, sender inspection etc.)"
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    if state.responses is None:
        raise HTTPException(status_code=400, detail="Responses data not loaded")
    
    labels = state.last_run_result.get('_labels')
    if labels is None:
        raise HTTPException(status_code=500, detail="Labels not found")
    
    # Add cluster labels to participants
    participants = state.participants.copy()
    participants['cluster'] = labels
    
    # Merge with responses
    responses = state.responses.merge(
        participants[['participant_id', 'cluster']],
        on='participant_id',
        how='inner'
    )
    
    # Filter to phishing emails for relevant metrics
    if 'email_type' in responses.columns:
        phishing = responses[responses['email_type'] == 'phishing']
    else:
        phishing = responses
    
    n_clusters = len(np.unique(labels))
    
    metrics = {}
    for cluster_id in range(n_clusters):
        cluster_responses = phishing[phishing['cluster'] == cluster_id]
        
        cluster_metrics = {
            "n_responses": len(cluster_responses),
            "n_participants": participants[participants['cluster'] == cluster_id].shape[0]
        }
        
        # Response latency
        if 'response_latency_ms' in cluster_responses.columns:
            latency = cluster_responses['response_latency_ms'].dropna()
            cluster_metrics['response_latency'] = {
                "mean": float(latency.mean()),
                "median": float(latency.median()),
                "std": float(latency.std()),
                "q25": float(latency.quantile(0.25)),
                "q75": float(latency.quantile(0.75)),
                "fast_decisions": int((latency < 3000).sum()),  # < 3 seconds
                "fast_decision_pct": float((latency < 3000).mean() * 100)
            }
        
        # Dwell time
        if 'dwell_time_ms' in cluster_responses.columns:
            dwell = cluster_responses['dwell_time_ms'].dropna()
            cluster_metrics['dwell_time'] = {
                "mean": float(dwell.mean()),
                "median": float(dwell.median()),
                "std": float(dwell.std())
            }
        
        # Hover rate
        if 'hovered_link' in cluster_responses.columns:
            cluster_metrics['hover_rate'] = float(cluster_responses['hovered_link'].mean())
        
        # Sender inspection rate
        if 'inspected_sender' in cluster_responses.columns:
            cluster_metrics['sender_inspection_rate'] = float(cluster_responses['inspected_sender'].mean())
        
        # Click rate
        if 'clicked' in cluster_responses.columns:
            cluster_metrics['click_rate'] = float(cluster_responses['clicked'].mean())
        
        # Report rate
        if 'reported' in cluster_responses.columns:
            cluster_metrics['report_rate'] = float(cluster_responses['reported'].mean())
        
        # Confidence rating
        if 'confidence_rating' in cluster_responses.columns:
            cluster_metrics['confidence'] = {
                "mean": float(cluster_responses['confidence_rating'].mean()),
                "std": float(cluster_responses['confidence_rating'].std())
            }
        
        # Suspicion rating
        if 'suspicion_rating' in cluster_responses.columns:
            cluster_metrics['suspicion'] = {
                "mean": float(cluster_responses['suspicion_rating'].mean()),
                "std": float(cluster_responses['suspicion_rating'].std())
            }
        
        metrics[cluster_id] = cluster_metrics
    
    # Identify fast vs slow deciders
    cognitive_style = {}
    for cluster_id, m in metrics.items():
        if 'response_latency' in m:
            if m['response_latency']['fast_decision_pct'] > 50:
                style = "fast_intuitive"  # System 1 dominant
            elif m['response_latency']['fast_decision_pct'] < 20:
                style = "deliberate_analytical"  # System 2 dominant
            else:
                style = "balanced"
        else:
            style = "unknown"
        cognitive_style[cluster_id] = style
    
    return {
        "by_cluster": metrics,
        "cognitive_style": cognitive_style,
        "boundary_conditions": {
            "fast_deciders": [c for c, s in cognitive_style.items() if s == "fast_intuitive"],
            "note": "Fast deciders (System 1) may be difficult for AI to simulate accurately"
        }
    }

# =============================================================================
# NEW ENDPOINT: AGGRESSIVE CONTENT ANALYSIS
# =============================================================================

@router.get("/aggressive-content")
async def get_aggressive_content_analysis():
    """
    Analyze how clusters respond to aggressive/emotionally manipulative content.
    
    Proposal specifically mentions "aggressive (e.g., emotionally manipulative) 
    phishing emails" as a key focus.
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    if state.responses is None:
        raise HTTPException(status_code=400, detail="Responses data not loaded")
    
    # Use framing_type (threat/reward) from factorial design instead of separate has_aggressive_content
    if 'framing_type' not in state.responses.columns:
        return {"error": "framing_type column not found - required for emotional susceptibility analysis"}

    labels = state.last_run_result.get('_labels')
    participants = state.participants.copy()
    participants['cluster'] = labels

    responses = state.responses.merge(
        participants[['participant_id', 'cluster']],
        on='participant_id',
        how='inner'
    )

    # Filter to phishing only
    if 'email_type' in responses.columns:
        phishing = responses[responses['email_type'] == 'phishing']
    else:
        phishing = responses

    n_clusters = len(np.unique(labels))

    # Compare threat-framed (emotional manipulation) vs reward-framed emails
    analysis = {}
    for cluster_id in range(n_clusters):
        cluster_data = phishing[phishing['cluster'] == cluster_id]

        # Threat framing = emotional manipulation tactic
        threat_framed = cluster_data[cluster_data['framing_type'] == 'threat']
        reward_framed = cluster_data[cluster_data['framing_type'] == 'reward']

        threat_click_rate = threat_framed['clicked'].mean() if len(threat_framed) > 0 else 0
        reward_click_rate = reward_framed['clicked'].mean() if len(reward_framed) > 0 else 0

        # Effect = difference in susceptibility (threat vs reward framing)
        effect = threat_click_rate - reward_click_rate

        analysis[cluster_id] = {
            "n_threat_framed": len(threat_framed),
            "n_reward_framed": len(reward_framed),
            "threat_click_rate": float(threat_click_rate),
            "reward_click_rate": float(reward_click_rate),
            "framing_effect_size": float(effect),
            "susceptibility": "high" if effect > 0.1 else "moderate" if effect > 0.05 else "low"
        }

    # Identify clusters most susceptible to threat framing
    most_susceptible = sorted(
        analysis.items(),
        key=lambda x: x[1]['framing_effect_size'],
        reverse=True
    )[:3]

    return {
        "by_cluster": analysis,
        "most_susceptible_to_threat_framing": [
            {"cluster": c, "effect": a['framing_effect_size']}
            for c, a in most_susceptible
        ],
        "boundary_condition_note": "Clusters highly susceptible to threat framing (emotional manipulation) may be difficult for AI to simulate"
    }

# =============================================================================
# NEW ENDPOINTS: EXPERT VALIDATION (Delphi Method)
# =============================================================================

@router.post("/expert/rate")
async def submit_expert_rating(rating: ExpertRating):
    """
    Submit expert rating for a cluster (Delphi method).
    
    Proposal: "Expert validation using Delphi method: ICC ≥ 0.70 for 
    realism, distinctiveness, actionability"
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    # Validate rating values
    for dim in [rating.realism, rating.distinctiveness, rating.actionability]:
        if not 1 <= dim <= 7:
            raise HTTPException(status_code=400, detail="Ratings must be 1-7")
    
    # Store rating
    rating_dict = rating.dict()
    rating_dict['timestamp'] = datetime.now().isoformat()
    rating_dict['delphi_round'] = state.delphi_round
    state.expert_ratings.append(rating_dict)
    
    return {
        "status": "recorded",
        "rating": rating_dict,
        "total_ratings": len(state.expert_ratings)
    }

@router.get("/expert/ratings")
async def get_expert_ratings():
    """Get all expert ratings for current round."""
    current_round_ratings = [
        r for r in state.expert_ratings 
        if r.get('delphi_round') == state.delphi_round
    ]
    
    return {
        "delphi_round": state.delphi_round,
        "ratings": current_round_ratings,
        "n_ratings": len(current_round_ratings)
    }

@router.get("/expert/icc")
async def calculate_icc():
    """
    Calculate Intra-class Correlation Coefficient for expert ratings.
    
    Target: ICC ≥ 0.70 for consensus
    """
    current_round_ratings = [
        r for r in state.expert_ratings 
        if r.get('delphi_round') == state.delphi_round
    ]
    
    if len(current_round_ratings) < 6:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 6 ratings, have {len(current_round_ratings)}",
            "delphi_round": state.delphi_round
        }
    
    # Organize ratings by cluster and expert
    df = pd.DataFrame(current_round_ratings)
    
    results = {}
    for dimension in ['realism', 'distinctiveness', 'actionability']:
        # Pivot to get experts as columns, clusters as rows
        try:
            pivot = df.pivot_table(
                index='cluster_id', 
                columns='expert_id', 
                values=dimension,
                aggfunc='first'
            )
            
            if pivot.shape[1] < 2:
                results[dimension] = {"icc": None, "status": "insufficient_experts"}
                continue
            
            # Calculate ICC(2,1) - two-way random, single measures
            # Simplified calculation
            n_clusters = pivot.shape[0]
            n_experts = pivot.shape[1]
            
            # Between-cluster variance
            cluster_means = pivot.mean(axis=1)
            grand_mean = cluster_means.mean()
            ms_between = n_experts * ((cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1)
            
            # Within-cluster variance
            ms_within = ((pivot.values - cluster_means.values.reshape(-1, 1)) ** 2).sum() / (n_clusters * (n_experts - 1))
            
            # ICC calculation
            icc = (ms_between - ms_within) / (ms_between + (n_experts - 1) * ms_within)
            icc = max(0, min(1, icc))  # Bound to [0, 1]
            
            results[dimension] = {
                "icc": float(icc),
                "meets_threshold": icc >= 0.70,
                "interpretation": "excellent" if icc >= 0.75 else "good" if icc >= 0.60 else "moderate" if icc >= 0.40 else "poor"
            }
        except Exception as e:
            results[dimension] = {"icc": None, "error": str(e)}
    
    # Overall consensus
    icc_values = [r['icc'] for r in results.values() if r.get('icc') is not None]
    consensus_reached = all(r.get('meets_threshold', False) for r in results.values())
    
    return {
        "delphi_round": state.delphi_round,
        "by_dimension": results,
        "mean_icc": float(np.mean(icc_values)) if icc_values else None,
        "consensus_reached": consensus_reached,
        "recommendation": "Proceed to Phase 2" if consensus_reached else "Continue to next Delphi round"
    }

@router.post("/expert/next-round")
async def advance_delphi_round():
    """Advance to next Delphi round."""
    state.delphi_round += 1
    return {
        "new_round": state.delphi_round,
        "previous_ratings_preserved": True
    }

# =============================================================================
# NEW ENDPOINTS: PERSONA NAMING/LABELING
# =============================================================================

@router.put("/persona/{cluster_id}/label")
async def set_persona_label(cluster_id: int, label: PersonaLabel):
    """Set human-readable name and label for a cluster/persona."""
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    state.persona_labels[cluster_id] = {
        "name": label.name,
        "archetype": label.archetype,
        "description": label.description,
        "updated_at": datetime.now().isoformat()
    }
    
    return {
        "status": "updated",
        "cluster_id": cluster_id,
        "label": state.persona_labels[cluster_id]
    }

@router.get("/persona/labels")
async def get_all_persona_labels():
    """Get all persona labels."""
    return {
        "labels": state.persona_labels,
        "n_labeled": len(state.persona_labels)
    }


# =============================================================================
# NEW ENDPOINT: HIERARCHICAL PERSONA TAXONOMY
# =============================================================================

from phase1.analysis.hierarchical_taxonomy import HierarchicalPersonaTaxonomy
from phase1.analysis.systematic_naming import merge_with_llm_names, generate_systematic_names

@router.get("/taxonomy")
async def get_persona_taxonomy():
    """
    Build and return hierarchical taxonomy of discovered personas.

    The taxonomy organizes personas into:
    - Level 1: Meta-types (Analytical vs Intuitive vs Balanced cognitive styles)
    - Level 2: Risk profiles within each meta-type (Critical/High/Medium/Low)
    - Level 3: Individual personas

    This helps business managers:
    1. See the "big picture" of the persona landscape
    2. Make strategic decisions at different granularities
    3. Target interventions at the right level of abstraction
    4. Communicate findings more effectively to stakeholders
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    clusters = state.last_run_result.get('clusters', {})
    if not clusters:
        raise HTTPException(status_code=400, detail="No clusters found in clustering result")

    try:
        taxonomy_builder = HierarchicalPersonaTaxonomy()
        taxonomy_builder.build_taxonomy(clusters, state.persona_labels)

        return {
            "status": "success",
            **taxonomy_builder.to_dict()
        }
    except Exception as e:
        print(f"[Taxonomy] Error building taxonomy: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/taxonomy/flat")
async def get_flat_taxonomy():
    """
    Get flattened taxonomy for tree visualization in UI.

    Returns a flat list with depth and parent_id for each node,
    making it easy to render as an expandable tree in the frontend.
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    clusters = state.last_run_result.get('clusters', {})
    if not clusters:
        raise HTTPException(status_code=400, detail="No clusters found in clustering result")

    try:
        taxonomy_builder = HierarchicalPersonaTaxonomy()
        taxonomy_builder.build_taxonomy(clusters, state.persona_labels)

        return {
            "status": "success",
            "nodes": taxonomy_builder.get_flat_tree(),
            "summary": taxonomy_builder.get_summary()
        }
    except Exception as e:
        print(f"[Taxonomy] Error building flat taxonomy: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/taxonomy/summary")
async def get_taxonomy_summary():
    """
    Get summary of persona taxonomy for quick overview.

    Includes:
    - Distribution by cognitive style (meta-type)
    - Distribution by risk level
    - Top 3 highest risk personas
    - Recommended intervention priorities
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    clusters = state.last_run_result.get('clusters', {})
    if not clusters:
        raise HTTPException(status_code=400, detail="No clusters found in clustering result")

    try:
        taxonomy_builder = HierarchicalPersonaTaxonomy()
        taxonomy_builder.build_taxonomy(clusters, state.persona_labels)

        return {
            "status": "success",
            **taxonomy_builder.get_summary()
        }
    except Exception as e:
        print(f"[Taxonomy] Error getting taxonomy summary: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NEW ENDPOINT: AI-POWERED PERSONA NAMING
# =============================================================================

class GenerateNamesRequest(BaseModel):
    """Request for AI-generated persona names."""
    clusters: List[Dict[str, Any]]  # Cluster data with traits and behavioral outcomes


@router.post("/persona/generate-names")
async def generate_persona_names(request: GenerateNamesRequest):
    """
    Generate descriptive persona names using Claude 3.5 Sonnet via OpenRouter.

    Uses the persona's psychological traits and behavioral outcomes to create
    meaningful, research-friendly names that describe each persona type.
    """
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_TOKEN")
    print(f"[AI Naming] OPENROUTER_TOKEN loaded: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print("[AI Naming] ERROR: OPENROUTER_TOKEN not found in environment")
        raise HTTPException(status_code=500, detail="OPENROUTER_TOKEN not configured in .env file")

    if not request.clusters:
        raise HTTPException(status_code=400, detail="No clusters provided")
    
    print(f"[AI Naming] Generating names for {len(request.clusters)} clusters")

    try:
        # Build the prompt for Claude with ALL traits organized by category
        cluster_descriptions = []
        for i, cluster in enumerate(request.clusters):
            cluster_id = cluster.get('cluster_id', i)

            # Extract all trait z-scores
            trait_zscores = cluster.get('trait_zscores', {})
            top_high_traits = cluster.get('top_high_traits', [])
            top_low_traits = cluster.get('top_low_traits', [])

            # Extract behavioral outcomes
            behavioral = cluster.get('behavioral_outcomes', {})
            phishing_click_rate = cluster.get('phishing_click_rate', 0)
            risk_level = cluster.get('risk_level', 'MEDIUM')

            # Format high traits (top 5)
            high_traits_str = ", ".join([f"{t[0].replace('_', ' ')} (+{t[1]:.2f}σ)" for t in top_high_traits[:5]]) if top_high_traits else "none notable"

            # Format low traits (top 5)
            low_traits_str = ", ".join([f"{t[0].replace('_', ' ')} ({t[1]:.2f}σ)" for t in top_low_traits[:5]]) if top_low_traits else "none notable"

            # Behavioral metrics
            click_rate = phishing_click_rate * 100 if phishing_click_rate < 1 else phishing_click_rate
            overall_accuracy = behavioral.get('overall_accuracy', {}).get('mean', 0) * 100 if behavioral.get('overall_accuracy', {}).get('mean', 0) < 1 else behavioral.get('overall_accuracy', {}).get('mean', 0)
            report_rate = behavioral.get('report_rate', {}).get('mean', 0) * 100 if behavioral.get('report_rate', {}).get('mean', 0) < 1 else behavioral.get('report_rate', {}).get('mean', 0)
            phishing_detection = behavioral.get('phishing_detection_rate', {}).get('mean', 0) * 100 if behavioral.get('phishing_detection_rate', {}).get('mean', 0) < 1 else behavioral.get('phishing_detection_rate', {}).get('mean', 0)
            hover_rate = behavioral.get('hover_rate', {}).get('mean', 0) * 100 if behavioral.get('hover_rate', {}).get('mean', 0) < 1 else behavioral.get('hover_rate', {}).get('mean', 0)
            sender_inspect = behavioral.get('sender_inspection_rate', {}).get('mean', 0) * 100 if behavioral.get('sender_inspection_rate', {}).get('mean', 0) < 1 else behavioral.get('sender_inspection_rate', {}).get('mean', 0)
            mean_latency = behavioral.get('mean_response_latency', {}).get('mean', 0)

            # Format ALL traits by category
            cognitive_traits = f"CRT={trait_zscores.get('crt_score', 0):.2f}σ, need_for_cognition={trait_zscores.get('need_for_cognition', 0):.2f}σ, working_memory={trait_zscores.get('working_memory', 0):.2f}σ"
            
            personality_traits = f"impulsivity={trait_zscores.get('impulsivity_total', 0):.2f}σ, trust={trait_zscores.get('trust_propensity', 0):.2f}σ, risk_taking={trait_zscores.get('risk_taking', 0):.2f}σ, sensation_seeking={trait_zscores.get('sensation_seeking', 0):.2f}σ, extraversion={trait_zscores.get('big5_extraversion', 0):.2f}σ, conscientiousness={trait_zscores.get('big5_conscientiousness', 0):.2f}σ, neuroticism={trait_zscores.get('big5_neuroticism', 0):.2f}σ"
            
            state_traits = f"anxiety={trait_zscores.get('state_anxiety', 0):.2f}σ, stress={trait_zscores.get('current_stress', 0):.2f}σ, fatigue={trait_zscores.get('fatigue_level', 0):.2f}σ"
            
            attitude_traits = f"self_efficacy={trait_zscores.get('phishing_self_efficacy', 0):.2f}σ, perceived_risk={trait_zscores.get('perceived_risk', 0):.2f}σ, security_attitudes={trait_zscores.get('security_attitudes', 0):.2f}σ, privacy_concern={trait_zscores.get('privacy_concern', 0):.2f}σ"
            
            experience_traits = f"phishing_knowledge={trait_zscores.get('phishing_knowledge', 0):.2f}σ, tech_expertise={trait_zscores.get('technical_expertise', 0):.2f}σ, prior_victim={trait_zscores.get('prior_victimization', 0):.2f}σ, security_training={trait_zscores.get('security_training', 0):.2f}σ"
            
            susceptibility_traits = f"authority_susceptibility={trait_zscores.get('authority_susceptibility', 0):.2f}σ, urgency_susceptibility={trait_zscores.get('urgency_susceptibility', 0):.2f}σ, scarcity_susceptibility={trait_zscores.get('scarcity_susceptibility', 0):.2f}σ"

            desc = f"""Cluster {cluster_id} (0-indexed):

== BEHAVIORAL OUTCOMES (8 metrics) ==
- Risk Level: {risk_level}
- Phishing Click Rate: {click_rate:.1f}%
- Phishing Detection Rate: {phishing_detection:.1f}%
- Overall Accuracy: {overall_accuracy:.1f}%
- Report Rate: {report_rate:.1f}%
- Hover Rate: {hover_rate:.1f}%
- Sender Inspection Rate: {sender_inspect:.1f}%
- Mean Response Latency: {mean_latency:.0f}ms

== MOST DISTINCTIVE TRAITS ==
- HIGH (above average): {high_traits_str}
- LOW (below average): {low_traits_str}

== ALL 29 PSYCHOLOGICAL TRAITS BY CATEGORY ==
Cognitive (3): {cognitive_traits}
Personality (7): {personality_traits}
State (3): {state_traits}
Attitudes (4): {attitude_traits}
Experience (4): {experience_traits}
Susceptibility (3): {susceptibility_traits}"""
            cluster_descriptions.append(desc)

        all_descriptions = "\n\n".join(cluster_descriptions)

        system_prompt = """You are a research assistant helping name behavioral personas discovered through clustering analysis of phishing susceptibility research.

Your task is to create DESCRIPTIVE names for each persona that capture their unique psychological profile and behavioral patterns.

Guidelines for naming:
1. Names should be 3-5 words that capture the persona's ESSENCE
2. Focus on the MOST DISTINCTIVE traits - look at the HIGH and LOW trait lists
3. Consider the interplay between cognitive style (CRT, need_for_cognition), personality (impulsivity, trust), and susceptibility factors
4. Names should reflect behavioral outcomes (click rate, report rate, response latency)
5. Use professional, neutral language suitable for research publications
6. Each name must clearly distinguish this persona from others

Naming strategies based on trait patterns:
- High impulsivity + high click rate → emphasize quick/reactive nature
- High CRT + low trust + high report rate → emphasize analytical/skeptical nature  
- High urgency/authority susceptibility → emphasize compliance tendency
- Low anxiety + high security attitudes → emphasize confident/aware nature
- High fatigue/stress + moderate accuracy → emphasize overwhelmed/burdened nature

Examples of good names:
- "Vigilant Skeptic" (high CRT, low trust, high report rate)
- "Reactive Compliant" (high impulsivity, high authority susceptibility)
- "Cautious Security-Minded" (low impulsivity, high security attitudes, low click rate)
- "Overwhelmed Trusting" (high fatigue, high trust, moderate click rate)
- "Analytical Deliberator" (high need_for_cognition, low urgency susceptibility, slow response)"""

        user_prompt = f"""Based on the following persona profiles from a phishing susceptibility study, generate a DESCRIPTIVE name for each persona.

I have provided ALL 29 psychological traits and 8 behavioral outcomes for each cluster. Focus on:
1. The MOST DISTINCTIVE traits (listed under HIGH and LOW)
2. Key behavioral patterns (click rate, report rate, response latency)
3. The interplay between cognitive, personality, and susceptibility factors

{all_descriptions}

IMPORTANT: Use the exact cluster_id numbers shown above (0-indexed). Generate names for clusters 0 through {len(request.clusters) - 1}.

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"personas": [
  {{"cluster_id": 0, "name": "3-5 Word Name", "archetype": "10-15 word description explaining key traits and behaviors"}},
  {{"cluster_id": 1, "name": "3-5 Word Name", "archetype": "10-15 word description explaining key traits and behaviors"}}
]}}

Generate names for all {len(request.clusters)} clusters (cluster_id 0 to {len(request.clusters) - 1})."""

        # Call OpenRouter API
        print(f"[AI Naming] Calling OpenRouter API with model: anthropic/claude-3.5-sonnet")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://cypearl.research.edu",
                    "X-Title": "CYPEARL Phase 1",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )

            print(f"[AI Naming] OpenRouter response status: {response.status_code}")
            
            if response.status_code != 200:
                error_text = response.text
                print(f"[AI Naming] ERROR: {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenRouter API error: {error_text}"
                )

            data = response.json()
            content = data['choices'][0]['message']['content']

            # Parse the JSON response
            try:
                # Clean up the response in case it has markdown code blocks
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()

                result = json.loads(content)
                generated_names = result.get('personas', [])
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return raw content for debugging
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse AI response as JSON: {content[:500]}"
                )

            # Step 1: Store raw LLM names temporarily
            print(f"[AI Naming] Successfully generated {len(generated_names)} LLM persona names")
            llm_labels = {}
            for persona in generated_names:
                cluster_id = persona.get('cluster_id', 0)
                llm_labels[cluster_id] = {
                    "name": persona.get('name', f'Persona {cluster_id + 1}'),
                    "archetype": persona.get('archetype', ''),
                }
                print(f"[AI Naming] LLM name for Cluster {cluster_id}: {persona.get('name')}")

            # Step 2: Merge with systematic codes to create hybrid names
            # Format: "CR-INT-IMP-CLK: Impulsive Phish Prone"
            clusters = state.last_run_result.get('clusters', {})
            if clusters:
                print(f"[AI Naming] Merging with systematic codes for {len(clusters)} clusters...")
                hybrid_labels = merge_with_llm_names(clusters, llm_labels)

                # Update state.persona_labels with hybrid names
                for cluster_id, label_data in hybrid_labels.items():
                    state.persona_labels[cluster_id] = {
                        "name": label_data['name'],  # "CR-INT-IMP-CLK: Creative Name"
                        "archetype": label_data['archetype'],  # "CR-INT-IMP-CLK"
                        "readable_code": label_data.get('readable_code', ''),
                        "llm_name": label_data.get('llm_name', ''),
                        "description": label_data['description'],
                        "components": label_data.get('components', {}),
                        "ai_generated": True,
                        "updated_at": datetime.now().isoformat()
                    }
                    print(f"[AI Naming] Hybrid name for Cluster {cluster_id}: {label_data['name']}")

                # Update generated_names to include hybrid format
                for persona in generated_names:
                    cid = persona.get('cluster_id', 0)
                    if cid in hybrid_labels:
                        persona['hybrid_name'] = hybrid_labels[cid]['name']
                        persona['systematic_code'] = hybrid_labels[cid]['archetype']
            else:
                # Fallback: No cluster data, use LLM names only
                print(f"[AI Naming] No cluster data available, using LLM names only")
                for persona in generated_names:
                    cluster_id = persona.get('cluster_id', 0)
                    state.persona_labels[cluster_id] = {
                        "name": persona.get('name', f'Persona {cluster_id + 1}'),
                        "archetype": persona.get('archetype', ''),
                        "description": persona.get('archetype', ''),
                        "ai_generated": True,
                        "updated_at": datetime.now().isoformat()
                    }

            return {
                "status": "success",
                "generated_names": generated_names,
                "labels": state.persona_labels,
                "model_used": "meta-llama/llama-3.3-70b-instruct",
                "naming_mode": "hybrid"
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[AI Naming] Error generating persona names: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# NEW ENDPOINT: AI EXPORT (Phase 2 Preparation)
# =============================================================================

@router.get("/export/ai-personas")
async def export_ai_personas(format: str = "full"):
    """
    Export persona definitions for Phase 2 AI agent creation.
    
    Formats:
    - baseline: Task-only, minimal description
    - stats: + behavioral statistics
    - full: + chain-of-thought reasoning examples
    
    Proposal: "three prompt configurations: (a) task-only baseline, 
    (b) augmented with behavioral statistics, (c) augmented with reasoning traces"
    """
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    clusters = state.last_run_result.get('clusters', {})
    labels = state.last_run_result.get('_labels')
    
    personas = []
    for cluster_id, cluster_data in clusters.items():
        # Get human label if set
        label_info = state.persona_labels.get(int(cluster_id), {})
        
        persona = {
            "persona_id": f"PERSONA_{cluster_id}",
            "cluster_id": int(cluster_id),
            "name": label_info.get('name', f"Persona {cluster_id}"),
            "archetype": label_info.get('archetype'),
            "risk_level": cluster_data.get('risk_level', 'MEDIUM'),
            "n_participants": cluster_data.get('n_participants', 0),
            "description": cluster_data.get('description', '')
        }
        
        # Add behavioral statistics if requested
        if format in ['stats', 'full']:
            persona['behavioral_statistics'] = {
                "phishing_click_rate": cluster_data.get('phishing_click_rate', 0),
                "overall_accuracy": cluster_data.get('behavioral_outcomes', {}).get('overall_accuracy', {}).get('mean'),
                "report_rate": cluster_data.get('behavioral_outcomes', {}).get('report_rate', {}).get('mean'),
                "mean_response_latency_ms": cluster_data.get('behavioral_outcomes', {}).get('mean_response_latency', {}).get('mean'),
                "hover_rate": cluster_data.get('behavioral_outcomes', {}).get('hover_rate', {}).get('mean'),
                "sender_inspection_rate": cluster_data.get('behavioral_outcomes', {}).get('sender_inspection_rate', {}).get('mean'),
            }
            
            persona['psychological_profile'] = {
                "distinguishing_high_traits": [t[0] for t in cluster_data.get('top_high_traits', [])],
                "distinguishing_low_traits": [t[0] for t in cluster_data.get('top_low_traits', [])],
                "trait_zscores": {k: round(v, 2) for k, v in cluster_data.get('trait_zscores', {}).items()},
            }
            
            # Determine cognitive style
            impulsivity = cluster_data.get('trait_zscores', {}).get('impulsivity_total', 0)
            crt = cluster_data.get('trait_zscores', {}).get('crt_score', 0)
            if impulsivity > 0.5:
                cognitive_style = "impulsive"
            elif crt > 0.5:
                cognitive_style = "analytical"
            else:
                cognitive_style = "balanced"
            persona['psychological_profile']['cognitive_style'] = cognitive_style
        
        # Add chain-of-thought examples if full format
        if format == 'full':
            persona['reasoning_examples'] = generate_reasoning_examples(cluster_data)
            persona['boundary_conditions'] = identify_boundary_conditions(cluster_data)
        
        # Target accuracy for Phase 2 validation
        persona['target_accuracy'] = 0.85
        persona['acceptance_range'] = [0.80, 0.90]
        
        personas.append(persona)
    
    return {
        "format": format,
        "n_personas": len(personas),
        "personas": personas,
        "export_timestamp": datetime.now().isoformat(),
        "phase2_models": [
            "Claude Sonnet 4.5",
            "Amazon Nova Pro",
            "Llama 4 Maverick",
            "Mistral Large",
            "Cohere Command R+"
        ]
    }

def generate_reasoning_examples(cluster_data):
    """Generate chain-of-thought reasoning examples for AI prompting."""
    examples = []
    
    risk_level = cluster_data.get('risk_level', 'MEDIUM')
    traits = cluster_data.get('trait_zscores', {})
    
    # High urgency susceptibility
    if traits.get('urgency_susceptibility', 0) > 0.5:
        examples.append({
            "scenario": "high_urgency_phishing",
            "email_cues": "Subject: URGENT - Your account will be suspended in 24 hours",
            "reasoning": "I feel anxious seeing the urgent warning. The threat of losing account access triggers my fear response. Without pausing to verify, I feel compelled to click immediately to prevent the negative outcome.",
            "action": "click",
            "confidence": "high"
        })
    
    # High trust propensity
    if traits.get('trust_propensity', 0) > 0.5:
        examples.append({
            "scenario": "familiar_sender_phishing",
            "email_cues": "From: IT Department <support@company-helpdesk.com>",
            "reasoning": "This appears to be from IT, a department I trust. The email looks professional. I don't typically question messages from internal teams.",
            "action": "click",
            "confidence": "medium"
        })
    
    # High analytical/CRT
    if traits.get('crt_score', 0) > 0.5:
        examples.append({
            "scenario": "sophisticated_phishing",
            "email_cues": "Request to verify account with link to external domain",
            "reasoning": "Let me analyze this carefully. First, I'll check the sender domain - it doesn't match our company. The link preview shows an unfamiliar URL. The request is unusual - IT never asks for credentials via email. This seems suspicious.",
            "action": "report",
            "confidence": "high"
        })
    
    # Low risk persona
    if risk_level == 'LOW':
        examples.append({
            "scenario": "obvious_phishing",
            "email_cues": "Poor grammar, suspicious link, unknown sender",
            "reasoning": "Multiple red flags here: spelling errors, pressure tactics, suspicious link. I've been trained to recognize these patterns. Reporting immediately.",
            "action": "report",
            "confidence": "high"
        })
    
    return examples

def identify_boundary_conditions(cluster_data):
    """Identify likely AI failure cases for this persona."""
    conditions = []
    traits = cluster_data.get('trait_zscores', {})
    
    if traits.get('impulsivity_total', 0) > 0.7:
        conditions.append({
            "type": "fast_heuristic_decisions",
            "description": "High impulsivity leads to instant reactions that LLMs may over-deliberate",
            "severity": "high"
        })
    
    if traits.get('urgency_susceptibility', 0) > 0.7:
        conditions.append({
            "type": "emotional_urgency_response",
            "description": "Strong emotional reaction to urgency that LLMs may not authentically simulate",
            "severity": "high"
        })
    
    if traits.get('state_anxiety', 0) > 0.7:
        conditions.append({
            "type": "anxiety_driven_behavior",
            "description": "Anxiety-based decisions may not translate to LLM reasoning",
            "severity": "medium"
        })
    
    if traits.get('crt_score', 0) < -0.7:
        conditions.append({
            "type": "intuitive_non_analytical",
            "description": "Gut-feel decisions that LLMs tend to over-rationalize",
            "severity": "medium"
        })
    
    return conditions

# =============================================================================
# EXISTING ENDPOINTS (Preserved from original)
# =============================================================================

@router.get("/summary")
async def get_data_summary():
    """Get comprehensive data summary."""
    if state.participants is None:
        return {"status": "no_data", "message": "Data not loaded"}
    
    all_features = GET_ALL_CLUSTERING_FEATURES()
    available_features = [f for f in all_features if f in state.participants.columns]
    
    email_factors = {}
    if state.responses is not None:
        for col in ['urgency_level', 'sender_familiarity', 'framing_type', 'email_type']:
            if col in state.responses.columns:
                email_factors[col] = state.responses[col].unique().tolist()
    
    # NEW: Add industry info
    industries = []
    if 'industry' in state.participants.columns:
        industries = state.participants['industry'].unique().tolist()
    
    return {
        "status": "loaded",
        "n_participants": len(state.participants),
        "n_features": len(state.participants.columns),
        "n_clustering_features": len(available_features),
        "n_responses": len(state.responses) if state.responses is not None else 0,
        "n_emails": state.responses['email_id'].nunique() if state.responses is not None and 'email_id' in state.responses.columns else 0,
        "features": list(state.participants.columns),
        "clustering_features": available_features,
        "email_factors": email_factors,
        "industries": industries  # NEW
    }

@router.get("/features")
async def get_features():
    """Get detailed feature information."""
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    from phase1.constants import CLUSTERING_FEATURES, OUTCOME_FEATURES, DEMOGRAPHIC_FEATURES
    
    return {
        "clustering_features": CLUSTERING_FEATURES,
        "outcome_features": OUTCOME_FEATURES,
        "demographic_features": DEMOGRAPHIC_FEATURES,
        "available_in_data": {
            "clustering": [f for f in GET_ALL_CLUSTERING_FEATURES() if f in state.participants.columns],
            "outcomes": [f for f in OUTCOME_FEATURES if f in state.participants.columns],
            "demographics": [f for f in DEMOGRAPHIC_FEATURES if f in state.participants.columns]
        }
    }

@router.post("/run")
async def run_clustering(req: ClusteringRequest):
    """Run clustering with specified configuration."""
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithm {req.algorithm} not supported")
    
    try:
        # NEW: Apply industry filter if specified
        df = state.participants.copy()
        if req.industry_filter and 'industry' in df.columns:
            df = df[df['industry'].isin(req.industry_filter)]
            if len(df) < 50:
                raise HTTPException(status_code=400, detail="Industry filter results in too few participants")
        
        # Configure preprocessor
        state.preprocessor.config.use_pca = req.use_pca
        state.preprocessor.config.pca_variance = req.pca_variance
        
        # Fit transform
        X = state.preprocessor.fit_transform(df, req.features)
        features_used = state.preprocessor.feature_names
        pca_components = X.shape[1] if req.use_pca else None
        
        # Run clustering
        algo = ALGORITHMS[req.algorithm]
        labels = algo.fit_predict(X, req.k, req.random_state)
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(df, OUTCOME_FEATURES)
        metrics = metrics_calc.calculate_all(X, labels, req.k)
        
        # Characterize clusters
        characterizer = ClusterCharacterizer(df, features_used)
        clusters = characterizer.characterize(labels)
        
        # Prepare scatter data (2D visualization)
        vis_coords = X[:, :2] if X.shape[1] >= 2 else X
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca_vis = PCA(n_components=2)
            vis_coords = pca_vis.fit_transform(X)
        
        scatter = []
        for i in range(len(labels)):
            scatter.append({
                "x": float(vis_coords[i, 0]) if vis_coords.shape[1] > 0 else 0,
                "y": float(vis_coords[i, 1]) if vis_coords.shape[1] > 1 else 0,
                "cluster": int(labels[i]),
                "id": str(df.iloc[i].get('participant_id', i))
            })
        
        response = {
            "k": req.k,
            "algorithm": req.algorithm,
            "n_clusters": len(clusters),
            "metrics": metrics,
            "clusters": clusters,
            "scatter_data": scatter,
            "features_used": features_used,
            "pca_components": pca_components,
            "industry_filter": req.industry_filter  # NEW
        }
        
        # Store for later analysis
        state.last_run_result = {
            **response,
            "_labels": labels,
            "_X": X,
            "_df": df  # Store filtered df
        }
        
        return response
        
    except Exception as e:
        print(f"Error in clustering: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_clustering(req: OptimizationRequest):
    """Optimize clustering across K values and algorithms."""
    import time
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[OPTIMIZE] Starting K-Sweep optimization")
    print(f"[OPTIMIZE] Config: algorithm={req.algorithm}, K={req.k_min}-{req.k_max}, PCA={req.use_pca}")
    print(f"{'='*60}")

    if state.participants is None:
        print("[OPTIMIZE] ERROR: Data not loaded")
        raise HTTPException(status_code=400, detail="Data not loaded")

    print(f"[OPTIMIZE] Data loaded: {len(state.participants)} participants")

    try:
        from phase1.analysis.optimizer import ClusteringOptimizer

        # Configure preprocessor
        state.preprocessor.config.use_pca = req.use_pca
        state.preprocessor.config.pca_variance = req.pca_variance

        optimizer = ClusteringOptimizer(
            state.participants,
            state.preprocessor,
            state.preprocessor.feature_names or GET_ALL_CLUSTERING_FEATURES()
        )

        # Determine algorithms
        if req.algorithm == "all":
            algos = list(ALGORITHMS.keys())
        else:
            algos = [req.algorithm]

        print(f"[OPTIMIZE] Running {len(algos)} algorithm(s): {algos}")
        print(f"[OPTIMIZE] Testing K values from {req.k_min} to {req.k_max}")

        results = optimizer.optimize(algos, req.k_min, req.k_max, req.use_pca)

        # Cache results
        state.optimization_cache = results

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[OPTIMIZE] COMPLETE in {elapsed:.2f}s")
        print(f"[OPTIMIZE] Results: {len(results)} algorithms processed")
        print(f"{'='*60}\n")

        return results

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[OPTIMIZE] ERROR after {elapsed:.2f}s: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/interactions")
async def analyze_interactions():
    """Analyze cluster × email type interactions."""
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    if state.responses is None:
        raise HTTPException(status_code=400, detail="Responses data not loaded")
    
    labels = state.last_run_result.get('_labels')
    if labels is None:
        raise HTTPException(status_code=500, detail="Labels not found in state")
    
    try:
        participants = state.participants.copy()
        participants['cluster'] = labels
        
        if 'participant_id' not in state.responses.columns:
            return {"error": "Missing participant_id in responses"}
        if 'participant_id' not in participants.columns:
            return {"error": "Missing participant_id in participants"}
        
        responses = state.responses.merge(
            participants[['participant_id', 'cluster']],
            on='participant_id',
            how='inner'
        )
        
        if len(responses) == 0:
            return {"error": "No matching participants found"}
        
        if 'email_type' in responses.columns:
            phishing = responses[responses['email_type'] == 'phishing'].copy()
        else:
            phishing = responses.copy()

        if len(phishing) == 0:
            return {"error": "No phishing emails found"}

        if 'clicked' not in phishing.columns:
            return {"error": "No 'clicked' column found"}

        # Normalize sender_familiarity values (known/unknown -> familiar/unfamiliar)
        if 'sender_familiarity' in phishing.columns:
            familiarity_map = {'known': 'familiar', 'unknown': 'unfamiliar'}
            phishing['sender_familiarity'] = phishing['sender_familiarity'].map(
                lambda x: familiarity_map.get(x, x)
            )
        if 'sender_familiarity' in responses.columns:
            familiarity_map = {'known': 'familiar', 'unknown': 'unfamiliar'}
            responses['sender_familiarity'] = responses['sender_familiarity'].map(
                lambda x: familiarity_map.get(x, x)
            )
        
        results = {
            'by_urgency': {},
            'by_familiarity': {},
            'by_framing': {},
            'by_aggressive': {},
            'by_email_type': {},
            'interaction_effects': {},
            'summary': {
                'n_responses': len(responses),
                'n_phishing': len(phishing),
                'n_clusters': len(np.unique(labels))
            }
        }
        
        def safe_pivot_to_dict(df, factor_col, value_col='clicked'):
            try:
                if factor_col not in df.columns:
                    return {}
                pivot = df.groupby(['cluster', factor_col])[value_col].mean().unstack()
                return {str(col): {int(idx): float(val) for idx, val in pivot[col].items()} 
                        for col in pivot.columns}
            except Exception as e:
                return {}
        
        results['by_urgency'] = safe_pivot_to_dict(phishing, 'urgency_level')
        results['by_familiarity'] = safe_pivot_to_dict(phishing, 'sender_familiarity')
        results['by_framing'] = safe_pivot_to_dict(phishing, 'framing_type')
        # Note: by_aggressive removed - use by_framing (threat vs reward) for emotional manipulation analysis
        results['by_email_type'] = safe_pivot_to_dict(responses, 'email_type')
        
        # Calculate interaction effects
        if 'urgency_level' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'urgency_level'])['clicked'].mean().unstack()
                if 'high' in pivot.columns and 'low' in pivot.columns:
                    diff = pivot['high'] - pivot['low']
                    results['interaction_effects']['urgency_effect'] = {int(k): float(v) for k, v in diff.items()}
            except:
                pass
        
        if 'sender_familiarity' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'sender_familiarity'])['clicked'].mean().unstack()
                if 'familiar' in pivot.columns and 'unfamiliar' in pivot.columns:
                    diff = pivot['familiar'] - pivot['unfamiliar']
                    results['interaction_effects']['familiarity_effect'] = {int(k): float(v) for k, v in diff.items()}
            except:
                pass
        
        if 'framing_type' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'framing_type'])['clicked'].mean().unstack()
                if 'threat' in pivot.columns and 'reward' in pivot.columns:
                    diff = pivot['threat'] - pivot['reward']
                    results['interaction_effects']['framing_effect'] = {int(k): float(v) for k, v in diff.items()}
            except:
                pass
        
        return results
        
    except Exception as e:
        print(f"Error in interaction analysis: {e}")
        traceback.print_exc()
        return {"error": str(e)}

@router.get("/algorithms")
async def get_algorithms():
    """Get available clustering algorithms."""
    return {
        name: {
            "name": algo.name,
            "display_name": algo.display_name,
            "supports_soft": algo.supports_soft
        }
        for name, algo in ALGORITHMS.items()
    }

@router.get("/email-stimuli")
async def get_email_stimuli():
    """Get email stimuli information."""
    if state.email_stimuli is None:
        if state.responses is not None and 'email_id' in state.responses.columns:
            emails = state.responses.groupby('email_id').agg({
                'email_type': 'first',
                'urgency_level': 'first' if 'urgency_level' in state.responses.columns else lambda x: None,
                'sender_familiarity': 'first' if 'sender_familiarity' in state.responses.columns else lambda x: None,
                'clicked': 'mean'
            }).reset_index().to_dict(orient='records')
            return {"emails": emails, "source": "responses"}
        return {"emails": [], "message": "No email stimuli data available"}
    
    return {
        "emails": state.email_stimuli.to_dict(orient='records'),
        "total": len(state.email_stimuli),
        "source": "stimuli_file"
    }

@router.get("/cluster/{cluster_id}")
async def get_cluster_details(cluster_id: int):
    """Get detailed information about a specific cluster."""
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    clusters = state.last_run_result.get('clusters', {})
    
    if cluster_id not in clusters:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    
    # Add persona label if exists
    cluster_data = clusters[cluster_id].copy()
    if cluster_id in state.persona_labels:
        cluster_data['persona_label'] = state.persona_labels[cluster_id]
    
    return cluster_data

@router.post("/export/assignments")
async def export_assignments():
    """Export cluster assignments."""
    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")
    
    labels = state.last_run_result.get('_labels')
    if labels is None:
        raise HTTPException(status_code=500, detail="Labels not found")
    
    assignments = state.participants[['participant_id']].copy()
    assignments['cluster'] = labels
    
    # Add persona names if available
    if state.persona_labels:
        assignments['persona_name'] = assignments['cluster'].map(
            lambda c: state.persona_labels.get(c, {}).get('name', f'Cluster {c}')
        )
    
    return {
        "assignments": assignments.to_dict(orient='records'),
        "config": {
            "algorithm": state.last_run_result.get('algorithm'),
            "k": state.last_run_result.get('k'),
            "features_used": state.last_run_result.get('features_used')
        },
        "persona_labels": state.persona_labels
    }

# =============================================================================
# SCIENTIFIC VALIDATION ENDPOINTS
# =============================================================================

# Import validation modules
from phase1.validation import (
    ClusteringValidator,
    GapStatisticAnalyzer,
    FeatureImportanceAnalyzer,
    ClusteringCrossValidator,
    PredictionErrorAnalyzer,
    ConsensusClusteringAnalyzer,
    AlgorithmComparisonAnalyzer,
    SoftAssignmentAnalyzer
)
from phase1.validation.cluster_visualization import ClusterVisualizationGenerator

class ValidationRequest(BaseModel):
    level: str = "standard"  # 'basic', 'standard', 'comprehensive'
    use_current_clustering: bool = True  # Use last run result
    k: Optional[int] = None  # Override K if not using current
    algorithm: Optional[str] = None

class GapStatisticRequest(BaseModel):
    k_min: int = 2
    k_max: int = 15

class FeatureImportanceRequest(BaseModel):
    k: Optional[int] = None

class CrossValidationRequest(BaseModel):
    k: Optional[int] = None
    n_folds: int = 5
    algorithm: str = "kmeans"

class AlgorithmComparisonRequest(BaseModel):
    k: Optional[int] = None
    algorithms: List[str] = ["kmeans", "gmm", "hierarchical"]
    metric: str = "composite"

class ClusterVisualizationRequest(BaseModel):
    method: str = "pca"  # 'pca' or 'tsne'
    perplexity: int = 30  # t-SNE perplexity


@router.post("/validate")
async def run_validation(request: ValidationRequest):
    """
    Run comprehensive clustering validation.

    Validates clustering quality with scientific rigor before
    proceeding to LLM persona conditioning.

    Levels:
    - basic: Gap statistic + cross-validation (fast)
    - standard: Basic + feature importance + prediction error (recommended)
    - comprehensive: Standard + consensus + algorithm comparison (thorough)
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    # Get clustering result
    if request.use_current_clustering and state.last_run_result is not None:
        labels = np.array(state.last_run_result.get('_labels', []))
        k = state.last_run_result.get('k')
        feature_names = state.last_run_result.get('features_used', GET_ALL_CLUSTERING_FEATURES())
    else:
        raise HTTPException(status_code=400, detail="Run clustering first or set use_current_clustering=False")

    if len(labels) == 0:
        raise HTTPException(status_code=400, detail="No clustering labels found")

    # Prepare data
    X = state.preprocessor.fit_transform(state.participants, feature_names)
    outcome_data = state.participants[OUTCOME_FEATURES].copy() if all(
        c in state.participants.columns for c in OUTCOME_FEATURES
    ) else None

    # Run validation
    try:
        validator = ClusteringValidator(
            n_bootstrap=50,
            n_cv_folds=5,
            n_consensus_iterations=30,
            random_state=42
        )

        report = validator.validate(
            X=X,
            labels=labels,
            k=k,
            feature_names=feature_names,
            outcome_data=outcome_data,
            outcome_cols=OUTCOME_FEATURES if outcome_data is not None else None,
            level=request.level
        )

        return report.to_dict()

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/validate/quick")
async def quick_validation():
    """
    Quick validation check (minimal analysis).

    Returns essential metrics without full analysis.
    Use for rapid iteration during development.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    labels = np.array(state.last_run_result.get('_labels', []))
    k = state.last_run_result.get('k')
    feature_names = state.last_run_result.get('features_used', GET_ALL_CLUSTERING_FEATURES())

    X = state.preprocessor.fit_transform(state.participants, feature_names)
    outcome_data = state.participants[OUTCOME_FEATURES].copy() if all(
        c in state.participants.columns for c in OUTCOME_FEATURES
    ) else None

    validator = ClusteringValidator()
    return validator.quick_validate(
        X=X,
        labels=labels,
        k=k,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES if outcome_data is not None else None
    )


@router.post("/validate/llm-readiness")
async def llm_readiness_validation():
    """
    Check if clustering is ready for LLM persona conditioning.

    Specialized validation focusing on metrics most relevant
    to whether clusters will produce effective persona prompts.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    labels = np.array(state.last_run_result.get('_labels', []))
    k = state.last_run_result.get('k')
    feature_names = state.last_run_result.get('features_used', GET_ALL_CLUSTERING_FEATURES())

    X = state.preprocessor.fit_transform(state.participants, feature_names)

    # Outcome data is required for LLM readiness check
    if not all(c in state.participants.columns for c in OUTCOME_FEATURES):
        raise HTTPException(status_code=400, detail="Outcome data required for LLM readiness check")

    outcome_data = state.participants[OUTCOME_FEATURES].copy()

    validator = ClusteringValidator()
    return validator.validate_for_llm_readiness(
        X=X,
        labels=labels,
        k=k,
        feature_names=feature_names,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES
    )


@router.post("/validate/gap-statistic")
async def gap_statistic_analysis(request: GapStatisticRequest):
    """
    Run Gap Statistic analysis for optimal K selection.

    Provides data-driven recommendation for number of clusters.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    feature_names = GET_ALL_CLUSTERING_FEATURES()
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    analyzer = GapStatisticAnalyzer(n_references=20, random_state=42)
    results = analyzer.analyze(X, k_min=request.k_min, k_max=request.k_max)

    # Add elbow comparison
    results['elbow_comparison'] = analyzer.compare_to_elbow(X, request.k_min, request.k_max)

    return results


@router.post("/validate/feature-importance")
async def feature_importance_analysis(request: FeatureImportanceRequest):
    """
    Analyze which features drive cluster separation.

    Identifies important vs. noise features for potential simplification.
    Note: PCA is disabled for this analysis to evaluate original features.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    feature_names = GET_ALL_CLUSTERING_FEATURES()

    # IMPORTANT: Disable PCA to analyze original features, not PCA components
    X = state.preprocessor.fit_transform(state.participants, feature_names, use_pca=False)

    # Get K from request or last run
    k = request.k
    if k is None:
        if state.last_run_result is not None:
            k = state.last_run_result.get('k', 6)
        else:
            k = 6

    outcome_data = state.participants[OUTCOME_FEATURES].copy() if all(
        c in state.participants.columns for c in OUTCOME_FEATURES
    ) else None

    analyzer = FeatureImportanceAnalyzer(n_permutations=20, random_state=42)
    return analyzer.analyze(
        X=X,
        feature_names=feature_names,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES if outcome_data is not None else None,
        k=k
    )


@router.post("/validate/cross-validation")
async def cross_validation_analysis(request: CrossValidationRequest):
    """
    Run cross-validation to assess generalization.

    Validates that clustering quality holds on unseen data.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    feature_names = GET_ALL_CLUSTERING_FEATURES()
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    k = request.k
    if k is None:
        if state.last_run_result is not None:
            k = state.last_run_result.get('k', 6)
        else:
            k = 6

    outcome_data = state.participants[OUTCOME_FEATURES].copy() if all(
        c in state.participants.columns for c in OUTCOME_FEATURES
    ) else None

    analyzer = ClusteringCrossValidator(n_folds=request.n_folds, random_state=42)
    return analyzer.cross_validate(
        X=X,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES if outcome_data is not None else None,
        k=k,
        algorithm=request.algorithm
    )


@router.post("/validate/prediction-error")
async def prediction_error_analysis():
    """
    Analyze predictive power of clusters.

    Measures whether cluster membership actually predicts behavior.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    labels = np.array(state.last_run_result.get('_labels', []))
    feature_names = state.last_run_result.get('features_used', GET_ALL_CLUSTERING_FEATURES())

    X = state.preprocessor.fit_transform(state.participants, feature_names)

    if not all(c in state.participants.columns for c in OUTCOME_FEATURES):
        raise HTTPException(status_code=400, detail="Outcome data required")

    outcome_data = state.participants[OUTCOME_FEATURES].copy()

    analyzer = PredictionErrorAnalyzer(random_state=42)
    return analyzer.analyze(
        X=X,
        labels=labels,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES
    )


@router.post("/validate/consensus")
async def consensus_clustering_analysis():
    """
    Run consensus clustering for robust cluster discovery.

    Identifies core members vs. boundary cases across multiple runs.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    k = state.last_run_result.get('k', 6) if state.last_run_result else 6
    feature_names = GET_ALL_CLUSTERING_FEATURES()
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    analyzer = ConsensusClusteringAnalyzer(n_iterations=50, random_state=42)
    return analyzer.analyze(X, k, algorithms=['kmeans', 'gmm'])


@router.post("/validate/algorithm-comparison")
async def algorithm_comparison_analysis(request: AlgorithmComparisonRequest):
    """
    Statistically compare clustering algorithms.

    Determines if differences between algorithms are significant.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    k = request.k
    if k is None:
        if state.last_run_result is not None:
            k = state.last_run_result.get('k', 6)
        else:
            k = 6

    feature_names = GET_ALL_CLUSTERING_FEATURES()
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    outcome_data = state.participants[OUTCOME_FEATURES].copy() if all(
        c in state.participants.columns for c in OUTCOME_FEATURES
    ) else None

    analyzer = AlgorithmComparisonAnalyzer(n_bootstrap=50, random_state=42)
    return analyzer.compare(
        X=X,
        algorithms=request.algorithms,
        k=k,
        outcome_data=outcome_data,
        outcome_cols=OUTCOME_FEATURES if outcome_data is not None else None,
        metric=request.metric
    )


@router.post("/validate/soft-assignments")
async def soft_assignments_analysis():
    """
    Get probabilistic cluster assignments.

    Provides uncertainty quantification for cluster membership.
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    k = state.last_run_result.get('k', 6) if state.last_run_result else 6
    feature_names = GET_ALL_CLUSTERING_FEATURES()
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    analyzer = SoftAssignmentAnalyzer(random_state=42)
    results = analyzer.analyze(X, k, algorithm='gmm')

    # Remove full assignment matrix to reduce response size
    results.pop('soft_assignments', None)
    results.pop('soft_probability_matrix', None)

    return results


@router.post("/validate/cluster-visualization")
async def cluster_visualization(request: ClusterVisualizationRequest):
    """
    Generate 2D visualization of cluster assignments.

    Projects high-dimensional clustering results to 2D for visualization.
    Includes:
    - Data points colored by cluster
    - Cluster centroids
    - Cluster statistics (compactness, size)
    - Axis interpretation (for PCA)

    Methods:
    - pca: Fast, preserves global structure (recommended)
    - tsne: Non-linear, preserves local structure (slower)
    """
    if state.participants is None:
        raise HTTPException(status_code=400, detail="Load data first")

    if state.last_run_result is None:
        raise HTTPException(status_code=400, detail="Run clustering first")

    labels = np.array(state.last_run_result.get('_labels', []))
    if len(labels) == 0:
        raise HTTPException(status_code=400, detail="No clustering labels found")

    feature_names = state.last_run_result.get('features_used', GET_ALL_CLUSTERING_FEATURES())
    X = state.preprocessor.fit_transform(state.participants, feature_names)

    try:
        generator = ClusterVisualizationGenerator(random_state=42)

        # Generate visualization with feature information
        result = generator.generate_with_features(
            X=X,
            labels=labels,
            feature_names=feature_names,
            method=request.method
        )

        # Add persona labels if available
        result['persona_labels'] = {}
        for cluster_id in range(result['k']):
            if cluster_id in state.persona_labels:
                result['persona_labels'][cluster_id] = state.persona_labels[cluster_id].get('name', f'Cluster {cluster_id}')
            else:
                result['persona_labels'][cluster_id] = f'Cluster {cluster_id}'

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@router.get("/validate/summary")
async def get_validation_summary():
    """
    Get summary of all available validation methods.
    """
    return {
        "available_endpoints": [
            {
                "endpoint": "/validate",
                "method": "POST",
                "description": "Run comprehensive validation (basic/standard/comprehensive)",
                "params": {"level": "standard"}
            },
            {
                "endpoint": "/validate/quick",
                "method": "POST",
                "description": "Quick validation check for rapid iteration"
            },
            {
                "endpoint": "/validate/llm-readiness",
                "method": "POST",
                "description": "Check readiness for LLM persona conditioning"
            },
            {
                "endpoint": "/validate/gap-statistic",
                "method": "POST",
                "description": "Gap statistic for optimal K selection",
                "params": {"k_min": 2, "k_max": 15}
            },
            {
                "endpoint": "/validate/feature-importance",
                "method": "POST",
                "description": "Identify important vs. noise features"
            },
            {
                "endpoint": "/validate/cross-validation",
                "method": "POST",
                "description": "Cross-validation for generalization assessment"
            },
            {
                "endpoint": "/validate/prediction-error",
                "method": "POST",
                "description": "Measure predictive power of clusters"
            },
            {
                "endpoint": "/validate/consensus",
                "method": "POST",
                "description": "Consensus clustering for robustness"
            },
            {
                "endpoint": "/validate/algorithm-comparison",
                "method": "POST",
                "description": "Statistical comparison of algorithms"
            },
            {
                "endpoint": "/validate/soft-assignments",
                "method": "POST",
                "description": "Probabilistic cluster assignments"
            }
        ],
        "recommended_workflow": [
            "1. Run /validate/quick to get baseline metrics",
            "2. Run /validate/gap-statistic to confirm optimal K",
            "3. Run /validate with level='standard' for full validation",
            "4. Run /validate/llm-readiness before Phase 2"
        ],
        "thresholds": {
            "min_cv_test_silhouette": 0.1,
            "max_generalization_gap": 0.15,
            "min_prediction_r2": 0.05,
            "max_boundary_case_pct": 0.30
        }
    }