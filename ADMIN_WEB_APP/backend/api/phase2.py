"""
CYPEARL Phase 2 API - AI Persona Simulation

API endpoints for Phase 2:
- /providers/* - Configure LLM providers
- /models/* - Model registry and health
- /personas/* - Import and manage personas from Phase 1
- /emails/* - Manage email stimuli
- /experiments/* - Create and run experiments
- /results/* - Access and analyze results
- /analysis/* - Fidelity metrics and comparisons
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime
import asyncio
import traceback

from core.schemas import (
    Persona, EmailStimulus, ExperimentConfig, SimulationTrial,
    PromptConfiguration, ExperimentStatus, ProviderType,
    ImportPersonasRequest, CreateExperimentRequest, ProviderSetupRequest
)
from core.config import config, MODEL_REGISTRY, PROVIDER_CONFIGS
from phase2.providers import get_router, LLMRequest
from phase2.simulation import ExecutionEngine, PromptBuilder
from phase2.analysis import FidelityAnalyzer
from phase2.calibration.calibration_logger import create_calibration_log
from phase2.simulation.experiment_logger import create_experiment_log

router = APIRouter()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class Phase2State:
    """Application state for Phase 2."""
    personas: Dict[str, Persona] = {}
    emails: Dict[str, EmailStimulus] = {}
    experiments: Dict[str, ExperimentConfig] = {}
    results: Dict[str, List[SimulationTrial]] = {}

    # Running experiments
    running_experiment_id: Optional[str] = None
    experiment_progress: Dict = {}

    # Experiment logs (experiment_id -> log_file_path)
    experiment_logs: Dict[str, str] = {}

    # Engine instance
    engine: Optional[ExecutionEngine] = None

state = Phase2State()

# Data directory for default email stimuli
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_EMAILS_PATH = DATA_DIR / "email_stimuli_phase2.json"


def load_default_emails():
    """Load default email stimuli from JSON file if state is empty."""
    if state.emails:
        return  # Already have emails loaded

    if not DEFAULT_EMAILS_PATH.exists():
        print(f"Warning: Default emails file not found: {DEFAULT_EMAILS_PATH}")
        return

    try:
        with open(DEFAULT_EMAILS_PATH, 'r') as f:
            emails_data = json.load(f)

        for email_dict in emails_data:
            email = EmailStimulus(
                email_id=email_dict.get('email_id', f"E{len(state.emails)+1}"),
                email_type=email_dict.get('email_type', 'phishing'),
                sender_familiarity=email_dict.get('sender_familiarity', 'unfamiliar'),
                urgency_level=email_dict.get('urgency_level', 'low'),
                framing_type=email_dict.get('framing_type', 'neutral'),
                content_domain=email_dict.get('content_domain', 'general'),
                has_aggressive_content=email_dict.get('has_aggressive_content', False),
                has_spelling_errors=email_dict.get('has_spelling_errors', False),
                has_suspicious_url=email_dict.get('has_suspicious_url', False),
                requests_sensitive_info=email_dict.get('requests_sensitive_info', False),
                phishing_quality=email_dict.get('phishing_quality'),
                ground_truth=email_dict.get('ground_truth', 1),
                aggression_level=email_dict.get('aggression_level'),
                subject_line=email_dict.get('subject_line'),
                sender_display=email_dict.get('sender_display'),
                sender_email=email_dict.get('sender_email'),
                body_text=email_dict.get('body_text'),
                link_url=email_dict.get('link_url'),
                link_display_text=email_dict.get('link_display_text'),
                sender=email_dict.get('sender'),
                body=email_dict.get('body')
            )
            state.emails[email.email_id] = email

        print(f"Loaded {len(state.emails)} default emails from {DEFAULT_EMAILS_PATH}")
    except Exception as e:
        print(f"Error loading default emails: {e}")


# =============================================================================
# PROVIDER ENDPOINTS
# =============================================================================

@router.get("/providers")
async def list_providers():
    """List all available providers and their configuration status."""
    router_instance = await get_router()
    initialized = router_instance.get_initialized_providers()
    
    providers = []
    for provider_type, config in PROVIDER_CONFIGS.items():
        providers.append({
            "provider_type": provider_type,
            "display_name": config['display_name'],
            "initialized": provider_type in initialized,
            "auth_type": config['auth_type'],
            "requires_region": config.get('requires_region', False),
        })
    
    return {"providers": providers}


@router.post("/providers/{provider_type}/setup")
async def setup_provider(provider_type: str, request: ProviderSetupRequest):
    """Configure and initialize a provider."""
    if provider_type not in PROVIDER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_type}")
    
    router_instance = await get_router()
    
    try:
        kwargs = {}
        
        # Common auth
        if request.api_key:
            kwargs['api_key'] = request.api_key
        if request.base_url:
            kwargs['base_url'] = request.base_url
            
        # AWS Bedrock specific
        if request.region:
            kwargs['region'] = request.region
        if request.aws_access_key_id:
            kwargs['aws_access_key_id'] = request.aws_access_key_id
        if request.aws_secret_access_key:
            kwargs['aws_secret_access_key'] = request.aws_secret_access_key
        if request.aws_session_token:
            kwargs['aws_session_token'] = request.aws_session_token
            
        # Additional options
        if request.timeout:
            kwargs['timeout'] = request.timeout
        if request.max_retries:
            kwargs['max_retries'] = request.max_retries
        
        success = await router_instance.initialize_provider(provider_type, **kwargs)
        
        return {
            "provider_type": provider_type,
            "initialized": success,
            "message": f"Provider {provider_type} initialized successfully" if success else "Failed to initialize"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))



@router.get("/providers/{provider_type}/health")
async def check_provider_health(provider_type: str):
    """Check health of a specific provider."""
    router_instance = await get_router()
    
    if provider_type not in router_instance.get_initialized_providers():
        return {"healthy": False, "error": "Provider not initialized"}
    
    # Check health for first model of this provider
    for model_id, model_config in MODEL_REGISTRY.items():
        if model_config['provider'] == provider_type:
            health = await router_instance.check_health(model_id)
            return health
    
    return {"healthy": False, "error": "No models for this provider"}


# =============================================================================
# MODEL ENDPOINTS
# =============================================================================

@router.get("/models")
async def list_models():
    """List all available models with their configurations."""
    router_instance = await get_router()
    models = router_instance.get_available_models()
    
    # Group by tier
    by_tier = {
        "frontier": [],
        "mid_tier": [],
        "open_source": [],
        "budget": []
    }
    
    for model in models:
        tier = model.get('tier', 'budget')
        by_tier[tier].append(model)
    
    return {
        "models": models,
        "by_tier": by_tier,
        "total": len(models)
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get details for a specific model."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    model = MODEL_REGISTRY[model_id]
    router_instance = await get_router()
    health = await router_instance.check_health(model_id)
    
    return {
        "model_id": model_id,
        **model,
        "health": health
    }


@router.post("/models/{model_id}/test")
async def test_model(model_id: str):
    """Send a test request to a model."""
    router_instance = await get_router()
    
    request = LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Model working!' in exactly 2 words.",
        temperature=0.0,
        max_tokens=10
    )
    
    try:
        response = await router_instance.complete(model_id, request)
        return {
            "success": response.success,
            "content": response.content,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "error": response.error_message
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/models/health")
async def check_all_models_health():
    """Check health of all models."""
    router_instance = await get_router()
    health = await router_instance.check_all_health()
    
    healthy_count = sum(1 for h in health.values() if h.get('healthy'))
    
    return {
        "health": health,
        "summary": {
            "total": len(health),
            "healthy": healthy_count,
            "unhealthy": len(health) - healthy_count
        }
    }


# =============================================================================
# PERSONA ENDPOINTS
# =============================================================================


@router.post("/personas/import")
async def import_personas(request: Dict[str, Any]):
    """
    Import personas from Phase 1 export.
    
    Accepts multiple formats:
    - { phase1_export: { personas: [...] } }
    - { personas: [...] }
    """
    try:
        # Handle various input formats
        if 'phase1_export' in request:
            data = request['phase1_export']
        else:
            data = request
        
        # Extract personas from different possible structures
        if 'personas' in data:
            personas_data = data['personas']
        elif 'clusters' in data:
            # Convert cluster format to persona format
            personas_data = []
            for cluster in data['clusters']:
                persona = {
                    'persona_id': f"PERSONA_{cluster.get('cluster_id', cluster.get('id', 0))}",
                    'cluster_id': cluster.get('cluster_id', cluster.get('id', 0)),
                    'name': cluster.get('name', cluster.get('label', f"Cluster {cluster.get('cluster_id', 0)}")),
                    'archetype': cluster.get('archetype', ''),
                    'risk_level': cluster.get('risk_level', 'MEDIUM'),
                    'n_participants': cluster.get('n_participants', cluster.get('size', 0)),
                    'pct_of_population': cluster.get('pct_of_population', 0),
                    'description': cluster.get('description', ''),
                    'trait_zscores': cluster.get('trait_zscores', cluster.get('centroid', {})),
                    'distinguishing_high_traits': cluster.get('distinguishing_high_traits', cluster.get('high_traits', [])),
                    'distinguishing_low_traits': cluster.get('distinguishing_low_traits', cluster.get('low_traits', [])),
                    'cognitive_style': cluster.get('cognitive_style', 'balanced'),
                    'behavioral_statistics': cluster.get('behavioral_statistics', cluster.get('behavior', {
                        'phishing_click_rate': cluster.get('phishing_click_rate', 0.3),
                        'overall_accuracy': cluster.get('overall_accuracy', 0.7),
                        'report_rate': cluster.get('report_rate', 0.2),
                        'mean_response_latency_ms': cluster.get('mean_response_latency_ms', 3000),
                        'hover_rate': cluster.get('hover_rate', 0.4),
                        'sender_inspection_rate': cluster.get('sender_inspection_rate', 0.3)
                    })),
                    'email_interaction_effects': cluster.get('email_interaction_effects', cluster.get('interaction_effects', {
                        'urgency_effect': cluster.get('urgency_effect', 0.1),
                        'familiarity_effect': cluster.get('familiarity_effect', 0.08),
                        'framing_effect': cluster.get('framing_effect', 0.05)
                    })),
                    'boundary_conditions': cluster.get('boundary_conditions', []),
                    'reasoning_examples': cluster.get('reasoning_examples', []),
                    'target_accuracy': 0.85,
                    'acceptance_range': [0.80, 0.90]
                }
                personas_data.append(persona)
        else:
            raise ValueError("Could not find personas or clusters in the provided data")
        
        # Validate and store personas
        imported = []
        for p_data in personas_data:
            # Convert to Persona model
            from core.schemas import (
                BehavioralStatistics, EmailInteractionEffects, 
                CognitiveStyle, RiskLevel, BoundaryCondition
            )
            
            # Handle behavioral_statistics - could be dict or already processed
            behav_data = p_data.get('behavioral_statistics', {})
            if isinstance(behav_data, dict):
                behavioral_stats = BehavioralStatistics(
                    phishing_click_rate=behav_data.get('phishing_click_rate', 0.3),
                    overall_accuracy=behav_data.get('overall_accuracy', 0.7),
                    report_rate=behav_data.get('report_rate', 0.2),
                    mean_response_latency_ms=behav_data.get('mean_response_latency_ms', 3000),
                    hover_rate=behav_data.get('hover_rate', 0.4),
                    sender_inspection_rate=behav_data.get('sender_inspection_rate', 0.3)
                )
            else:
                behavioral_stats = behav_data
            
            # Handle email_interaction_effects
            effect_data = p_data.get('email_interaction_effects', {})
            if isinstance(effect_data, dict):
                interaction_effects = EmailInteractionEffects(
                    urgency_effect=effect_data.get('urgency_effect', 0.1),
                    familiarity_effect=effect_data.get('familiarity_effect', 0.08),
                    framing_effect=effect_data.get('framing_effect', 0.05)
                )
            else:
                interaction_effects = effect_data
            
            # Handle cognitive_style
            cog_style = p_data.get('cognitive_style', 'balanced')
            if isinstance(cog_style, str):
                cog_style = CognitiveStyle(cog_style.lower())
            
            # Handle risk_level
            risk = p_data.get('risk_level', 'MEDIUM')
            if isinstance(risk, str):
                risk = RiskLevel(risk.upper())
            
            # Handle boundary_conditions
            boundary_conds = []
            for bc in p_data.get('boundary_conditions', []):
                if isinstance(bc, dict):
                    boundary_conds.append(BoundaryCondition(**bc))
                else:
                    boundary_conds.append(bc)
            
            persona = Persona(
                persona_id=p_data.get('persona_id', f"PERSONA_{p_data.get('cluster_id', 0)}"),
                cluster_id=p_data.get('cluster_id', 0),
                name=p_data.get('name', 'Unknown'),
                archetype=p_data.get('archetype'),
                risk_level=risk,
                n_participants=p_data.get('n_participants', 0),
                pct_of_population=p_data.get('pct_of_population', 0),
                description=p_data.get('description', ''),
                trait_zscores=p_data.get('trait_zscores', {}),
                distinguishing_high_traits=p_data.get('distinguishing_high_traits', []),
                distinguishing_low_traits=p_data.get('distinguishing_low_traits', []),
                cognitive_style=cog_style,
                behavioral_statistics=behavioral_stats,
                email_interaction_effects=interaction_effects,
                boundary_conditions=boundary_conds,
                reasoning_examples=p_data.get('reasoning_examples', []),
                target_accuracy=p_data.get('target_accuracy', 0.85),
                acceptance_range=p_data.get('acceptance_range', [0.80, 0.90])
            )
            
            state.personas[persona.persona_id] = persona
            imported.append(persona.persona_id)
        
        return {
            "success": True,
            "imported_count": len(imported),
            "persona_ids": imported
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


@router.get("/personas")
async def list_personas():
    """List all imported personas."""
    return {
        "personas": [
            {
                "persona_id": p.persona_id,
                "name": p.name,
                "archetype": p.archetype,
                "risk_level": p.risk_level.value,
                "n_participants": p.n_participants,
                "pct_of_population": p.pct_of_population,
                "phishing_click_rate": p.behavioral_statistics.phishing_click_rate,
                "cognitive_style": p.cognitive_style.value
            }
            for p in state.personas.values()
        ],
        "total": len(state.personas)
    }


@router.get("/personas/{persona_id}")
async def get_persona(persona_id: str):
    """Get detailed information for a specific persona."""
    if persona_id not in state.personas:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona = state.personas[persona_id]
    return persona.model_dump()


# =============================================================================
# EMAIL ENDPOINTS
# =============================================================================

@router.post("/emails/import")
async def import_emails(emails_data: List[Dict[str, Any]]):
    """Import email stimuli for Phase 2 testing."""
    imported = []
    
    for email_dict in emails_data:
        email = EmailStimulus(
            email_id=email_dict.get('email_id', f"E{len(state.emails)+1}"),
            email_type=email_dict.get('email_type', 'phishing'),
            sender_familiarity=email_dict.get('sender_familiarity', 'unfamiliar'),
            urgency_level=email_dict.get('urgency_level', 'low'),
            framing_type=email_dict.get('framing_type', 'neutral'),
            content_domain=email_dict.get('content_domain', 'general'),
            has_aggressive_content=email_dict.get('has_aggressive_content', False),
            has_spelling_errors=email_dict.get('has_spelling_errors', False),
            has_suspicious_url=email_dict.get('has_suspicious_url', False),
            requests_sensitive_info=email_dict.get('requests_sensitive_info', False),
            phishing_quality=email_dict.get('phishing_quality'),
            ground_truth=email_dict.get('ground_truth', 1),
            # Content fields
            subject_line=email_dict.get('subject_line'),
            sender_display=email_dict.get('sender_display'),
            sender_email=email_dict.get('sender_email'),
            body_text=email_dict.get('body_text'),
            link_url=email_dict.get('link_url'),
            link_display_text=email_dict.get('link_display_text'),
            # Legacy fields
            sender=email_dict.get('sender'),
            body=email_dict.get('body')
        )
        state.emails[email.email_id] = email
        imported.append(email.email_id)
    
    return {
        "success": True,
        "imported_count": len(imported),
        "email_ids": imported
    }


@router.get("/emails")
async def list_emails():
    """List all email stimuli. Auto-loads defaults if none exist."""
    # Auto-load default emails if state is empty
    load_default_emails()

    return {
        "emails": [
            {
                "email_id": e.email_id,
                "email_type": e.email_type,
                "sender_familiarity": e.sender_familiarity,
                "urgency_level": e.urgency_level,
                "framing_type": e.framing_type,
                "has_aggressive_content": e.has_aggressive_content,
                "ground_truth": e.ground_truth,
                # FIX Issue #5: Include full email content for display
                "subject_line": e.subject_line,
                "sender_display": e.sender_display,
                "sender_email": e.sender_email,
                "body_text": e.body_text,
                "link_url": e.link_url,
                "link_display_text": e.link_display_text,
                "content_domain": e.content_domain,
                # Legacy field names for compatibility
                "subject": e.subject_line,
                "sender": e.sender_display or e.sender,
                "body": e.body_text or e.body
            }
            for e in state.emails.values()
        ],
        "total": len(state.emails)
    }


@router.get("/emails/{email_id}")
async def get_email(email_id: str):
    """Get detailed information for a specific email."""
    if email_id not in state.emails:
        raise HTTPException(status_code=404, detail="Email not found")
    
    return state.emails[email_id].model_dump()


# =============================================================================
# EXPERIMENT ENDPOINTS
# =============================================================================

@router.post("/experiments/create")
async def create_experiment(request: CreateExperimentRequest):
    """Create a new experiment configuration."""
    import uuid
    
    # Validate selections
    for pid in request.persona_ids:
        if pid not in state.personas:
            raise HTTPException(status_code=400, detail=f"Unknown persona: {pid}")
    
    for eid in request.email_ids:
        if eid not in state.emails:
            raise HTTPException(status_code=400, detail=f"Unknown email: {eid}")
    
    # Calculate total trials
    n_conditions = (
        len(request.persona_ids) * 
        len(request.model_ids) * 
        len(request.prompt_configs) * 
        len(request.email_ids)
    )
    total_trials = n_conditions * request.trials_per_condition
    
    experiment = ExperimentConfig(
        experiment_id=str(uuid.uuid4())[:8],
        name=request.name,
        description=request.description,
        persona_ids=request.persona_ids,
        model_ids=request.model_ids,
        prompt_configs=request.prompt_configs,
        email_ids=request.email_ids,
        trials_per_condition=request.trials_per_condition,
        temperature=request.temperature,
        total_trials=total_trials
    )
    
    state.experiments[experiment.experiment_id] = experiment
    
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "total_trials": total_trials,
        "n_conditions": n_conditions,
        "status": experiment.status.value
    }


@router.get("/experiments")
async def list_experiments():
    """List all experiments."""
    return {
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "status": e.status.value,
                "total_trials": e.total_trials,
                "completed_trials": e.completed_trials,
                "created_at": e.created_at.isoformat()
            }
            for e in state.experiments.values()
        ],
        "total": len(state.experiments)
    }


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get detailed information for a specific experiment."""
    if experiment_id not in state.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp = state.experiments[experiment_id]
    return {
        **exp.model_dump(),
        "status": exp.status.value,
        "prompt_configs": [p.value for p in exp.prompt_configs]
    }


@router.post("/experiments/{experiment_id}/run")
async def run_experiment(experiment_id: str, background_tasks: BackgroundTasks):
    """Start running an experiment."""
    if experiment_id not in state.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if state.running_experiment_id:
        raise HTTPException(status_code=400, detail="Another experiment is already running")
    
    experiment = state.experiments[experiment_id]
    
    # Update status
    experiment.status = ExperimentStatus.RUNNING
    experiment.started_at = datetime.now()
    state.running_experiment_id = experiment_id
    state.experiment_progress = {
        "completed": 0,
        "total": experiment.total_trials,
        "failed": 0,
        "cost": 0.0
    }
    
    # Start background execution
    background_tasks.add_task(execute_experiment, experiment_id)
    
    return {
        "experiment_id": experiment_id,
        "status": "started",
        "total_trials": experiment.total_trials
    }


async def execute_experiment(experiment_id: str):
    """Background task to execute experiment."""
    experiment = state.experiments[experiment_id]
    
    try:
        print(f"[EXPERIMENT] Starting execution for {experiment_id}")
        print(f"[EXPERIMENT] Personas: {experiment.persona_ids}")
        print(f"[EXPERIMENT] Models: {experiment.model_ids}")
        print(f"[EXPERIMENT] Emails: {experiment.email_ids}")
        
        # Create execution engine
        engine = ExecutionEngine()
        state.engine = engine
        
        # Set up progress callback
        def on_progress(progress):
            state.experiment_progress = {
                "completed": progress.completed_trials,
                "total": progress.total_trials,
                "failed": progress.failed_trials,
                "cost": progress.cost_so_far,
                "current_persona": progress.current_persona,
                "current_model": progress.current_model
            }
            experiment.completed_trials = progress.completed_trials
        
        engine.set_progress_callback(on_progress)
        
        print(f"[EXPERIMENT] Starting engine.run_experiment...")
        results = await engine.run_experiment(
            experiment,
            state.personas,
            state.emails
        )
        
        print(f"[EXPERIMENT] Execution completed with {len(results)} results")
        state.results[experiment_id] = results
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()

        # Generate experiment log
        try:
            log_path = create_experiment_log(
                experiment=experiment,
                trials=results,
                personas=state.personas,
                emails=state.emails
            )
            state.experiment_logs[experiment_id] = log_path
            print(f"[EXPERIMENT] Log saved to: {log_path}")
        except Exception as log_error:
            print(f"[EXPERIMENT] Warning: Failed to create log: {log_error}")

    except Exception as e:
        print(f"[EXPERIMENT] ERROR: {str(e)}")
        traceback.print_exc()
        experiment.status = ExperimentStatus.FAILED
    finally:
        state.running_experiment_id = None
        state.engine = None
        print(f"[EXPERIMENT] Background task finished for {experiment_id}")


@router.get("/experiments/{experiment_id}/progress")
async def get_experiment_progress(experiment_id: str):
    """Get current progress of running experiment."""
    if experiment_id != state.running_experiment_id:
        return {"running": False}
    
    return {
        "running": True,
        **state.experiment_progress
    }


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    if state.engine:
        state.engine.stop()
        return {"status": "stopping"}
    return {"status": "not_running"}


# =============================================================================
# EXPERIMENT LOGS ENDPOINTS
# =============================================================================

@router.get("/experiments/{experiment_id}/logs")
async def get_experiment_logs(experiment_id: str):
    """Get log file path and content for an experiment."""
    if experiment_id not in state.experiment_logs:
        # Try to find log from results if available
        if experiment_id in state.results:
            experiment = state.experiments.get(experiment_id)
            if experiment:
                try:
                    log_path = create_experiment_log(
                        experiment=experiment,
                        trials=state.results[experiment_id],
                        personas=state.personas,
                        emails=state.emails
                    )
                    state.experiment_logs[experiment_id] = log_path
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to create log: {str(e)}")
            else:
                raise HTTPException(status_code=404, detail="Experiment not found")
        else:
            raise HTTPException(status_code=404, detail="No logs available for this experiment")

    log_path = state.experiment_logs[experiment_id]

    # Read log content
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log: {str(e)}")

    return {
        "experiment_id": experiment_id,
        "log_path": log_path,
        "log_content": content,
        "file_name": Path(log_path).name
    }


@router.get("/experiments/logs/list")
async def list_experiment_logs():
    """List all available experiment logs."""
    logs = []
    for exp_id, log_path in state.experiment_logs.items():
        experiment = state.experiments.get(exp_id)
        logs.append({
            "experiment_id": exp_id,
            "experiment_name": experiment.name if experiment else "Unknown",
            "log_path": log_path,
            "file_name": Path(log_path).name
        })

    return {"logs": logs}


@router.post("/experiments/{experiment_id}/logs/generate")
async def generate_experiment_log(experiment_id: str):
    """Manually generate a log for an experiment with results."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="No results found for this experiment")

    experiment = state.experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        log_path = create_experiment_log(
            experiment=experiment,
            trials=state.results[experiment_id],
            personas=state.personas,
            emails=state.emails
        )
        state.experiment_logs[experiment_id] = log_path

        return {
            "status": "success",
            "log_path": log_path,
            "file_name": Path(log_path).name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create log: {str(e)}")


# =============================================================================
# RESULTS ENDPOINTS
# =============================================================================

@router.get("/results/{experiment_id}")
async def get_results(experiment_id: str):
    """Get results for an experiment."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    
    # Summary
    n_success = sum(1 for t in trials if t.parse_success)
    n_click = sum(1 for t in trials if t.action.value == 'click')
    n_report = sum(1 for t in trials if t.action.value == 'report')
    
    return {
        "experiment_id": experiment_id,
        "total_trials": len(trials),
        "successful_parses": n_success,
        "parse_rate": n_success / len(trials) if trials else 0,
        "click_rate": n_click / n_success if n_success else 0,
        "report_rate": n_report / n_success if n_success else 0,
        "total_cost": sum(t.cost_usd for t in trials)
    }


# =============================================================================
# ANALYSIS ENDPOINTS - ENHANCED
# =============================================================================

@router.get("/analysis/{experiment_id}/fidelity")
async def analyze_fidelity(experiment_id: str):
    """Calculate fidelity metrics for an experiment."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    analyzer = FidelityAnalyzer()
    
    # Calculate fidelity per persona-model-prompt condition
    fidelity_results = []
    
    # Group by condition
    from collections import defaultdict
    grouped = defaultdict(list)
    for trial in trials:
        key = (trial.persona_id, trial.model_id, trial.prompt_config)
        grouped[key].append(trial)
    
    passing_count = 0
    total_accuracy = 0.0
    
    for (persona_id, model_id, prompt_config), condition_trials in grouped.items():
        persona = state.personas.get(persona_id)
        if not persona:
            continue
        
        metrics = analyzer.calculate_fidelity(
            condition_trials, persona, model_id, prompt_config
        )
        
        # Calculate cost and latency for this condition
        condition_cost = sum(t.cost_usd for t in condition_trials)
        condition_latencies = [t.model_latency_ms for t in condition_trials if t.model_latency_ms > 0]
        avg_latency = sum(condition_latencies) / len(condition_latencies) if condition_latencies else 0

        # Convert all values to Python native types
        result = {
            "persona_id": persona_id,
            "persona_name": persona.name,
            "model_id": model_id,
            "prompt_config": prompt_config.value if hasattr(prompt_config, 'value') else str(prompt_config),
            "normalized_accuracy": float(metrics.normalized_accuracy),
            "decision_agreement": float(metrics.decision_agreement),
            "ai_click_rate": float(metrics.ai_click_rate),
            "human_click_rate": float(metrics.human_click_rate),
            "click_rate_diff": float(metrics.click_rate_diff),
            "ai_report_rate": float(metrics.ai_report_rate),
            "human_report_rate": float(metrics.human_report_rate),
            "report_rate_diff": float(metrics.report_rate_diff),
            "n_trials": int(metrics.n_trials),
            "n_emails": int(metrics.n_emails),
            "ci_lower": float(metrics.ci_lower),
            "ci_upper": float(metrics.ci_upper),
            "meets_threshold": bool(metrics.meets_threshold),
            "threshold_used": float(metrics.threshold_used),
            # Cost and latency fields
            "cost": float(condition_cost),
            "cost_per_trial": float(condition_cost / len(condition_trials)) if condition_trials else 0.0,
            "latency_ms": float(avg_latency),
            "trial_count": len(condition_trials)
        }
        
        fidelity_results.append(result)
        total_accuracy += result["normalized_accuracy"]
        if result["meets_threshold"]:
            passing_count += 1
    
    # Calculate summary
    n_conditions = len(fidelity_results)
    mean_accuracy = total_accuracy / n_conditions if n_conditions > 0 else 0.0
    pass_rate = passing_count / n_conditions if n_conditions > 0 else 0.0
    total_cost = sum(t.cost_usd for t in trials)
    total_trials = len(trials)

    return {
        "experiment_id": experiment_id,
        "fidelity_results": fidelity_results,
        # Frontend expects thresholds.accuracy
        "thresholds": {
            "accuracy": 0.85,
            "decision_agreement": 0.80,
            "effect_preservation": 0.80
        },
        "summary": {
            "total_conditions": n_conditions,
            "passing_conditions": passing_count,
            "pass_rate": float(pass_rate),
            "mean_normalized_accuracy": float(mean_accuracy)
        },
        "total_cost": float(total_cost),
        "total_trials": total_trials
    }


@router.get("/analysis/{experiment_id}/detailed-combinations")
async def analyze_detailed_combinations(experiment_id: str):
    """
    Full 4-dimensional analysis: LLM × Persona × Email × Prompt
    Returns all combinations with fidelity and cost metrics for business comparison.
    """
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")

    trials = state.results[experiment_id]

    from collections import defaultdict

    # Group by full 4-dimensional condition: persona × model × prompt × email
    grouped_4d = defaultdict(list)
    for trial in trials:
        key = (trial.persona_id, trial.model_id, trial.prompt_config, trial.email_id)
        grouped_4d[key].append(trial)

    # Also group by 3D for aggregations
    grouped_3d_no_email = defaultdict(list)  # persona × model × prompt
    for trial in trials:
        key = (trial.persona_id, trial.model_id, trial.prompt_config)
        grouped_3d_no_email[key].append(trial)

    detailed_results = []

    for (persona_id, model_id, prompt_config, email_id), condition_trials in grouped_4d.items():
        persona = state.personas.get(persona_id)
        email = state.emails.get(email_id)

        if not persona:
            continue

        # Calculate metrics for this specific combination
        n_trials = len(condition_trials)
        n_clicked = sum(1 for t in condition_trials if t.action.value == "click")
        n_reported = sum(1 for t in condition_trials if t.action.value == "report")
        n_ignored = sum(1 for t in condition_trials if t.action.value == "ignore")

        ai_click_rate = n_clicked / n_trials if n_trials > 0 else 0
        ai_report_rate = n_reported / n_trials if n_trials > 0 else 0

        # Get human baseline from persona's behavioral statistics
        # Note: Using overall persona rates as approximation since we don't have per-email data
        human_click_rate = persona.behavioral_statistics.phishing_click_rate if persona.behavioral_statistics else 0.0
        human_report_rate = persona.behavioral_statistics.report_rate if persona.behavioral_statistics else 0.0

        # Calculate fidelity (1 - |ai_click_rate - human_click_rate|)
        click_deviation = abs(ai_click_rate - human_click_rate)
        fidelity = 1.0 - click_deviation

        # Cost and latency
        condition_cost = sum(t.cost_usd for t in condition_trials)
        condition_latencies = [t.model_latency_ms for t in condition_trials if t.model_latency_ms > 0]
        avg_latency = sum(condition_latencies) / len(condition_latencies) if condition_latencies else 0

        # Email metadata - use actual EmailStimulus attributes
        email_metadata = {}
        if email:
            email_metadata = {
                "email_subject": getattr(email, 'subject_line', '') or '',
                "email_type": getattr(email, 'email_type', 'unknown'),  # phishing/legitimate
                "email_is_phishing": getattr(email, 'email_type', '') == 'phishing' or getattr(email, 'ground_truth', 0) == 1,
                "sender_familiarity": getattr(email, 'sender_familiarity', 'unknown'),  # familiar/unfamiliar
                "urgency_level": getattr(email, 'urgency_level', 'unknown'),  # high/medium/low
                "framing_type": getattr(email, 'framing_type', 'unknown'),  # threat/reward/neutral
                "has_aggressive_content": getattr(email, 'has_aggressive_content', False),
                "aggression_level": getattr(email, 'aggression_level', 'unknown') if hasattr(email, 'aggression_level') else 'unknown',
            }

        result = {
            "persona_id": persona_id,
            "persona_name": persona.name,
            "model_id": model_id,
            "prompt_config": prompt_config.value if hasattr(prompt_config, 'value') else str(prompt_config),
            "email_id": email_id,
            **email_metadata,
            "n_trials": n_trials,
            "ai_click_rate": float(ai_click_rate),
            "ai_report_rate": float(ai_report_rate),
            "human_click_rate": float(human_click_rate),
            "human_report_rate": float(human_report_rate),
            "fidelity": float(fidelity),
            "click_deviation": float(click_deviation),
            "cost": float(condition_cost),
            "cost_per_trial": float(condition_cost / n_trials) if n_trials > 0 else 0.0,
            "latency_ms": float(avg_latency),
            "meets_threshold": fidelity >= 0.85
        }
        detailed_results.append(result)

    # Sort by fidelity descending
    detailed_results.sort(key=lambda x: x['fidelity'], reverse=True)

    # Calculate aggregations for filters
    unique_personas = list(set(r['persona_name'] for r in detailed_results))
    unique_models = list(set(r['model_id'] for r in detailed_results))
    unique_prompts = list(set(r['prompt_config'] for r in detailed_results))
    unique_emails = list(set(r['email_id'] for r in detailed_results))

    # Email attribute dimensions for filtering
    unique_email_types = list(set(r.get('email_type', 'unknown') for r in detailed_results if r.get('email_type')))
    unique_sender_familiarities = list(set(r.get('sender_familiarity', 'unknown') for r in detailed_results if r.get('sender_familiarity')))
    unique_urgency_levels = list(set(r.get('urgency_level', 'unknown') for r in detailed_results if r.get('urgency_level')))
    unique_framing_types = list(set(r.get('framing_type', 'unknown') for r in detailed_results if r.get('framing_type')))
    unique_aggression_levels = list(set(r.get('aggression_level', 'unknown') for r in detailed_results if r.get('aggression_level') and r.get('aggression_level') != 'unknown'))

    # Summary stats
    total_cost = sum(t.cost_usd for t in trials)
    avg_fidelity = sum(r['fidelity'] for r in detailed_results) / len(detailed_results) if detailed_results else 0
    passing_count = sum(1 for r in detailed_results if r['meets_threshold'])

    return {
        "experiment_id": experiment_id,
        "detailed_results": detailed_results,
        "dimensions": {
            "personas": unique_personas,
            "models": unique_models,
            "prompts": unique_prompts,
            "emails": unique_emails,
            # Email attribute dimensions for business filtering
            "email_types": sorted(unique_email_types),  # phishing, legitimate
            "sender_familiarities": sorted(unique_sender_familiarities),  # familiar, unfamiliar
            "urgency_levels": sorted(unique_urgency_levels),  # high, medium, low
            "framing_types": sorted(unique_framing_types),  # threat, reward, neutral
            "aggression_levels": sorted(unique_aggression_levels),  # very_high, high, medium, low
        },
        "summary": {
            "total_combinations": len(detailed_results),
            "passing_combinations": passing_count,
            "pass_rate": passing_count / len(detailed_results) if detailed_results else 0,
            "average_fidelity": float(avg_fidelity),
            "total_cost": float(total_cost),
            "total_trials": len(trials)
        }
    }


@router.get("/analysis/{experiment_id}/model-comparison")
async def compare_models(experiment_id: str):
    """Compare model performance for an experiment."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    analyzer = FidelityAnalyzer()
    
    # Get comparison with Python native types
    comparison = analyzer.compare_models(trials, state.personas)
    
    # Calculate Pareto frontier
    pareto_frontier = analyzer.calculate_pareto_frontier(comparison)
    
    # Mark Pareto optimal models
    for model_id in comparison:
        comparison[model_id]['is_pareto_optimal'] = model_id in pareto_frontier
    
    # Rank by fidelity
    ranked = sorted(
        comparison.items(),
        key=lambda x: x[1]['mean_fidelity'],
        reverse=True
    )
    
    # Find best models
    best_model = ranked[0][0] if ranked else None
    best_value_model = None
    for model_id in pareto_frontier:
        if comparison.get(model_id, {}).get('mean_fidelity', 0) >= 0.85:
            best_value_model = model_id
            break
    
    return {
        "experiment_id": experiment_id,
        "model_comparison": comparison,
        "ranking": [
            {"rank": i+1, "model_id": m, **data}
            for i, (m, data) in enumerate(ranked)
        ],
        "pareto_frontier": pareto_frontier,
        "best_model": best_model,
        "best_value_model": best_value_model or best_model
    }


@router.get("/analysis/{experiment_id}/boundary-conditions")
async def find_boundaries(experiment_id: str):
    """Identify boundary conditions where AI fails."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    analyzer = FidelityAnalyzer()
    
    boundaries = analyzer.find_boundary_conditions(
        trials, state.personas, state.emails
    )
    
    # Group by severity
    by_severity = {"high": [], "medium": [], "low": []}
    for bc in boundaries:
        severity = bc.get("severity", "low")
        by_severity[severity].append(bc)
    
    # Group by type
    by_type = {}
    for bc in boundaries:
        bc_type = bc.get("type", "unknown")
        if bc_type not in by_type:
            by_type[bc_type] = []
        by_type[bc_type].append(bc)
    
    return {
        "experiment_id": experiment_id,
        "boundary_conditions": boundaries,
        "by_severity": by_severity,
        "by_type": by_type,
        "summary": {
            "total_found": len(boundaries),
            "high_severity": len(by_severity["high"]),
            "medium_severity": len(by_severity["medium"]),
            "low_severity": len(by_severity["low"])
        }
    }


@router.get("/analysis/{experiment_id}/recommendations")
async def get_recommendations(experiment_id: str):
    """Get deployment recommendations based on analysis."""
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    analyzer = FidelityAnalyzer()
    
    comparison = analyzer.compare_models(trials, state.personas)
    boundaries = analyzer.find_boundary_conditions(trials, state.personas, state.emails)
    pareto_frontier = analyzer.calculate_pareto_frontier(comparison)
    
    # Find best overall model
    best_fidelity_model = None
    best_fidelity = 0.0
    for model_id, metrics in comparison.items():
        if metrics['mean_fidelity'] > best_fidelity:
            best_fidelity = metrics['mean_fidelity']
            best_fidelity_model = model_id
    
    # Find best value model
    valid_pareto = [
        m for m in pareto_frontier 
        if comparison.get(m, {}).get('mean_fidelity', 0) >= 0.85
    ]
    best_value_model = valid_pareto[0] if valid_pareto else best_fidelity_model
    
    # Find problematic personas
    problematic_personas = list(set(bc['persona_id'] for bc in boundaries))
    high_severity_count = sum(1 for bc in boundaries if bc['severity'] == 'high')
    
    # Generate recommendation text
    if best_fidelity >= 0.90:
        quality = "excellent"
    elif best_fidelity >= 0.85:
        quality = "good"
    else:
        quality = "acceptable"
    
    recommendation_text = f"For highest accuracy, use {best_fidelity_model} ({best_fidelity:.1%} fidelity). "
    if best_value_model and best_value_model != best_fidelity_model:
        value_cost = comparison.get(best_value_model, {}).get('cost_per_decision', 0)
        recommendation_text += f"For cost-efficiency, {best_value_model} offers {quality} fidelity at ${value_cost:.4f}/decision. "
    if high_severity_count > 0:
        recommendation_text += f"Note: {high_severity_count} high-severity boundary conditions detected."
    
    return {
        "experiment_id": experiment_id,
        "recommendations": {
            "best_fidelity_model": best_fidelity_model,
            "best_fidelity_score": float(best_fidelity),
            "best_value_model": best_value_model,
            "best_value_score": float(comparison.get(best_value_model, {}).get('mean_fidelity', 0)),
            "best_value_cost": float(comparison.get(best_value_model, {}).get('cost_per_decision', 0)),
            "pareto_optimal_models": pareto_frontier,
            "problematic_personas": problematic_personas,
            "high_severity_issues": high_severity_count,
            "overall_recommendation": recommendation_text
        }
    }
    

@router.get("/analysis/{experiment_id}/effect-preservation")
async def analyze_effect_preservation(experiment_id: str):
    """
    Detailed analysis of how well AI preserves email manipulation effects.
    
    Returns:
        - Per-persona effect preservation scores
        - Urgency effect comparison (AI vs Human)
        - Familiarity effect comparison
        - Framing effect comparison (if available)
    """
    if experiment_id not in state.results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    trials = state.results[experiment_id]
    analyzer = FidelityAnalyzer()
    
    # Calculate effect preservation per persona
    effect_results = []
    
    from collections import defaultdict
    persona_trials = defaultdict(list)
    for trial in trials:
        if trial.parse_success:
            persona_trials[trial.persona_id].append(trial)
    
    for persona_id, p_trials in persona_trials.items():
        persona = state.personas.get(persona_id)
        if not persona:
            continue
        
        effects = analyzer._calculate_effect_preservation(
            p_trials, persona, state.emails
        )
        
        effect_results.append({
            "persona_id": persona_id,
            "persona_name": persona.name,
            "n_trials": len(p_trials),
            
            # Urgency
            "urgency_effect_human": persona.email_interaction_effects.urgency_effect,
            "urgency_effect_ai": effects.get('urgency_ai'),
            "urgency_preservation": effects.get('urgency_r'),
            
            # Familiarity
            "familiarity_effect_human": persona.email_interaction_effects.familiarity_effect,
            "familiarity_effect_ai": effects.get('familiarity_ai'),
            "familiarity_preservation": effects.get('familiarity_r'),
            
            # Overall
            "overall_effect_preservation": effects.get('overall_r'),
            "meets_threshold": (effects.get('overall_r') or 0) >= 0.80
        })
    
    return {
        "experiment_id": experiment_id,
        "effect_preservation_results": effect_results,
        "threshold": 0.80
    }


# =============================================================================
# CALIBRATION ENDPOINTS - Prompt Validation via Held-Out Data
# =============================================================================

from phase2.calibration import (
    DataSplitter, CalibrationEngine, SelfReflectionEngine,
    generate_synthetic_trials
)

# Calibration state
class CalibrationState:
    """State for calibration runs."""
    split_results: Dict[str, Any] = {}  # persona_id -> SplitResult
    calibration_results: Dict[str, Any] = {}  # key -> CalibrationResult
    reflection_results: Dict[str, Any] = {}  # key -> ReflectionResult

    # Stop calibration support
    stop_requested: bool = False
    running_calibration_key: Optional[str] = None
    calibration_progress: Dict[str, Any] = {}

calibration_state = CalibrationState()


class CalibrationRequest(BaseModel):
    """Request for calibration run."""
    persona_id: str
    model_id: str
    prompt_config: str  # baseline, stats, cot
    split_ratio: float = 0.8
    use_synthetic: bool = True  # If no real trials, generate synthetic
    use_icl: bool = True  # Whether to use In-Context Learning examples
    test_sample_size: Optional[int] = None  # NEW: Limit test trials for quick testing (None = use all)


class ReflectionRequest(BaseModel):
    """Request for self-reflection."""
    calibration_key: str
    reflection_model: str = "claude-sonnet-4"


class ApplySuggestionsRequest(BaseModel):
    """Request to apply suggestions and rerun calibration."""
    calibration_key: str
    suggestion_indices: List[int] = []  # Which suggestions to apply (empty = all)
    auto_rerun: bool = True  # Automatically rerun calibration after applying
    max_iterations: int = 3  # Maximum auto-rerun iterations
    test_sample_size: Optional[int] = None  # Quick test sample size (None = use all)


@router.post("/calibration/split")
async def split_persona_data(persona_id: str, split_ratio: float = 0.8):
    """
    Split persona behavioral data into train/test sets.

    Uses REAL trial data from Phase 1 if available, otherwise falls back
    to generating synthetic trials from behavioral statistics.
    """
    if persona_id not in state.personas:
        raise HTTPException(status_code=404, detail="Persona not found")

    persona = state.personas[persona_id]

    emails = list(state.emails.values())
    if not emails:
        raise HTTPException(status_code=400, detail="No emails loaded. Import emails first.")

    # Try to get REAL trial data from Phase 1
    real_trials = None
    try:
        from api.phase1 import state as phase1_state

        if (phase1_state.responses is not None and
            phase1_state.participants is not None and
            phase1_state.last_run_result is not None):

            labels = phase1_state.last_run_result.get('_labels')
            if labels is not None:
                # Extract cluster_id from persona_id (e.g., "PERSONA_8" -> 8)
                cluster_id = int(persona_id.split('_')[-1])

                # Add cluster labels to participants
                participants_df = phase1_state.participants.copy()
                if len(labels) == len(participants_df):
                    participants_df['cluster'] = labels

                    # Get participant IDs for this cluster
                    cluster_participants = participants_df[
                        participants_df['cluster'] == cluster_id
                    ]['participant_id'].tolist()

                    if cluster_participants:
                        # Filter responses for these participants
                        cluster_responses = phase1_state.responses[
                            phase1_state.responses['participant_id'].isin(cluster_participants)
                        ]

                        if len(cluster_responses) > 0:
                            print(f"\n✅ Using REAL trial data from Phase 1!")
                            print(f"   • Cluster {cluster_id}: {len(cluster_participants)} participants")
                            print(f"   • {len(cluster_responses)} real behavioral trials found")

                            # Convert responses to trial format
                            real_trials = []
                            for _, row in cluster_responses.iterrows():
                                # Map action
                                if row.get('clicked', 0) == 1:
                                    action = 'click'
                                elif row.get('reported', 0) == 1:
                                    action = 'report'
                                else:
                                    action = 'ignore'

                                real_trials.append({
                                    'trial_id': f"{row['participant_id']}_{row['email_id']}",
                                    'email_id': row.get('email_id', ''),
                                    'email_type': row.get('email_type', 'unknown'),
                                    'urgency_level': row.get('urgency_level', 'low'),
                                    'sender_familiarity': row.get('sender_familiarity', 'unfamiliar'),
                                    'framing_type': row.get('framing_type', 'neutral'),
                                    'action': action,
                                    'response_time_ms': int(row.get('response_latency_ms', 5000)),
                                    # Qualitative fields for Chain-of-Thought prompting
                                    'details_noticed': row.get('details_noticed'),
                                    'steps_taken': row.get('steps_taken'),
                                    'decision_reason': row.get('decision_reason'),
                                    'confidence_reason': row.get('confidence_reason'),
                                    'unsure_about': row.get('unsure_about'),
                                })
    except Exception as e:
        print(f"⚠️ Could not load real trial data: {e}")
        real_trials = None

    # Use real trials if available, otherwise generate synthetic
    if real_trials and len(real_trials) >= 10:
        trials_to_use = real_trials
        print(f"   • Using {len(trials_to_use)} REAL trials for calibration")
    else:
        # Fall back to synthetic trials
        from phase2.calibration.data_splitter import generate_synthetic_trials
        print(f"\n⚠️ No real trial data available, generating SYNTHETIC trials...")

        trials_to_use = generate_synthetic_trials(
            persona.model_dump(),
            [e.model_dump() for e in emails],
            n_trials_per_email=5  # Increased from 3 for better statistics
        )
        print(f"   • Generated {len(trials_to_use)} synthetic trials")

    # Split the data
    splitter = DataSplitter(split_ratio=split_ratio)
    split_result = splitter.split(
        persona.model_dump(),
        trials_to_use,
        [e.model_dump() for e in emails]
    )

    # Store result
    calibration_state.split_results[persona_id] = split_result

    # Track whether we used real or synthetic data
    used_real_data = real_trials is not None and len(real_trials) >= 10

    return {
        "persona_id": persona_id,
        "persona_name": split_result.persona_name,
        "split_ratio": split_ratio,
        "train_count": split_result.n_train,
        "test_count": split_result.n_test,
        "train_statistics": split_result.train_statistics,
        "test_statistics": split_result.test_statistics,
        "random_seed": split_result.random_seed,
        "data_source": "real_phase1_trials" if used_real_data else "synthetic",
        "total_trials": len(trials_to_use)
    }


@router.post("/calibration/run")
async def run_calibration(request: CalibrationRequest):
    """
    Run calibration trials on held-out test data.

    Tests if the prompt configuration can make the LLM predict
    human responses accurately.
    
    NEW: Supports test_sample_size parameter for quick testing with fewer trials.
    """
    persona_id = request.persona_id
    model_id = request.model_id
    prompt_config = request.prompt_config
    test_sample_size = request.test_sample_size  # NEW: Optional limit on test trials

    print("\n" + "="*70)
    print("🧪 CALIBRATION STARTED")
    print("="*70)
    print(f"📋 Configuration:")
    print(f"   • Persona: {persona_id}")
    print(f"   • Model: {model_id}")
    print(f"   • Prompt Config: {prompt_config}")
    print(f"   • Split Ratio: {request.split_ratio}")
    if test_sample_size:
        print(f"   • Test Sample Size: {test_sample_size} (QUICK TEST MODE)")

    if persona_id not in state.personas:
        raise HTTPException(status_code=404, detail="Persona not found")

    # Reset stop flag at start of new calibration
    calibration_state.stop_requested = False
    cal_key = f"{persona_id}_{model_id}_{prompt_config}"
    if test_sample_size:
        cal_key += f"_sample{test_sample_size}"  # Distinguish sampled runs in cache
    calibration_state.running_calibration_key = cal_key
    calibration_state.calibration_progress = {"completed": 0, "total": 0, "status": "starting"}

    # FIX Issue #2: Check if split ratio changed - force re-split if different
    existing_split = calibration_state.split_results.get(persona_id)
    if existing_split is None or abs(existing_split.split_ratio - request.split_ratio) > 0.01:
        # Need to (re-)split with the new ratio
        print(f"\n📊 Splitting data ({request.split_ratio*100:.0f}% train / {(1-request.split_ratio)*100:.0f}% test)...")
        await split_persona_data(persona_id, request.split_ratio)

    split_result = calibration_state.split_results[persona_id]
    persona = state.personas[persona_id]

    # NEW: Apply test sample size limit if specified
    actual_test_count = split_result.n_test
    if test_sample_size and test_sample_size < split_result.n_test:
        # Create a sampled split result with fewer test trials
        import random
        sampled_test_trials = random.sample(split_result.test_trials, test_sample_size)
        # Create a modified split result (without modifying the original)
        # Note: n_train and n_test are properties computed from train_trials/test_trials length
        from phase2.calibration.data_splitter import SplitResult
        split_result = SplitResult(
            persona_id=split_result.persona_id,
            persona_name=split_result.persona_name,
            split_ratio=split_result.split_ratio,
            random_seed=split_result.random_seed,
            train_trials=split_result.train_trials,
            test_trials=sampled_test_trials,
            train_statistics=split_result.train_statistics,
            test_statistics=split_result.test_statistics
        )
        actual_test_count = test_sample_size
        print(f"\n⚡ QUICK TEST MODE: Using {test_sample_size} of {calibration_state.split_results[persona_id].n_test} test trials")

    print(f"\n📈 Data Split Results:")
    print(f"   • Training trials: {split_result.n_train}")
    print(f"   • Test trials: {actual_test_count}" + (f" (sampled from {calibration_state.split_results[persona_id].n_test})" if test_sample_size else ""))
    print(f"   • Training click rate: {split_result.train_statistics.get('phishing_click_rate', 0)*100:.1f}%")

    # Create calibration engine
    router_instance = await get_router()
    prompt_builder = PromptBuilder()
    engine = CalibrationEngine(router_instance, prompt_builder)

    # Prepare emails list for ICL
    emails_list = [e.model_dump() for e in state.emails.values()] if state.emails else None

    print(f"\n🤖 Running {actual_test_count} LLM trials on test set...")
    print(f"   • ICL: {'Enabled' if request.use_icl else 'Disabled'}")
    print("-"*70)

    # Define stop check and progress callback
    def check_stop():
        return calibration_state.stop_requested

    def update_progress(completed, total):
        calibration_state.calibration_progress = {
            "completed": completed,
            "total": total,
            "status": "running"
        }

    # Run calibration with ICL support and stop/progress callbacks
    config_enum = PromptConfiguration(prompt_config)
    cal_result = await engine.run_calibration(
        persona, split_result, config_enum, model_id,
        use_icl=request.use_icl, emails=emails_list,
        stop_check=check_stop, progress_callback=update_progress
    )

    print("-"*70)
    print(f"\n✅ CALIBRATION COMPLETE")
    print(f"   • Accuracy: {cal_result.accuracy*100:.1f}% ({cal_result.n_correct}/{cal_result.n_trials} correct)")
    print(f"   • Human click rate: {cal_result.human_click_rate*100:.1f}%")
    print(f"   • LLM click rate: {cal_result.llm_click_rate*100:.1f}%")
    print(f"   • Click rate error: {cal_result.click_rate_error*100:.1f}%")
    print(f"   • Meets 80% threshold: {'✅ YES' if cal_result.meets_threshold(0.80) else '❌ NO'}")

    # Generate detailed log file
    log_file_path = None
    try:
        log_file_path = create_calibration_log(
            persona=persona,
            model_id=model_id,
            prompt_config=prompt_config,
            split_result=split_result,
            calibration_result=cal_result,
            prompts_used=cal_result.prompts_used,
            emails_used=cal_result.emails_used,
            use_icl=request.use_icl
        )
        print(f"   • Detailed log saved: {log_file_path}")
    except Exception as e:
        print(f"   ⚠️ Could not generate log file: {e}")

    print("="*70 + "\n")

    # Store result
    calibration_state.calibration_results[cal_key] = cal_result

    # Clear running state
    calibration_state.running_calibration_key = None
    calibration_state.calibration_progress = {"completed": cal_result.n_trials, "total": cal_result.n_trials, "status": "completed"}

    # Check if stopped early
    was_stopped = calibration_state.stop_requested
    calibration_state.stop_requested = False

    # Calculate full test set size for reference
    full_test_count = calibration_state.split_results[persona_id].n_test if persona_id in calibration_state.split_results else cal_result.n_trials

    return {
        "calibration_key": cal_key,
        "persona_id": persona_id,
        "persona_name": cal_result.persona_name,
        "model_id": model_id,
        "prompt_config": prompt_config,
        "accuracy": float(cal_result.accuracy),
        "n_trials": cal_result.n_trials,
        "n_correct": cal_result.n_correct,
        "human_click_rate": float(cal_result.human_click_rate),
        "llm_click_rate": float(cal_result.llm_click_rate),
        "click_rate_error": float(cal_result.click_rate_error),
        "meets_threshold": cal_result.meets_threshold(0.80),
        "failure_summary": cal_result.get_failure_summary(),
        "status": "stopped" if was_stopped else cal_result.status.value,
        "use_icl": request.use_icl,
        "log_file": log_file_path,
        "stopped_early": was_stopped,
        # NEW: Sample size info
        "test_sample_size": request.test_sample_size,
        "full_test_count": full_test_count,
        "is_quick_test": request.test_sample_size is not None and request.test_sample_size < full_test_count
    }


@router.post("/calibration/stop")
async def stop_calibration():
    """
    Stop a running calibration.

    Sets the stop flag which will be checked during the calibration loop.
    The calibration will stop after completing the current trial and
    return partial results.
    """
    if not calibration_state.running_calibration_key:
        return {"status": "not_running", "message": "No calibration is currently running"}

    calibration_state.stop_requested = True
    print(f"\n⛔ STOP REQUESTED for calibration: {calibration_state.running_calibration_key}")

    return {
        "status": "stopping",
        "calibration_key": calibration_state.running_calibration_key,
        "message": "Stop requested. Calibration will stop after current trial."
    }


@router.get("/calibration/progress")
async def get_calibration_progress():
    """
    Get the progress of a running calibration.
    """
    if not calibration_state.running_calibration_key:
        return {
            "running": False,
            "calibration_key": None,
            "progress": calibration_state.calibration_progress
        }

    return {
        "running": True,
        "calibration_key": calibration_state.running_calibration_key,
        "progress": calibration_state.calibration_progress,
        "stop_requested": calibration_state.stop_requested
    }


@router.post("/calibration/reflect")
async def run_self_reflection(request: ReflectionRequest):
    """
    Run LLM self-reflection on calibration failures.

    The LLM analyzes its failures and suggests prompt improvements.
    """
    cal_key = request.calibration_key

    if cal_key not in calibration_state.calibration_results:
        raise HTTPException(status_code=404, detail="Calibration result not found. Run calibration first.")

    cal_result = calibration_state.calibration_results[cal_key]

    if cal_result.meets_threshold(0.80):
        return {
            "message": "Calibration already meets 80% threshold. No reflection needed.",
            "accuracy": float(cal_result.accuracy)
        }

    # Get persona description
    persona = state.personas.get(cal_result.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    # Get original prompt (from prompt builder)
    prompt_builder = PromptBuilder()
    sample_email = list(state.emails.values())[0] if state.emails else None
    original_prompt = ""
    if sample_email:
        built = prompt_builder.build(
            persona, sample_email,
            PromptConfiguration(cal_result.prompt_config)
        )
        original_prompt = built.system_prompt

    # Run self-reflection
    router_instance = await get_router()
    reflection_engine = SelfReflectionEngine(router_instance)

    reflection_result = await reflection_engine.reflect(
        cal_result,
        original_prompt,
        persona.description,
        request.reflection_model
    )

    # Store result
    calibration_state.reflection_results[cal_key] = reflection_result

    return {
        "calibration_key": cal_key,
        "accuracy_before": float(cal_result.accuracy),
        "failure_patterns": reflection_result.failure_patterns,
        "root_cause_analysis": reflection_result.root_cause_analysis,
        "suggestions": [
            {
                "category": s.category,
                "issue": s.issue_identified,
                "change": s.suggested_change,
                "confidence": s.confidence,
                "index": idx
            }
            for idx, s in enumerate(reflection_result.suggestions)
        ],
        "improved_prompt": reflection_result.improved_prompt,
        "reflection_model": reflection_result.reflection_model
    }


@router.post("/calibration/apply-suggestions")
async def apply_suggestions_and_rerun(request: ApplySuggestionsRequest):
    """
    Apply suggested prompt improvements and optionally rerun calibration.
    
    This is the AUTO-CALIBRATION feature: Accept suggestions and rerun until
    accuracy threshold is met or max iterations reached.
    
    Flow:
    1. Get the reflection suggestions for the calibration
    2. Apply selected suggestions to create improved prompt config
    3. If auto_rerun=True, run calibration with improved prompt
    4. If still below threshold and iterations remain, get new suggestions and repeat
    
    Returns the calibration result and iteration history.
    """
    cal_key = request.calibration_key
    
    if cal_key not in calibration_state.calibration_results:
        raise HTTPException(status_code=404, detail="Calibration result not found. Run calibration first.")
    
    if cal_key not in calibration_state.reflection_results:
        raise HTTPException(status_code=404, detail="No reflection results found. Run reflection analysis first.")
    
    cal_result = calibration_state.calibration_results[cal_key]
    reflection = calibration_state.reflection_results[cal_key]
    
    # Get persona
    persona = state.personas.get(cal_result.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    # Prepare iteration tracking
    iteration_history = []
    current_accuracy = cal_result.accuracy
    current_iteration = 0
    
    # Get suggestions to apply (all or selected indices)
    suggestions_to_apply = reflection.suggestions
    if request.suggestion_indices:
        suggestions_to_apply = [
            reflection.suggestions[i] 
            for i in request.suggestion_indices 
            if i < len(reflection.suggestions)
        ]
    
    if not suggestions_to_apply:
        return {
            "success": False,
            "message": "No valid suggestions to apply",
            "calibration_key": cal_key
        }
    
    # Store improved prompt if available
    improved_prompt_text = reflection.improved_prompt
    applied_suggestions = [
        {
            "category": s.category,
            "issue": s.issue_identified,
            "change": s.suggested_change,
            "confidence": s.confidence
        }
        for s in suggestions_to_apply
    ]
    
    iteration_history.append({
        "iteration": 0,
        "accuracy": float(current_accuracy),
        "suggestions_applied": len(applied_suggestions),
        "status": "initial"
    })
    
    # Auto-rerun loop if enabled
    final_result = None
    if request.auto_rerun:
        # Store the improved prompt in calibration state for use
        calibration_state.improved_prompts = getattr(calibration_state, 'improved_prompts', {})
        calibration_state.improved_prompts[cal_result.persona_id] = {
            "prompt_text": improved_prompt_text,
            "suggestions": applied_suggestions,
            "original_config": cal_result.prompt_config
        }
        
        while current_iteration < request.max_iterations:
            current_iteration += 1
            print(f"\n🔄 AUTO-CALIBRATION: Iteration {current_iteration}/{request.max_iterations}")
            
            # Rerun calibration with same config (prompt builder will use improved prompt)
            rerun_request = CalibrationRequest(
                persona_id=cal_result.persona_id,
                model_id=cal_result.model_id,
                prompt_config=cal_result.prompt_config,
                split_ratio=calibration_state.split_results.get(cal_result.persona_id).split_ratio if cal_result.persona_id in calibration_state.split_results else 0.8,
                use_icl=True,  # Keep ICL enabled
                test_sample_size=request.test_sample_size  # Pass quick test sample size
            )
            
            try:
                rerun_result = await run_calibration(rerun_request)
                new_accuracy = rerun_result["accuracy"]
                
                iteration_history.append({
                    "iteration": current_iteration,
                    "accuracy": new_accuracy,
                    "improvement": new_accuracy - current_accuracy,
                    "meets_threshold": rerun_result["meets_threshold"],
                    "status": "completed"
                })
                
                current_accuracy = new_accuracy
                final_result = rerun_result
                
                # Check if we've met the threshold
                if rerun_result["meets_threshold"]:
                    print(f"✅ AUTO-CALIBRATION SUCCESS: Achieved {new_accuracy*100:.1f}% accuracy")
                    break
                    
                # If not met and more iterations available, get new suggestions
                if current_iteration < request.max_iterations:
                    # Run new reflection
                    new_reflection_request = ReflectionRequest(
                        calibration_key=rerun_result["calibration_key"],
                        reflection_model="claude-sonnet-4"
                    )
                    try:
                        new_reflection = await run_self_reflection(new_reflection_request)
                        if new_reflection.get("improved_prompt"):
                            calibration_state.improved_prompts[cal_result.persona_id]["prompt_text"] = new_reflection["improved_prompt"]
                            print(f"   📝 Got new suggestions for next iteration")
                    except Exception as e:
                        print(f"   ⚠️ Could not get new suggestions: {e}")
                        break
                        
            except Exception as e:
                iteration_history.append({
                    "iteration": current_iteration,
                    "error": str(e),
                    "status": "failed"
                })
                print(f"   ❌ Iteration {current_iteration} failed: {e}")
                break
    
    # Clear improved prompt after auto-calibration
    if hasattr(calibration_state, 'improved_prompts') and cal_result.persona_id in calibration_state.improved_prompts:
        del calibration_state.improved_prompts[cal_result.persona_id]
    
    return {
        "success": True,
        "calibration_key": cal_key,
        "initial_accuracy": float(cal_result.accuracy),
        "final_accuracy": current_accuracy,
        "improvement": current_accuracy - cal_result.accuracy,
        "meets_threshold": current_accuracy >= 0.80,
        "iterations_run": current_iteration,
        "max_iterations": request.max_iterations,
        "applied_suggestions": applied_suggestions,
        "improved_prompt": improved_prompt_text,
        "iteration_history": iteration_history,
        "final_result": final_result
    }


@router.get("/calibration/results")
async def list_calibration_results():
    """List all calibration results."""
    results = []
    for key, cal_result in calibration_state.calibration_results.items():
        results.append({
            "calibration_key": key,
            "persona_id": cal_result.persona_id,
            "model_id": cal_result.model_id,
            "prompt_config": cal_result.prompt_config,
            "accuracy": float(cal_result.accuracy),
            "meets_threshold": cal_result.meets_threshold(0.80),
            "status": cal_result.status.value
        })

    return {"calibration_results": results}


@router.get("/calibration/{calibration_key}")
async def get_calibration_result(calibration_key: str):
    """Get detailed calibration result."""
    if calibration_key not in calibration_state.calibration_results:
        raise HTTPException(status_code=404, detail="Calibration result not found")

    cal_result = calibration_state.calibration_results[calibration_key]

    # Get reflection if available
    reflection = calibration_state.reflection_results.get(calibration_key)

    return {
        "calibration_key": calibration_key,
        "persona_id": cal_result.persona_id,
        "persona_name": cal_result.persona_name,
        "model_id": cal_result.model_id,
        "prompt_config": cal_result.prompt_config,

        # Metrics
        "accuracy": float(cal_result.accuracy),
        "n_trials": cal_result.n_trials,
        "n_correct": cal_result.n_correct,

        # Click rates
        "human_click_rate": float(cal_result.human_click_rate),
        "llm_click_rate": float(cal_result.llm_click_rate),
        "click_rate_error": float(cal_result.click_rate_error),
        "click_precision": float(cal_result.click_precision),
        "click_recall": float(cal_result.click_recall),

        # Report rates
        "human_report_rate": float(cal_result.human_report_rate),
        "llm_report_rate": float(cal_result.llm_report_rate),
        "report_rate_error": float(cal_result.report_rate_error),

        # Factor-specific
        "phishing_accuracy": float(cal_result.phishing_accuracy),
        "legitimate_accuracy": float(cal_result.legitimate_accuracy),
        "high_urgency_accuracy": float(cal_result.high_urgency_accuracy),
        "low_urgency_accuracy": float(cal_result.low_urgency_accuracy),

        # Status
        "meets_threshold": cal_result.meets_threshold(0.80),
        "status": cal_result.status.value,
        "failure_summary": cal_result.get_failure_summary(),

        # Reflection (if available)
        "has_reflection": reflection is not None,
        "reflection": {
            "root_cause": reflection.root_cause_analysis if reflection else None,
            "n_suggestions": len(reflection.suggestions) if reflection else 0,
            "has_improved_prompt": reflection.improved_prompt is not None if reflection else False
        } if reflection else None
    }


@router.post("/calibration/compare-configs")
async def compare_prompt_configs(
    persona_id: str, 
    model_id: str, 
    use_icl: bool = True, 
    force_rerun: bool = False,
    test_sample_size: Optional[int] = None
):
    """
    Compare all three prompt configurations for a persona.

    Runs calibration for baseline, stats, and cot and compares results.
    All configs use the same ICL setting for fair comparison.
    
    If test_sample_size is provided, all configs use the SAME sampled test data
    for fair comparison (quick test mode).

    Uses cached results when available to avoid duplicate API costs.
    Set force_rerun=True to ignore cache and run fresh calibrations.
    """
    results = {}
    cached_count = 0
    fresh_count = 0
    
    # When using quick test mode, we need to ensure all configs use the same test sample
    # So we need to force fresh runs with the same sample size
    is_quick_test = test_sample_size is not None
    if is_quick_test:
        print(f"\n⚡ Quick Test Mode: All configs will use {test_sample_size} test trials")
        # Force rerun to ensure all configs use the same sample size
        force_rerun = True

    for config in ["baseline", "stats", "cot"]:
        cal_key = f"{persona_id}_{model_id}_{config}"

        # Check if we already have this calibration result cached (unless force_rerun)
        if not force_rerun and cal_key in calibration_state.calibration_results:
            # Use cached result
            existing = calibration_state.calibration_results[cal_key]
            results[config] = {
                "calibration_key": cal_key,
                "persona_id": persona_id,
                "persona_name": existing.persona_name,
                "model_id": model_id,
                "prompt_config": config,
                "accuracy": float(existing.accuracy),
                "n_trials": existing.n_trials,
                "n_correct": existing.n_correct,
                "human_click_rate": float(existing.human_click_rate),
                "llm_click_rate": float(existing.llm_click_rate),
                "click_rate_error": float(existing.click_rate_error),
                "meets_threshold": existing.meets_threshold(0.80),
                "failure_summary": existing.get_failure_summary(),
                "status": existing.status.value,
                "use_icl": use_icl,
                "from_cache": True
            }
            cached_count += 1
            print(f"   ✅ Using cached result for '{config}' (accuracy: {existing.accuracy:.1%})")
        else:
            # Run fresh calibration
            request = CalibrationRequest(
                persona_id=persona_id,
                model_id=model_id,
                prompt_config=config,
                use_icl=use_icl,
                test_sample_size=test_sample_size  # Pass quick test sample size
            )
            result = await run_calibration(request)
            result["from_cache"] = False
            result["is_quick_test"] = is_quick_test
            results[config] = result
            fresh_count += 1

    # Find best config
    best_config = max(results.keys(), key=lambda k: results[k]["accuracy"])

    print(f"\n📊 Compare-configs summary: {cached_count} cached, {fresh_count} fresh runs")

    return {
        "persona_id": persona_id,
        "model_id": model_id,
        "use_icl": use_icl,
        "results": results,
        "best_config": best_config,
        "best_accuracy": results[best_config]["accuracy"],
        "recommendation": f"Use '{best_config}' prompt configuration for best accuracy ({results[best_config]['accuracy']:.1%})",
        "cache_stats": {
            "cached_results": cached_count,
            "fresh_runs": fresh_count,
            "api_calls_saved": cached_count * (calibration_state.split_results.get(persona_id).n_test if persona_id in calibration_state.split_results else 0)
        }
    }


# =============================================================================
# CALIBRATION LOG ENDPOINTS
# =============================================================================

@router.get("/calibration/logs")
async def list_calibration_logs():
    """
    List all available calibration log files.

    Returns a list of log files with their metadata.
    """
    from pathlib import Path

    log_dir = Path(__file__).parent.parent / "data" / "calibration_logs"

    if not log_dir.exists():
        return {"logs": [], "log_directory": str(log_dir)}

    logs = []
    for log_file in sorted(log_dir.glob("calibration_*.txt"), reverse=True):
        # Parse filename for metadata: calibration_PERSONA_MODEL_CONFIG_TIMESTAMP.txt
        parts = log_file.stem.split("_")
        logs.append({
            "filename": log_file.name,
            "path": str(log_file),
            "size_kb": round(log_file.stat().st_size / 1024, 2),
            "created": log_file.stat().st_mtime,
            "persona_id": parts[1] if len(parts) > 1 else "unknown",
        })

    return {
        "logs": logs,
        "log_directory": str(log_dir),
        "total_count": len(logs)
    }


@router.get("/calibration/logs/{filename}")
async def get_calibration_log(filename: str):
    """
    Get the content of a specific calibration log file.

    Args:
        filename: The log filename (e.g., calibration_PERSONA_8_llama4_stats_20250119_123456.txt)

    Returns:
        The full content of the log file.
    """
    from pathlib import Path

    log_dir = Path(__file__).parent.parent / "data" / "calibration_logs"
    log_path = log_dir / filename

    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")

    # Security check: ensure path is within log_dir
    if not str(log_path.resolve()).startswith(str(log_dir.resolve())):
        raise HTTPException(status_code=403, detail="Invalid file path")

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "filename": filename,
            "path": str(log_path),
            "content": content,
            "size_kb": round(log_path.stat().st_size / 1024, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")


@router.delete("/calibration/logs/{filename}")
async def delete_calibration_log(filename: str):
    """Delete a calibration log file."""
    from pathlib import Path

    log_dir = Path(__file__).parent.parent / "data" / "calibration_logs"
    log_path = log_dir / filename

    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")

    # Security check
    if not str(log_path.resolve()).startswith(str(log_dir.resolve())):
        raise HTTPException(status_code=403, detail="Invalid file path")

    try:
        log_path.unlink()
        return {"message": f"Deleted log file: {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting log file: {e}")


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/cost-estimate")
async def estimate_cost(
    model_ids: str,  # comma-separated
    n_trials: int = 1000,
    avg_input_tokens: int = 800,
    avg_output_tokens: int = 200
):
    """Estimate cost for a simulation run."""
    models = model_ids.split(',')
    router_instance = await get_router()
    
    estimate = router_instance.estimate_experiment_cost(
        models, n_trials, avg_input_tokens, avg_output_tokens
    )
    
    return estimate


@router.get("/usage")
async def get_usage():
    """Get usage statistics."""
    router_instance = await get_router()
    return router_instance.get_usage_stats()


@router.post("/test-prompt")
async def test_prompt(
    persona_id: str,
    email_id: str,
    prompt_config: str = "cot"
):
    """Preview the prompt that would be generated."""
    if persona_id not in state.personas:
        raise HTTPException(status_code=404, detail="Persona not found")
    if email_id not in state.emails:
        raise HTTPException(status_code=404, detail="Email not found")
    
    builder = PromptBuilder()
    built = builder.build(
        state.personas[persona_id],
        state.emails[email_id],
        PromptConfiguration(prompt_config)
    )
    
    return {
        "system_prompt": built.system_prompt,
        "user_prompt": built.user_prompt,
        "config": prompt_config
    }

@router.get("/debug/state")
async def debug_state():
    """Debug endpoint to check stored state."""
    return {
        "personas": {
            "count": len(state.personas),
            "ids": list(state.personas.keys())
        },
        "emails": {
            "count": len(state.emails),
            "ids": list(state.emails.keys())
        },
        "experiments": {
            "count": len(state.experiments),
            "ids": list(state.experiments.keys())
        }
    }