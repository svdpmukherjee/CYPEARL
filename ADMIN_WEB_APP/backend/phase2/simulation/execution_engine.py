"""
CYPEARL Phase 2 - Simulation Execution Engine
Runs persona simulations with checkpointing, progress tracking, and error handling.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
import random

from core.schemas import (
    Persona, EmailStimulus, ExperimentConfig, SimulationTrial,
    PromptConfiguration, ExperimentStatus, ActionType
)
from core.config import config
from phase2.providers import get_router, LLMRequest
from .prompt_builder import PromptBuilder, ResponseParser


@dataclass
class ExperimentProgress:
    """Tracks experiment progress."""
    experiment_id: str
    total_trials: int
    completed_trials: int = 0
    failed_trials: int = 0
    current_persona: str = ""
    current_model: str = ""
    current_email: str = ""
    estimated_remaining_seconds: int = 0
    cost_so_far: float = 0.0
    started_at: Optional[datetime] = None
    
    @property
    def progress_pct(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return (self.completed_trials / self.total_trials) * 100


@dataclass
class Checkpoint:
    """Checkpoint for experiment resumption."""
    experiment_id: str
    completed_trial_ids: List[str] = field(default_factory=list)
    last_saved: datetime = field(default_factory=datetime.now)
    progress: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """
    Executes persona simulation experiments.
    
    Features:
    - Async execution for performance
    - Checkpointing for resumption
    - Progress callbacks
    - Rate limit handling
    - Cost tracking
    """
    
    def __init__(
        self,
        checkpoints_dir: Optional[Path] = None,
        max_concurrent: int = 5
    ):
        self.checkpoints_dir = checkpoints_dir or config.checkpoints_dir
        self.max_concurrent = max_concurrent
        
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()
        
        self._progress: Optional[ExperimentProgress] = None
        self._checkpoint: Optional[Checkpoint] = None
        self._results: List[SimulationTrial] = []
        self._is_running: bool = False
        self._should_stop: bool = False
        
        # Callbacks
        self._progress_callback: Optional[Callable] = None
        self._trial_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[ExperimentProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def set_trial_callback(self, callback: Callable[[SimulationTrial], None]):
        """Set callback for completed trials."""
        self._trial_callback = callback
    
    async def run_experiment(
        self,
        experiment: ExperimentConfig,
        personas: Dict[str, Persona],
        emails: Dict[str, EmailStimulus],
        resume: bool = False
    ) -> List[SimulationTrial]:
        """
        Run a complete experiment.
        
        Args:
            experiment: Experiment configuration
            personas: Dict of persona_id -> Persona
            emails: Dict of email_id -> EmailStimulus
            resume: Whether to resume from checkpoint
        
        Returns:
            List of completed simulation trials
        """
        print(f"[ENGINE] run_experiment called")
        print(f"[ENGINE] Personas available: {list(personas.keys())}")
        print(f"[ENGINE] Emails available: {list(emails.keys())}")
        print(f"[ENGINE] Experiment persona_ids: {experiment.persona_ids}")
        print(f"[ENGINE] Experiment email_ids: {experiment.email_ids}")
        print(f"[ENGINE] Experiment model_ids: {experiment.model_ids}")
        
        self._is_running = True
        self._should_stop = False
        self._results = []
        
        # Load checkpoint if resuming
        completed_ids = set()
        if resume:
            checkpoint = self._load_checkpoint(experiment.experiment_id)
            if checkpoint:
                completed_ids = set(checkpoint.completed_trial_ids)
                self._results = []  # Will reload from storage if needed
        
        # Build trial queue
        trials_to_run = self._build_trial_queue(
            experiment, personas, emails, completed_ids
        )
        
        print(f"[ENGINE] Built {len(trials_to_run)} trials to run")
        
        if len(trials_to_run) == 0:
            print(f"[ENGINE] WARNING: No trials to run! Check if personas/emails match experiment config")
            self._is_running = False
            return []
        
        # Initialize progress
        self._progress = ExperimentProgress(
            experiment_id=experiment.experiment_id,
            total_trials=len(trials_to_run) + len(completed_ids),
            completed_trials=len(completed_ids),
            started_at=datetime.now()
        )
        
        # Get router
        router = await get_router()
        print(f"[ENGINE] Router initialized, providers: {router.get_initialized_providers()}")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Run trials
        async def run_trial(trial: SimulationTrial) -> SimulationTrial:
            async with semaphore:
                if self._should_stop:
                    return trial
                return await self._execute_single_trial(trial, router)
        
        # Execute with progress tracking
        tasks = [run_trial(trial) for trial in trials_to_run]
        print(f"[ENGINE] Starting {len(tasks)} trial tasks...")
        
        for coro in asyncio.as_completed(tasks):
            if self._should_stop:
                break
            
            trial = await coro
            self._results.append(trial)
            
            # Update progress
            self._progress.completed_trials += 1
            if not trial.parse_success:
                self._progress.failed_trials += 1
            self._progress.cost_so_far += trial.cost_usd
            self._progress.current_persona = trial.persona_id
            self._progress.current_model = trial.model_id
            self._progress.current_email = trial.email_id
            
            # Estimate remaining time
            elapsed = (datetime.now() - self._progress.started_at).seconds
            if self._progress.completed_trials > 0:
                avg_time = elapsed / self._progress.completed_trials
                remaining = self._progress.total_trials - self._progress.completed_trials
                self._progress.estimated_remaining_seconds = int(avg_time * remaining)
            
            # Callbacks
            if self._progress_callback:
                self._progress_callback(self._progress)
            if self._trial_callback:
                self._trial_callback(trial)
            
            # Checkpoint every 50 trials
            if self._progress.completed_trials % 50 == 0:
                self._save_checkpoint(experiment.experiment_id)
        
        # Final checkpoint
        self._save_checkpoint(experiment.experiment_id)
        
        self._is_running = False
        print(f"[ENGINE] Experiment completed with {len(self._results)} results")
        return self._results
    
    async def run_single(
        self,
        persona: Persona,
        email: EmailStimulus,
        model_id: str,
        prompt_config: PromptConfiguration,
        temperature: float = 0.3
    ) -> SimulationTrial:
        """
        Run a single simulation trial.
        
        Useful for testing and quick validation.
        """
        trial = SimulationTrial(
            trial_id=str(uuid.uuid4()),
            experiment_id="single_trial",
            persona_id=persona.persona_id,
            model_id=model_id,
            prompt_config=prompt_config,
            email_id=email.email_id,
            trial_number=1,
            system_prompt="",
            user_prompt="",
            temperature=temperature
        )
        
        router = await get_router()
        return await self._execute_single_trial(trial, router, persona, email)
    
    async def _execute_single_trial(
        self,
        trial: SimulationTrial,
        router,
        persona: Optional[Persona] = None,
        email: Optional[EmailStimulus] = None
    ) -> SimulationTrial:
        """Execute a single trial and return updated trial with results."""
        
        # Build prompt if persona/email provided (MODIFIED)
        if persona and email:
            built = self.prompt_builder.build(
                persona=persona,
                email=email,
                config=trial.prompt_config,
                trial_number=trial.trial_number  # NEW: Pass trial number
            )
            trial.system_prompt = built.system_prompt
            trial.user_prompt = built.user_prompt
            trial.temperature = built.temperature  # NEW: Use recommended temperature
        
        # Create LLM request (MODIFIED)
        request = LLMRequest(
            system_prompt=trial.system_prompt,
            user_prompt=trial.user_prompt,
            temperature=trial.temperature,
            top_p=0.95,  # NEW: Add nucleus sampling
            max_tokens=500,
            request_id=trial.trial_id
        )
        
        try:
            # Call LLM
            response = await router.complete(trial.model_id, request)
            
            # Update trial with response
            trial.raw_response = response.content
            trial.model_latency_ms = response.latency_ms
            trial.input_tokens = response.input_tokens
            trial.output_tokens = response.output_tokens
            trial.cost_usd = response.cost_usd
            trial.timestamp = datetime.now()
            
            if response.success:
                # Parse response
                parsed = self.response_parser.parse(
                    response.content,
                    trial.prompt_config.value
                )
                
                trial.action = parsed['action']
                trial.confidence = parsed.get('confidence')
                trial.simulated_speed = parsed.get('speed')
                trial.reasoning_text = parsed.get('reasoning')
                trial.parse_success = parsed['parse_success']
            else:
                trial.action = ActionType.ERROR
                trial.parse_success = False
                trial.error_message = response.error_message
                
        except Exception as e:
            trial.action = ActionType.ERROR
            trial.parse_success = False
            trial.error_message = str(e)
            trial.timestamp = datetime.now()
    
        return trial
    
    def _build_trial_queue(
        self,
        experiment: ExperimentConfig,
        personas: Dict[str, Persona],
        emails: Dict[str, EmailStimulus],
        completed_ids: set
    ) -> List[SimulationTrial]:
        """Build queue of trials to run."""
    
        trials = []
    
        for persona_id in experiment.persona_ids:
            persona = personas.get(persona_id)
            if not persona:
                print(f"[ENGINE] WARNING: Persona {persona_id} not found in personas dict")
                continue
        
            for model_id in experiment.model_ids:
                for prompt_config in experiment.prompt_configs:
                    for email_id in experiment.email_ids:
                        email = emails.get(email_id)
                        if not email:
                            print(f"[ENGINE] WARNING: Email {email_id} not found in emails dict")
                            continue
                        
                        for trial_num in range(1, experiment.trials_per_condition + 1):
                            trial_id = f"{experiment.experiment_id}_{persona_id}_{model_id}_{prompt_config.value}_{email_id}_{trial_num}"
                            
                            if trial_id in completed_ids:
                                continue
                            
                            # Build prompt WITH trial number for context variation (MODIFIED)
                            built = self.prompt_builder.build(
                                persona=persona,
                                email=email,
                                config=prompt_config,
                                trial_number=trial_num  # NEW
                            )
                            
                            trial = SimulationTrial(
                                trial_id=trial_id,
                                experiment_id=experiment.experiment_id,
                                persona_id=persona_id,
                                model_id=model_id,
                                prompt_config=prompt_config,
                                email_id=email_id,
                                trial_number=trial_num,
                                system_prompt=built.system_prompt,
                                user_prompt=built.user_prompt,
                                temperature=built.temperature  # MODIFIED: Use persona-specific
                            )
                            
                            trials.append(trial)
        
        print(f"[ENGINE] Built {len(trials)} trials from {len(experiment.persona_ids)} personas, {len(experiment.model_ids)} models, {len(experiment.email_ids)} emails")
    
        # Shuffle to distribute load across models
        random.shuffle(trials)
    
        return trials
    
    def _save_checkpoint(self, experiment_id: str):
        """Save checkpoint to disk."""
        checkpoint = Checkpoint(
            experiment_id=experiment_id,
            completed_trial_ids=[t.trial_id for t in self._results],
            last_saved=datetime.now(),
            progress={
                'completed': self._progress.completed_trials if self._progress else 0,
                'failed': self._progress.failed_trials if self._progress else 0,
                'cost': self._progress.cost_so_far if self._progress else 0,
            }
        )
        
        checkpoint_path = self.checkpoints_dir / f"{experiment_id}.checkpoint.json"
        
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'experiment_id': checkpoint.experiment_id,
                'completed_trial_ids': checkpoint.completed_trial_ids,
                'last_saved': checkpoint.last_saved.isoformat(),
                'progress': checkpoint.progress
            }, f)
    
    def _load_checkpoint(self, experiment_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from disk."""
        checkpoint_path = self.checkpoints_dir / f"{experiment_id}.checkpoint.json"
        
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        return Checkpoint(
            experiment_id=data['experiment_id'],
            completed_trial_ids=data['completed_trial_ids'],
            last_saved=datetime.fromisoformat(data['last_saved']),
            progress=data.get('progress', {})
        )
    
    def stop(self):
        """Stop the running experiment."""
        self._should_stop = True
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def progress(self) -> Optional[ExperimentProgress]:
        return self._progress
    
    @property
    def results(self) -> List[SimulationTrial]:
        return self._results


# Async generator for streaming results
async def stream_experiment(
    experiment: ExperimentConfig,
    personas: Dict[str, Persona],
    emails: Dict[str, EmailStimulus],
    max_concurrent: int = 5
) -> AsyncGenerator[SimulationTrial, None]:
    """
    Stream experiment results as they complete.
    
    Usage:
        async for trial in stream_experiment(exp, personas, emails):
            print(f"Completed: {trial.trial_id}")
    """
    engine = ExecutionEngine(max_concurrent=max_concurrent)
    
    results_queue = asyncio.Queue()
    
    def on_trial(trial: SimulationTrial):
        results_queue.put_nowait(trial)
    
    engine.set_trial_callback(on_trial)
    
    # Start experiment in background
    task = asyncio.create_task(
        engine.run_experiment(experiment, personas, emails)
    )
    
    # Yield results as they come
    while not task.done() or not results_queue.empty():
        try:
            trial = await asyncio.wait_for(results_queue.get(), timeout=0.1)
            yield trial
        except asyncio.TimeoutError:
            continue
    
    # Wait for task to complete
    await task