import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from phase1.constants import GET_ALL_CLUSTERING_FEATURES, OUTCOME_FEATURES

class DataLoader:
    """Load and validate study data."""
    
    def __init__(self, participants_path: str, responses_path: str = None):
        self.participants_path = Path(participants_path)
        self.responses_path = Path(responses_path) if responses_path else None
        
        self.participants = None
        self.responses = None
        self.data_stats = {}
    
    def load(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load data files and compute statistics."""
        
        # Load participants
        if not self.participants_path.exists():
             raise FileNotFoundError(f"Participants file not found at {self.participants_path}")

        self.participants = pd.read_csv(self.participants_path)
        self.data_stats['n_participants'] = len(self.participants)
        self.data_stats['n_features'] = len(self.participants.columns)
        
        # Load responses if provided
        if self.responses_path and self.responses_path.exists():
            self.responses = pd.read_csv(self.responses_path)
            self.data_stats['n_responses'] = len(self.responses)
            n_emails = self.responses['email_id'].nunique()
            self.data_stats['n_emails'] = n_emails
        
        # Check available features
        all_clustering_features = GET_ALL_CLUSTERING_FEATURES()
        available = [f for f in all_clustering_features if f in self.participants.columns]
        missing = [f for f in all_clustering_features if f not in self.participants.columns]
        
        self.data_stats['available_features'] = len(available)
        self.data_stats['missing_features'] = len(missing)
        self.data_stats['missing_feature_list'] = missing
        
        # Check outcomes
        outcome_available = [f for f in OUTCOME_FEATURES if f in self.participants.columns]
        self.data_stats['outcome_features_available'] = len(outcome_available)
        
        # Data quality
        if available:
            missing_pct = self.participants[available].isnull().mean().mean() * 100
            self.data_stats['missing_pct'] = missing_pct
        else:
            self.data_stats['missing_pct'] = 0.0
        
        return self.participants, self.responses
    
    def get_summary(self) -> Dict:
        """Get data summary statistics."""
        return self.data_stats
