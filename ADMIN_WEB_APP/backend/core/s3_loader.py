"""
S3 Data Loader for CYPEARL Admin Web App

Loads experiment data from AWS S3 bucket with intelligent caching and fallback.

Features:
- S3 with local file fallback for development
- In-memory caching with TTL to reduce S3 calls
- Supports the enhanced pipeline's prefix structure

Usage:
    from core.s3_loader import s3_loader

    participants_df, responses_df = s3_loader.load_study_data()
    email_stimuli_df = s3_loader.load_email_stimuli()
    s3_loader.refresh()  # Force reload from S3
"""

import os
import pandas as pd
from io import StringIO
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3DataLoader:
    """
    Loads research data from S3 with caching and local fallback.

    S3 Bucket Structure (matches s3_export.py):
        processed/
            phishing_study_responses.csv
            phishing_study_participants.csv
            email_stimuli.csv
        archive/
            YYYY-MM-DD_HHMMSS/  (snapshots)
    """

    def __init__(self, cache_ttl_minutes: int = 5):
        """
        Initialize S3 client if credentials are available.

        Args:
            cache_ttl_minutes: How long to cache data before re-fetching (default 5 min)
        """
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'cypearl-research-data')
        self.enabled = False
        self.region = os.getenv('AWS_REGION', 'eu-west-1')

        # S3 prefix (matches s3_export.py)
        self.prefix = 'processed/'

        # Caching
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Initialize S3 client
        if BOTO3_AVAILABLE and os.getenv('AWS_ACCESS_KEY_ID'):
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=self.region
                )
                # Verify bucket access
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.enabled = True
                print(f"✓ S3 Data Loader initialized (bucket: {self.bucket_name})")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == '404':
                    print(f"✗ S3 bucket '{self.bucket_name}' does not exist")
                elif error_code == '403':
                    print(f"✗ Access denied to S3 bucket '{self.bucket_name}'")
                else:
                    print(f"✗ S3 initialization failed: {e}")
            except Exception as e:
                print(f"✗ S3 initialization failed: {e}")
        else:
            if not BOTO3_AVAILABLE:
                print("⚠ S3 Data Loader disabled: boto3 not installed")
            else:
                print("⚠ S3 Data Loader disabled: AWS_ACCESS_KEY_ID not set")

        # Local data directory fallback
        self.data_dir = Path(__file__).parent.parent.parent / "data"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return datetime.now() - cached['timestamp'] < self.cache_ttl

    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        if self._is_cache_valid(key):
            return self._cache[key]['data'].copy()
        return None

    def _set_cached(self, key: str, data: pd.DataFrame) -> None:
        """Store data in cache."""
        self._cache[key] = {
            'data': data.copy(),
            'timestamp': datetime.now()
        }

    def load_study_data(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load participant and response data.

        Args:
            use_cache: If True, return cached data if available (default True)

        Returns:
            Tuple of (participants_df, responses_df)
        """
        cache_key = 'study_data'

        # Check cache
        if use_cache:
            participants = self._get_cached('participants')
            responses = self._get_cached('responses')
            if participants is not None and responses is not None:
                return participants, responses

        # Try S3 first
        if self.enabled:
            try:
                participants = self._load_from_s3('phishing_study_participants.csv')
                responses = self._load_from_s3('phishing_study_responses.csv')
                print(f"✓ Loaded study data from S3 ({len(participants)} participants, {len(responses)} responses)")

                # Cache the data
                self._set_cached('participants', participants)
                self._set_cached('responses', responses)

                return participants, responses
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print("⚠ S3 files not found, falling back to local")
                else:
                    print(f"⚠ S3 load failed, falling back to local: {e}")
            except Exception as e:
                print(f"⚠ S3 load failed, falling back to local: {e}")

        # Fallback to local files
        participants_path = self.data_dir / "phishing_study_participants.csv"
        responses_path = self.data_dir / "phishing_study_responses.csv"

        participants = pd.read_csv(participants_path) if participants_path.exists() else pd.DataFrame()
        responses = pd.read_csv(responses_path) if responses_path.exists() else pd.DataFrame()

        print(f"✓ Loaded study data from local files ({len(participants)} participants, {len(responses)} responses)")

        # Cache local data too
        self._set_cached('participants', participants)
        self._set_cached('responses', responses)

        return participants, responses

    def load_email_stimuli(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load email stimuli data.

        Args:
            use_cache: If True, return cached data if available (default True)

        Returns:
            DataFrame with email stimuli
        """
        cache_key = 'email_stimuli'

        # Check cache
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Try S3 first
        if self.enabled:
            try:
                df = self._load_from_s3('email_stimuli.csv')
                print(f"✓ Loaded email stimuli from S3 ({len(df)} emails)")
                self._set_cached(cache_key, df)
                return df
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print("⚠ S3 email_stimuli.csv not found, falling back to local")
                else:
                    print(f"⚠ S3 load failed, falling back to local: {e}")
            except Exception as e:
                print(f"⚠ S3 load failed, falling back to local: {e}")

        # Fallback to local
        local_path = self.data_dir / "email_stimuli.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)
            print(f"✓ Loaded email stimuli from local file ({len(df)} emails)")
            self._set_cached(cache_key, df)
            return df

        return pd.DataFrame()

    def _load_from_s3(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from S3 with the configured prefix."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")

        key = f"{self.prefix}{filename}"
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        content = response['Body'].read().decode('utf-8')
        return pd.read_csv(StringIO(content))

    def refresh(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Force reload data from source (bypasses cache).

        Returns:
            Tuple of (participants_df, responses_df)
        """
        # Clear cache
        self._cache.clear()
        return self.load_study_data(use_cache=False)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def get_data_source(self) -> str:
        """Return the current data source ('s3' or 'local')."""
        return 's3' if self.enabled else 'local'

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the data loader."""
        status = {
            "source": self.get_data_source(),
            "s3_enabled": self.enabled,
            "bucket": self.bucket_name if self.enabled else None,
            "region": self.region if self.enabled else None,
            "prefix": self.prefix,
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60,
            "cached_keys": list(self._cache.keys()),
            "local_fallback_dir": str(self.data_dir)
        }

        # Add cache freshness info
        for key in self._cache:
            age = datetime.now() - self._cache[key]['timestamp']
            status[f"cache_{key}_age_seconds"] = age.total_seconds()

        return status

    def get_s3_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about an S3 file.

        Args:
            filename: The CSV filename (without prefix)

        Returns:
            Dict with file metadata or None if not found/not enabled
        """
        if not self.enabled:
            return None

        try:
            key = f"{self.prefix}{filename}"
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                "key": key,
                "size_bytes": response.get('ContentLength'),
                "last_modified": response.get('LastModified').isoformat() if response.get('LastModified') else None,
                "metadata": response.get('Metadata', {})
            }
        except ClientError:
            return None


# Singleton instance
s3_loader = S3DataLoader()
