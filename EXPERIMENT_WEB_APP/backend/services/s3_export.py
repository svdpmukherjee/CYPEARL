"""
S3 Data Export Service for CYPEARL Experiment Web App

Enhanced data pipeline from MongoDB to AWS S3 for ADMIN_WEB_APP consumption.

Features:
- Deduplication: Prevents duplicate exports via export tracking
- Versioning: Creates timestamped snapshots for audit trail
- Incremental: Appends only new records to existing CSVs
- Reconciliation: Batch export to sync all data

Files exported:
- phishing_study_responses.csv: Per-email response data
- phishing_study_participants.csv: Aggregated participant data
- email_stimuli.csv: Static email design file (once)
- _export_metadata.json: Tracks export history for deduplication

Usage:
    from services.s3_export import s3_exporter

    await s3_exporter.export_participant_completion(participant_id, db)
    await s3_exporter.batch_reconcile(db)  # Full sync
"""

import os
import json
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorDatabase


class S3Exporter:
    """
    Production-ready S3 exporter with deduplication and tracking.

    Industry patterns implemented:
    - Export tracking to prevent duplicates
    - Atomic updates with read-modify-write
    - Metadata for audit trail
    - Batch reconciliation for data integrity
    """

    def __init__(self):
        """Initialize S3 client with environment variables."""
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'cypearl-research-data')
        self.enabled = False
        self.region = os.getenv('AWS_REGION', 'eu-west-1')

        # S3 key prefixes for organization
        self.prefix_processed = 'processed/'  # Analysis-ready CSVs
        self.prefix_archive = 'archive/'      # Versioned snapshots
        self.metadata_key = '_export_metadata.json'

        if os.getenv('AWS_ACCESS_KEY_ID'):
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
                print(f"✓ S3 Export initialized (bucket: {self.bucket_name})")
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
            print("⚠ S3 Export disabled: AWS_ACCESS_KEY_ID not set")

    # ─────────────────────────────────────────────────────────────────────────
    # Export Tracking & Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def _get_export_metadata(self) -> Dict[str, Any]:
        """Load export tracking metadata from S3."""
        if not self.enabled:
            return {"exported_participants": [], "last_export": None, "export_count": 0}

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.metadata_key
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return {"exported_participants": [], "last_export": None, "export_count": 0}
            raise

    def _save_export_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save export tracking metadata to S3."""
        if not self.enabled:
            return

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=self.metadata_key,
            Body=json.dumps(metadata, indent=2, default=str),
            ContentType='application/json'
        )

    def _mark_participant_exported(self, participant_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a participant as exported in metadata."""
        if participant_id not in metadata.get('exported_participants', []):
            metadata.setdefault('exported_participants', []).append(participant_id)
        metadata['last_export'] = datetime.now(timezone.utc).isoformat()
        metadata['export_count'] = metadata.get('export_count', 0) + 1
        return metadata

    def is_participant_exported(self, participant_id: str) -> bool:
        """Check if participant has already been exported (deduplication)."""
        if not self.enabled:
            return False
        metadata = self._get_export_metadata()
        return participant_id in metadata.get('exported_participants', [])

    # ─────────────────────────────────────────────────────────────────────────
    # Main Export Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def export_participant_completion(
        self,
        participant_id: str,
        db: AsyncIOMotorDatabase,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Export a completed participant's data to S3.

        Features:
        - Deduplication: Skips if already exported (unless force=True)
        - Atomic: Updates metadata only after successful export
        - Appends to existing CSVs without duplication

        Args:
            participant_id: The MongoDB ObjectId as string
            db: The MongoDB database instance
            force: If True, re-export even if already exported

        Returns:
            Status dict with export results
        """
        if not self.enabled:
            return {"status": "skipped", "reason": "S3 not configured"}

        # Deduplication check
        if not force and self.is_participant_exported(participant_id):
            return {
                "status": "skipped",
                "reason": "Already exported",
                "participant_id": participant_id
            }

        try:
            from bson import ObjectId

            # 1. Fetch participant and responses
            participant = await db.participants.find_one({"_id": ObjectId(participant_id)})
            if not participant:
                return {"status": "error", "reason": "Participant not found"}

            responses = await db.responses.find(
                {"participant_id": participant_id}
            ).to_list(length=100)

            if not responses:
                return {"status": "skipped", "reason": "No responses found"}

            # 2. Get current metadata
            metadata = self._get_export_metadata()

            # 3. Update CSVs
            responses_updated = await self._update_responses_csv(responses, participant_id)
            participants_updated = await self._update_participants_csv(participant, responses, db)

            # 4. Mark as exported (only after successful update)
            metadata = self._mark_participant_exported(participant_id, metadata)
            self._save_export_metadata(metadata)

            return {
                "status": "success",
                "participant_id": participant_id,
                "responses_exported": len(responses),
                "responses_csv_rows": responses_updated,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"status": "error", "reason": str(e), "participant_id": participant_id}

    async def _update_responses_csv(self, responses: List[Dict], participant_id: str) -> int:
        """
        Append new responses to the responses CSV in S3.
        Returns the total number of rows after update.
        """
        csv_key = f"{self.prefix_processed}phishing_study_responses.csv"

        # Column schema matching ADMIN_WEB_APP expectations
        columns = [
            'participant_id', 'email_id', 'email_type', 'sender_familiarity',
            'urgency_level', 'framing_type', 'phishing_quality', 'ground_truth',
            'action', 'clicked', 'reported', 'ignored', 'deleted', 'correct_response',
            'response_latency_ms', 'dwell_time_ms', 'hovered_link', 'inspected_sender',
            'confidence_rating', 'suspicion_rating', 'details_noticed', 'steps_taken',
            'decision_reason', 'confidence_reason', 'unsure_about'
        ]

        # Load existing CSV (if exists)
        try:
            existing_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=csv_key)
            existing_df = pd.read_csv(existing_obj['Body'])

            # Remove any existing rows for this participant (handles re-exports)
            existing_df = existing_df[existing_df['participant_id'] != participant_id]
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                existing_df = pd.DataFrame(columns=columns)
            else:
                raise

        # Prepare new rows
        new_rows = []
        for r in responses:
            row = {col: r.get(col) for col in columns}
            # Ensure binary fields are integers
            for binary_col in ['clicked', 'reported', 'ignored', 'deleted',
                              'correct_response', 'hovered_link', 'inspected_sender', 'ground_truth']:
                if row.get(binary_col) is not None:
                    row[binary_col] = int(row[binary_col])
            new_rows.append(row)

        # Combine and upload
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        csv_buffer = StringIO()
        combined_df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=csv_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv',
            Metadata={
                'last-updated': datetime.now(timezone.utc).isoformat(),
                'total-rows': str(len(combined_df)),
                'last-participant': participant_id
            }
        )

        return len(combined_df)

    async def _update_participants_csv(
        self,
        participant: Dict,
        responses: List[Dict],
        db: AsyncIOMotorDatabase
    ) -> int:
        """
        Append/update participant aggregate data in participants CSV.
        Returns the total number of participants after update.
        """
        csv_key = f"{self.prefix_processed}phishing_study_participants.csv"
        participant_id = str(participant['_id'])

        # Calculate aggregated metrics
        total = len(responses)
        if total == 0:
            return 0

        phishing_responses = [r for r in responses if r.get('ground_truth') == 1]
        legit_responses = [r for r in responses if r.get('ground_truth') == 0]

        correct_count = sum(1 for r in responses if r.get('correct_response') == 1)
        phishing_clicked = sum(1 for r in phishing_responses if r.get('clicked') == 1)
        legit_false_pos = sum(1 for r in legit_responses if r.get('action') in ['report', 'delete'])

        pre_survey = participant.get('pre_survey_data', {}) or {}
        post_survey = participant.get('post_survey_data', {}) or {}

        # Calculate response latency standard deviation
        latencies = [r.get('response_latency_ms', 0) for r in responses]
        mean_latency = sum(latencies) / total if total else 0
        variance = sum((x - mean_latency) ** 2 for x in latencies) / total if total else 0
        latency_sd = variance ** 0.5

        # Build participant record
        record = {
            'participant_id': participant_id,
            # Demographics
            'age': pre_survey.get('age'),
            'gender': pre_survey.get('gender'),
            'education': pre_survey.get('education'),
            'education_numeric': pre_survey.get('education_numeric'),
            'technical_field': pre_survey.get('technical_field'),
            'employment': pre_survey.get('employment'),
            'industry': pre_survey.get('industry'),
            # Cognitive
            'crt_score': pre_survey.get('crt_score'),
            'need_for_cognition': pre_survey.get('need_for_cognition'),
            'working_memory': pre_survey.get('working_memory'),
            # Big Five
            'big5_extraversion': pre_survey.get('big5_extraversion'),
            'big5_agreeableness': pre_survey.get('big5_agreeableness'),
            'big5_conscientiousness': pre_survey.get('big5_conscientiousness'),
            'big5_neuroticism': pre_survey.get('big5_neuroticism'),
            'big5_openness': pre_survey.get('big5_openness'),
            # Personality
            'impulsivity_total': pre_survey.get('impulsivity_total'),
            'sensation_seeking': pre_survey.get('sensation_seeking'),
            'trust_propensity': pre_survey.get('trust_propensity'),
            'risk_taking': pre_survey.get('risk_taking'),
            # State (post-experiment)
            'state_anxiety': post_survey.get('state_anxiety'),
            'current_stress': post_survey.get('current_stress'),
            'fatigue_level': post_survey.get('fatigue_level'),
            # Security attitudes
            'phishing_self_efficacy': pre_survey.get('phishing_self_efficacy'),
            'perceived_risk': pre_survey.get('perceived_risk'),
            'security_attitudes': pre_survey.get('security_attitudes'),
            'privacy_concern': pre_survey.get('privacy_concern'),
            # Knowledge & experience
            'phishing_knowledge': pre_survey.get('phishing_knowledge'),
            'technical_expertise': pre_survey.get('technical_expertise'),
            'prior_victimization': pre_survey.get('prior_victimization'),
            'security_training': pre_survey.get('security_training'),
            'years_email_use': pre_survey.get('years_email_use'),
            # Email habits
            'daily_email_volume': pre_survey.get('daily_email_volume'),
            'email_volume_numeric': pre_survey.get('email_volume_numeric'),
            'email_check_frequency': pre_survey.get('email_check_frequency'),
            'link_click_tendency': pre_survey.get('link_click_tendency'),
            'social_media_usage': pre_survey.get('social_media_usage'),
            # Influence susceptibility
            'authority_susceptibility': pre_survey.get('authority_susceptibility'),
            'urgency_susceptibility': pre_survey.get('urgency_susceptibility'),
            'scarcity_susceptibility': pre_survey.get('scarcity_susceptibility'),
            # Aggregated outcomes
            'overall_accuracy': round(correct_count / total, 3) if total else 0,
            'phishing_detection_rate': round(
                sum(1 for r in phishing_responses if r.get('correct_response') == 1) / len(phishing_responses), 3
            ) if phishing_responses else 0,
            'phishing_click_rate': round(
                phishing_clicked / len(phishing_responses), 3
            ) if phishing_responses else 0,
            'false_positive_rate': round(
                legit_false_pos / len(legit_responses), 3
            ) if legit_responses else 0,
            'report_rate': round(
                sum(1 for r in responses if r.get('action') == 'report') / total, 3
            ),
            'mean_response_latency': round(mean_latency),
            'mean_dwell_time': round(
                sum(r.get('dwell_time_ms', 0) for r in responses) / total
            ),
            'response_latency_sd': round(latency_sd),
            'hover_rate': round(
                sum(1 for r in responses if r.get('hovered_link') == 1) / total, 3
            ),
            'sender_inspection_rate': round(
                sum(1 for r in responses if r.get('inspected_sender') == 1) / total, 3
            ),
            'mean_confidence': round(
                sum(r.get('confidence_rating', 5) for r in responses) / total, 2
            ),
            'mean_suspicion_phishing': round(
                sum(r.get('suspicion_rating', 5) for r in phishing_responses) / len(phishing_responses), 2
            ) if phishing_responses else 0,
            'mean_suspicion_legit': round(
                sum(r.get('suspicion_rating', 5) for r in legit_responses) / len(legit_responses), 2
            ) if legit_responses else 0
        }

        # Load existing CSV (remove this participant if re-exporting)
        try:
            existing_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=csv_key)
            existing_df = pd.read_csv(existing_obj['Body'])
            existing_df = existing_df[existing_df['participant_id'] != participant_id]
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                existing_df = pd.DataFrame()
            else:
                raise

        # Combine and upload
        new_df = pd.DataFrame([record])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        csv_buffer = StringIO()
        combined_df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=csv_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv',
            Metadata={
                'last-updated': datetime.now(timezone.utc).isoformat(),
                'total-participants': str(len(combined_df)),
                'last-participant': participant_id
            }
        )

        return len(combined_df)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch Operations & Reconciliation
    # ─────────────────────────────────────────────────────────────────────────

    async def batch_reconcile(
        self,
        db: AsyncIOMotorDatabase,
        force_all: bool = False
    ) -> Dict[str, Any]:
        """
        Batch export all participants - useful for initial sync or reconciliation.

        This ensures S3 data matches MongoDB by:
        1. Finding all completed participants in MongoDB
        2. Exporting any not yet in S3 (or all if force_all=True)
        3. Creating a fresh snapshot in the archive folder

        Args:
            db: MongoDB database instance
            force_all: If True, re-export all participants (fresh rebuild)

        Returns:
            Summary of reconciliation results
        """
        if not self.enabled:
            return {"status": "skipped", "reason": "S3 not configured"}

        try:
            # Get all completed participants from MongoDB
            completed_participants = await db.participants.find(
                {"completed": True}
            ).to_list(length=10000)

            if not completed_participants:
                return {"status": "skipped", "reason": "No completed participants found"}

            # Get current export metadata
            metadata = self._get_export_metadata()
            exported_set: Set[str] = set(metadata.get('exported_participants', []))

            # Determine which participants need export
            results = {
                "total_completed": len(completed_participants),
                "already_exported": 0,
                "newly_exported": 0,
                "failed": 0,
                "errors": []
            }

            if force_all:
                # Reset metadata for full rebuild
                metadata = {"exported_participants": [], "last_export": None, "export_count": 0}
                # Clear existing CSVs for clean rebuild
                await self._clear_processed_csvs()

            for participant in completed_participants:
                participant_id = str(participant['_id'])

                if not force_all and participant_id in exported_set:
                    results["already_exported"] += 1
                    continue

                # Export this participant
                export_result = await self.export_participant_completion(
                    participant_id, db, force=True
                )

                if export_result.get("status") == "success":
                    results["newly_exported"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "participant_id": participant_id,
                        "error": export_result.get("reason")
                    })

            # Create archive snapshot if any exports happened
            if results["newly_exported"] > 0 or force_all:
                await self._create_archive_snapshot()

            results["status"] = "success"
            results["timestamp"] = datetime.now(timezone.utc).isoformat()
            return results

        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def _clear_processed_csvs(self) -> None:
        """Clear processed CSVs for a fresh rebuild."""
        keys = [
            f"{self.prefix_processed}phishing_study_responses.csv",
            f"{self.prefix_processed}phishing_study_participants.csv"
        ]
        for key in keys:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            except ClientError:
                pass  # File doesn't exist, that's fine

    async def _create_archive_snapshot(self) -> Dict[str, Any]:
        """
        Create a timestamped snapshot of current CSVs in the archive folder.
        Useful for audit trail and rollback capability.
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')
        snapshot_prefix = f"{self.prefix_archive}{timestamp}/"

        files_archived = []
        for filename in ['phishing_study_responses.csv', 'phishing_study_participants.csv']:
            source_key = f"{self.prefix_processed}{filename}"
            dest_key = f"{snapshot_prefix}{filename}"

            try:
                # Copy from processed to archive
                self.s3_client.copy_object(
                    Bucket=self.bucket_name,
                    Key=dest_key,
                    CopySource={'Bucket': self.bucket_name, 'Key': source_key}
                )
                files_archived.append(filename)
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    raise

        return {"snapshot": snapshot_prefix, "files": files_archived}

    # ─────────────────────────────────────────────────────────────────────────
    # Email Stimuli Export (Static, one-time)
    # ─────────────────────────────────────────────────────────────────────────

    async def export_email_stimuli(self, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
        """
        Export email stimuli to S3 (static file, run once at experiment setup).
        """
        if not self.enabled:
            return {"status": "skipped", "reason": "S3 not configured"}

        try:
            emails = await db.emails.find({"experimental": True}).to_list(length=100)

            if not emails:
                return {"status": "error", "reason": "No experimental emails found"}

            # Build records matching ADMIN_WEB_APP schema
            records = []
            for i, email in enumerate(emails, 1):
                factorial = email.get('factorial_category', {}) or {}
                records.append({
                    'email_id': f"E{i:02d}",
                    'email_type': 'phishing' if email.get('is_phishing') else 'legitimate',
                    'sender_familiarity': factorial.get('sender', 'unknown'),
                    'urgency_level': factorial.get('urgency', 'low'),
                    'framing_type': factorial.get('framing', 'neutral'),
                    'subject_line': email.get('subject', ''),
                    'phishing_quality': email.get('phishing_quality'),
                    'ground_truth': 1 if email.get('is_phishing') else 0
                })

            # Upload to S3
            df = pd.DataFrame(records)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            csv_key = f"{self.prefix_processed}email_stimuli.csv"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=csv_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv',
                Metadata={
                    'created': datetime.now(timezone.utc).isoformat(),
                    'total-emails': str(len(records))
                }
            )

            return {
                "status": "success",
                "emails_exported": len(records),
                "s3_key": csv_key,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"status": "error", "reason": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # Survey Exports (Raw data backup)
    # ─────────────────────────────────────────────────────────────────────────

    async def export_pre_survey_responses(self, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
        """Export all pre-survey responses to S3."""
        if not self.enabled:
            return {"status": "skipped", "reason": "S3 not configured"}

        try:
            responses = await db.pre_survey_responses.find({}).to_list(length=10000)

            if not responses:
                return {"status": "skipped", "reason": "No pre-survey responses found"}

            records = []
            for r in responses:
                record = {k: v for k, v in r.items() if k != '_id'}
                for date_field in ['submitted_at', 'completed_at', 'linked_at']:
                    if date_field in record and record[date_field]:
                        record[date_field] = record[date_field].isoformat() if hasattr(record[date_field], 'isoformat') else record[date_field]
                record.pop('raw_responses', None)
                records.append(record)

            df = pd.DataFrame(records)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            csv_key = f"{self.prefix_processed}pre_survey_responses.csv"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=csv_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv',
                Metadata={
                    'last-updated': datetime.now(timezone.utc).isoformat(),
                    'total-responses': str(len(records))
                }
            )

            return {
                "status": "success",
                "responses_exported": len(records),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def export_post_survey_responses(self, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
        """Export all post-survey responses to S3."""
        if not self.enabled:
            return {"status": "skipped", "reason": "S3 not configured"}

        try:
            responses = await db.post_survey_responses.find({}).to_list(length=10000)

            if not responses:
                return {"status": "skipped", "reason": "No post-survey responses found"}

            records = []
            for r in responses:
                record = {k: v for k, v in r.items() if k != '_id'}
                for date_field in ['submitted_at', 'study_started_at', 'study_completed_at']:
                    if date_field in record and record[date_field]:
                        record[date_field] = record[date_field].isoformat() if hasattr(record[date_field], 'isoformat') else record[date_field]
                record.pop('raw_responses', None)
                records.append(record)

            df = pd.DataFrame(records)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            csv_key = f"{self.prefix_processed}post_survey_responses.csv"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=csv_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv',
                Metadata={
                    'last-updated': datetime.now(timezone.utc).isoformat(),
                    'total-responses': str(len(records))
                }
            )

            return {
                "status": "success",
                "responses_exported": len(records),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {"status": "error", "reason": str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_signed_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a pre-signed URL for secure access to S3 objects."""
        if not self.enabled:
            return None

        try:
            # Prepend processed prefix if not already present
            if not key.startswith(self.prefix_processed) and not key.startswith(self.prefix_archive):
                key = f"{self.prefix_processed}{key}"

            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of S3 export service."""
        status = {
            "enabled": self.enabled,
            "bucket": self.bucket_name if self.enabled else None,
            "region": self.region if self.enabled else None,
            "prefix_processed": self.prefix_processed,
            "prefix_archive": self.prefix_archive
        }

        if self.enabled:
            try:
                metadata = self._get_export_metadata()
                status["exported_participants_count"] = len(metadata.get('exported_participants', []))
                status["last_export"] = metadata.get('last_export')
                status["total_exports"] = metadata.get('export_count', 0)
            except Exception as e:
                status["metadata_error"] = str(e)

        return status


# Singleton instance for import
s3_exporter = S3Exporter()
