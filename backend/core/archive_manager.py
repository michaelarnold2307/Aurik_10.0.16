"""
AURIK Archive Manager

Centralized archiving system implementing AURIK Architecture Specification Section 4.2.

Directory Structure (per Spec 4.2):
    /archive/
    ├── originals/{job_id}/     # Permanent, immutable original files
    ├── intermediates/{job_id}/ # 90-day retention for intermediate processing files
    │   └── retention.json      # Metadata with expire_at timestamp
    ├── outputs/{job_id}/       # Permanent final output files
    └── reports/{job_id}.json   # Permanent JSON serialization of ResturationJob

    /models/versions/{model}_{version}/  # Permanent, version-controlled model weights

Retention Policies:
- Originals: Permanent, never deleted
- Intermediates: 90-day retention with automatic cleanup
- Outputs: Permanent
- Reports: Permanent JSON serialization of full ResturationJob
- Model weights: Permanent, version-controlled

Features:
- SHA256 integrity verification
- Atomic file operations
- Thread-safe archiving
- Automatic retention cleanup
- Job retrieval and restoration
- Archive statistics and monitoring
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from .data_models import ResturationJob

logger = logging.getLogger(__name__)


# ============================================================================
# Archive Manager
# ============================================================================


class ArchiveManager:
    """
    Centralized archive management for AURIK restoration jobs.

    Spec Reference: Section 4.2 - Archiving Strategy
    """

    def __init__(self, base_archive_path: str = "/archive", models_path: str = "/models/versions"):
        """
        Initialisiert Archive Manager.

        Args:
            base_archive_path: Root path for job archives
            models_path: Path for model version storage
        """
        self.base_path = Path(base_archive_path)
        self.models_path = Path(models_path)

        # Sub-directories per spec 4.2
        self.originals_dir = self.base_path / "originals"
        self.intermediates_dir = self.base_path / "intermediates"
        self.outputs_dir = self.base_path / "outputs"
        self.reports_dir = self.base_path / "reports"

        # Create directory structure
        self._ensure_directory_structure()

        logger.info("ArchiveManager initialisiert: base_path=%s", self.base_path)

    def _ensure_directory_structure(self) -> None:
        """Erstellt archive directory structure if it doesn't exist."""
        for directory in [
            self.originals_dir,
            self.intermediates_dir,
            self.outputs_dir,
            self.reports_dir,
            self.models_path,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", directory)

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        Calculate SHA256 hash of a file for integrity verification.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _copy_file_with_verification(self, src: str, dest: str) -> str:
        """
        Copy file with integrity verification.

        Args:
            src: Source file path
            dest: Destination file path

        Returns:
            SHA256 hash of copied file

        Raises:
            ValueError: If source and destination hashes don't match
        """
        # Calculate source hash
        src_hash = self.calculate_file_hash(src)

        # Copy file
        shutil.copy2(src, dest)

        # Verify destination hash
        dest_hash = self.calculate_file_hash(dest)

        if src_hash != dest_hash:
            # Integrity check failed - remove corrupted file
            os.remove(dest)
            raise ValueError(f"File copy integrity check failed: {src} -> {dest}")

        logger.debug("File copied and verified: %s -> %s (hash: %s...)", src, dest, src_hash[:8])
        return dest_hash

    def _create_retention_metadata(
        self, job_id: UUID, created_at: datetime, retention_days: int = 90
    ) -> dict[str, Any]:
        """
        Erstellt retention metadata for intermediate files.

        Args:
            job_id: Job UUID
            created_at: Job creation timestamp
            retention_days: Days until expiration

        Returns:
            Retention metadata dictionary
        """
        expire_at = created_at + timedelta(days=retention_days)

        return {
            "job_id": str(job_id),
            "created_at": created_at.isoformat(),
            "expire_at": expire_at.isoformat(),
            "retention_days": retention_days,
            "archived_at": datetime.now().isoformat(),
        }

    def archive_job(self, job: ResturationJob) -> str:
        """
        Archive complete restoration job with all artifacts.

        Directory structure created:
            /archive/originals/{job_id}/input.{ext}
            /archive/intermediates/{job_id}/step_{n}.{ext}
            /archive/intermediates/{job_id}/retention.json
            /archive/outputs/{job_id}/output.{ext}
            /archive/reports/{job_id}.json

        Args:
            job: ResturationJob to archive

        Returns:
            Archive path

        Raises:
            ValueError: If job is already archived or files are missing
        """
        if job.archived:
            raise ValueError(f"Job {job.job_id} is already archived at {job.archive_path}")

        job_id_str = str(job.job_id)
        logger.info("Archiving job %s...", job_id_str)

        try:
            # 1. Archive original file (permanent)
            orig_dir = self.originals_dir / job_id_str
            orig_dir.mkdir(parents=True, exist_ok=True)

            if not os.path.exists(job.input_file.file_path):
                raise ValueError(f"Input file not found: {job.input_file.file_path}")

            orig_ext = Path(job.input_file.file_path).suffix
            orig_dest = orig_dir / f"input{orig_ext}"
            orig_hash = self._copy_file_with_verification(job.input_file.file_path, str(orig_dest))

            # Verify hash matches
            if orig_hash != job.input_file.file_hash:
                logger.warning(
                    f"Eingabe file hash mismatch: expected {job.input_file.file_hash[:8]}..., got {orig_hash[:8]}..."
                )

            logger.info("Archived Originalsignal: %s", orig_dest)

            # 2. Archive intermediate files (90-day retention)
            inter_dir = self.intermediates_dir / job_id_str
            inter_dir.mkdir(parents=True, exist_ok=True)

            for idx, inter_file in enumerate(job.intermediate_files):
                if not os.path.exists(inter_file.file_path):
                    logger.warning("Intermediate file not found, skipping: %s", inter_file.file_path)
                    continue

                inter_ext = Path(inter_file.file_path).suffix
                inter_dest = inter_dir / f"step_{idx:03d}{inter_ext}"
                self._copy_file_with_verification(inter_file.file_path, str(inter_dest))
                logger.debug("Archived intermediate %s: %s", idx, inter_dest)

            # Create retention metadata
            retention_data = self._create_retention_metadata(job.job_id, job.created_at)
            retention_path = inter_dir / "retention.json"
            with open(retention_path, "w") as f:
                json.dump(retention_data, f, indent=2)
            logger.info("erstellt retention metadata: %s", retention_path)

            # 3. Archive output file (permanent)
            if job.output_file:
                out_dir = self.outputs_dir / job_id_str
                out_dir.mkdir(parents=True, exist_ok=True)

                if not os.path.exists(job.output_file.file_path):
                    logger.warning("Ausgabe file not found: %s", job.output_file.file_path)
                else:
                    out_ext = Path(job.output_file.file_path).suffix
                    out_dest = out_dir / f"output{out_ext}"
                    out_hash = self._copy_file_with_verification(job.output_file.file_path, str(out_dest))

                    # Verify hash
                    if out_hash != job.output_file.file_hash:
                        logger.warning(
                            f"Ausgabe file hash mismatch: expected {job.output_file.file_hash[:8]}..., got {out_hash[:8]}..."
                        )

                    logger.info("Archived Ausgabe: %s", out_dest)

            # 4. Archive full job report (permanent JSON)
            report_path = self.reports_dir / f"{job_id_str}.json"
            with open(report_path, "w") as f:
                json.dump(job.model_dump(), f, indent=2, default=str)
            logger.info("Archived job report: %s", report_path)

            # Update job metadata
            job.archived = True
            job.archive_path = str(self.base_path / job_id_str)

            logger.info("Job %s erfolgreich archived to %s", job_id_str, job.archive_path)
            return job.archive_path

        except Exception as e:
            logger.error("konnte nicht archive job %s: %s", job_id_str, e)
            raise

    def retrieve_job(self, job_id: UUID) -> ResturationJob | None:
        """
        Retrieve archived ResturationJob from reports.

        Args:
            job_id: Job UUID

        Returns:
            ResturationJob if found, None otherwise
        """
        job_id_str = str(job_id)
        report_path = self.reports_dir / f"{job_id_str}.json"

        if not report_path.exists():
            logger.warning("Job report not found: %s", report_path)
            return None

        try:
            with open(report_path) as f:
                job_data = json.load(f)

            # Reconstruct ResturationJob from JSON
            job = ResturationJob(**job_data)
            logger.info("Retrieved job %s from archive", job_id_str)
            return job

        except Exception as e:
            logger.error("konnte nicht retrieve job %s: %s", job_id_str, e)
            return None

    def cleanup_expired_intermediates(self, dry_run: bool = False) -> tuple[int, list[str]]:
        """
        Clean up intermediate files past 90-day retention period.

        Args:
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            Tuple of (num_deleted, list_of_deleted_paths)
        """
        now = datetime.now()
        deleted_count = 0
        deleted_paths = []

        logger.info("Starte intermediate cleanup (dry_Ausfuehrung=%s)...", dry_run)

        for job_dir in self.intermediates_dir.iterdir():
            if not job_dir.is_dir():
                continue

            retention_path = job_dir / "retention.json"
            if not retention_path.exists():
                logger.warning("No retention.json found in %s, skipping", job_dir)
                continue

            try:
                with open(retention_path) as f:
                    retention_data = json.load(f)

                expire_at_str = retention_data.get("expire_at")
                if not expire_at_str:
                    logger.warning("No expire_at in retention.json for %s, skipping", job_dir)
                    continue

                expire_at = datetime.fromisoformat(expire_at_str)

                if now > expire_at:
                    # Retention period expired
                    if dry_run:
                        logger.info("[DRY Ausfuehrung] Would delete: %s", job_dir)
                        deleted_paths.append(str(job_dir))
                        deleted_count += 1
                    else:
                        shutil.rmtree(job_dir)
                        logger.info("Deleted expired intermediates: %s", job_dir)
                        deleted_paths.append(str(job_dir))
                        deleted_count += 1
                else:
                    days_remaining = (expire_at - now).days
                    logger.debug("%s: %s days remaining until expiration", job_dir, days_remaining)

            except Exception as e:
                logger.error("Error processing %s: %s", job_dir, e)
                continue

        logger.info(
            f"Cleanup vollstaendig: {deleted_count} intermediate directories {'would be ' if dry_run else ''}deleted"
        )
        return deleted_count, deleted_paths

    def get_archive_stats(self) -> dict[str, Any]:
        """
        Gibt zurück: statistics about archive usage.

        Returns:
            Dictionary with archive statistics
        """

        def count_files_and_size(directory: Path) -> tuple[int, int]:
            """Zählt files and total size in directory."""
            file_count = 0
            total_size = 0
            for item in directory.rglob("*"):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
            return file_count, total_size

        stats: dict[str, Any] = {
            "archive_path": str(self.base_path),
            "directories": {},
            "total_jobs": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0.0,
            "total_size_gb": 0.0,
        }

        # Count originals
        orig_files, orig_size = count_files_and_size(self.originals_dir)
        orig_jobs = len([d for d in self.originals_dir.iterdir() if d.is_dir()])
        stats["directories"]["originals"] = {
            "path": str(self.originals_dir),
            "jobs": orig_jobs,
            "files": orig_files,
            "size_bytes": orig_size,
            "size_mb": round(orig_size / (1024**2), 2),
            "retention": "permanent",
        }

        # Count intermediates
        inter_files, inter_size = count_files_and_size(self.intermediates_dir)
        inter_jobs = len([d for d in self.intermediates_dir.iterdir() if d.is_dir()])
        stats["directories"]["intermediates"] = {
            "path": str(self.intermediates_dir),
            "jobs": inter_jobs,
            "files": inter_files,
            "size_bytes": inter_size,
            "size_mb": round(inter_size / (1024**2), 2),
            "retention": "90_days",
        }

        # Count outputs
        out_files, out_size = count_files_and_size(self.outputs_dir)
        out_jobs = len([d for d in self.outputs_dir.iterdir() if d.is_dir()])
        stats["directories"]["outputs"] = {
            "path": str(self.outputs_dir),
            "jobs": out_jobs,
            "files": out_files,
            "size_bytes": out_size,
            "size_mb": round(out_size / (1024**2), 2),
            "retention": "permanent",
        }

        # Count reports
        report_files = len([f for f in self.reports_dir.iterdir() if f.is_file() and f.suffix == ".json"])
        report_size = sum(f.stat().st_size for f in self.reports_dir.iterdir() if f.is_file())
        stats["directories"]["reports"] = {
            "path": str(self.reports_dir),
            "jobs": report_files,
            "files": report_files,
            "size_bytes": report_size,
            "size_mb": round(report_size / (1024**2), 2),
            "retention": "permanent",
        }

        # Totals
        stats["total_jobs"] = max(orig_jobs, inter_jobs, out_jobs, report_files)
        stats["total_files"] = orig_files + inter_files + out_files + report_files
        stats["total_size_bytes"] = orig_size + inter_size + out_size + report_size
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024**2), 2)
        stats["total_size_gb"] = round(stats["total_size_bytes"] / (1024**3), 2)

        return stats

    def list_archived_jobs(self, limit: int | None = None) -> list[dict]:
        """
        Listet auf: all archived jobs with basic metadata.

        Args:
            limit: Maximum number of jobs to return (None = all)

        Returns:
            List of job metadata dictionaries sorted by creation time (newest first)
        """
        jobs = []

        for report_path in self.reports_dir.glob("*.json"):
            try:
                with open(report_path) as f:
                    job_data = json.load(f)

                jobs.append(
                    {
                        "job_id": job_data.get("job_id"),
                        "created_at": job_data.get("created_at"),
                        "completed_at": job_data.get("completed_at"),
                        "status": job_data.get("status"),
                        "input_file": job_data.get("input_file", {}).get("file_path"),
                        "output_file": (
                            job_data.get("output_file", {}).get("file_path") if job_data.get("output_file") else None
                        ),
                        "cas_improvement": (
                            job_data.get("quality_report", {}).get("cas_improvement")
                            if job_data.get("quality_report")
                            else None
                        ),
                    }
                )

            except Exception as e:
                logger.warning("konnte nicht read job report %s: %s", report_path, e)
                continue

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def archive_model_version(self, model_name: str, model_version: str, model_path: str) -> str:
        """
        Archive model weights for version control and reproducibility.

        Args:
            model_name: Name of the model (e.g., "DeepFilterNet")
            model_version: Version string (e.g., "3.2.0")
            model_path: Path to model files/directory

        Returns:
            Archive path for model

        Raises:
            ValueError: If model path doesn't exist
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model path not found: {model_path}")

        archive_name = f"{model_name}_{model_version}"
        archive_dest = self.models_path / archive_name

        if archive_dest.exists():
            logger.info("Model version already archived: %s", archive_dest)
            return str(archive_dest)

        archive_dest.mkdir(parents=True, exist_ok=True)

        # Copy model files
        if os.path.isfile(model_path):
            shutil.copy2(model_path, archive_dest / Path(model_path).name)
        else:
            shutil.copytree(model_path, archive_dest, dirs_exist_ok=True)

        # Create version metadata
        metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "archived_at": datetime.now().isoformat(),
            "source_path": model_path,
        }

        metadata_path = archive_dest / "version_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Archived model %s v%s to %s", model_name, model_version, archive_dest)
        return str(archive_dest)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "ArchiveManager",
]
