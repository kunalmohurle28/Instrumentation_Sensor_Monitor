"""Data Logger and Pipeline Coordination Module.

Integrates data ingestion, automated fault detection, statistical anomaly
detection, and persistent database storage:
- Manages standard Python logging configuration.
- Executes the unified telemetry processing pipeline.
- Synchronously commits sensor readings, faults, and anomalies to the SQLite database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.anomaly_detector import AnomalyDetector
from src.database_manager import DatabaseManager
from src.fault_detector import FaultDetector

logger = logging.getLogger("IndustrialSensorLogger")


def setup_system_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure system logging for console outputs."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)
    return logger


class DataLogger:
    """Coordinates the logging, evaluation, and persistent storage of telemetry."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize the DataLogger.

        Args:
            db_manager: Instance of DatabaseManager. If None, instantiates a default.
            log_level: Python logging level.
        """
        setup_system_logging(log_level)
        self.db_manager = db_manager if db_manager is not None else DatabaseManager()

    def process_and_persist_batch(
        self,
        raw_df: pd.DataFrame,
        fault_detector: FaultDetector,
        anomaly_detector: AnomalyDetector,
    ) -> Dict[str, Any]:
        """Execute end-to-end detection pipeline on a telemetry batch and persist findings.

        Args:
            raw_df: Ingested telemetry DataFrame.
            fault_detector: Configured FaultDetector instance.
            anomaly_detector: Configured AnomalyDetector instance.

        Returns:
            Dictionary containing:
                - annotated_df: DataFrame with both fault and anomaly columns
                - fault_summary: DataFrame of fault events
                - anomaly_summary: DataFrame of anomaly events
                - health_status: System health dictionary
                - anomaly_metrics: Anomaly metrics dictionary
                - persisted_counts: Dict of counts saved to DB
        """
        if raw_df.empty:
            logger.warning("Empty batch passed to pipeline. Nothing to process.")
            return {
                "annotated_df": pd.DataFrame(),
                "fault_summary": pd.DataFrame(),
                "anomaly_summary": pd.DataFrame(),
                "health_status": {},
                "anomaly_metrics": {},
                "persisted_counts": {"readings": 0, "faults": 0, "anomalies": 0},
            }

        # Step 1: Run threshold-based fault detection
        fault_annotated = fault_detector.analyze_dataframe(raw_df)
        fault_summary = fault_detector.get_fault_summary(fault_annotated)
        health_status = fault_detector.evaluate_system_health(fault_annotated)

        # Step 2: Run statistical anomaly detection
        anomaly_annotated = anomaly_detector.detect_anomalies(fault_annotated)
        anomaly_summary = anomaly_detector.get_anomaly_summary(anomaly_annotated)
        anomaly_metrics = anomaly_detector.compute_anomaly_metrics(anomaly_annotated)

        # Step 3: Persist telemetry and events to database
        readings_saved = self.db_manager.save_readings(raw_df)
        faults_saved = self.db_manager.save_faults(fault_summary) if not fault_summary.empty else 0
        anomalies_saved = self.db_manager.save_anomalies(anomaly_summary) if not anomaly_summary.empty else 0

        logger.info(
            f"Processed batch of {len(raw_df)} readings: "
            f"Saved {readings_saved} readings, {faults_saved} faults, {anomalies_saved} anomalies."
        )

        return {
            "annotated_df": anomaly_annotated,
            "fault_summary": fault_summary,
            "anomaly_summary": anomaly_summary,
            "health_status": health_status,
            "anomaly_metrics": anomaly_metrics,
            "persisted_counts": {
                "readings": readings_saved,
                "faults": faults_saved,
                "anomalies": anomalies_saved,
            },
        }
