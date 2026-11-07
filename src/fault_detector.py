"""Automatic Sensor Fault Detection Module.

Implements threshold-based fault detection and classification for industrial telemetry:
- Configurable Upper Operating Limit (UOL) and Lower Operating Limit (LOL).
- Classification: NORMAL, HIGH_FAULT, LOW_FAULT, INVALID_READING.
- Individual sensor status and comprehensive system health assessment.
- Fault event logging and summary reporting.

DISCLAIMER: Default threshold limits are educational examples for demonstration
and must be adjusted to match actual plant piping and instrumentation diagrams (P&ID).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class OperatingLimits:
    """Configurable operating thresholds and physical boundaries for a sensor."""

    lower_limit: float
    upper_limit: float
    min_physical: float = -float("inf")
    max_physical: float = float("inf")
    unit: str = ""


# Configurable default baseline operating thresholds (Educational Reference Values)
DEFAULT_OPERATING_LIMITS: Dict[str, OperatingLimits] = {
    "TEMP_001": OperatingLimits(
        lower_limit=50.0,
        upper_limit=90.0,
        min_physical=-273.15,  # Absolute zero
        max_physical=500.0,
        unit="°C",
    ),
    "PRES_001": OperatingLimits(
        lower_limit=120.0,
        upper_limit=180.0,
        min_physical=0.0,  # Absolute vacuum
        max_physical=1000.0,
        unit="kPa",
    ),
    "LVL_001": OperatingLimits(
        lower_limit=20.0,
        upper_limit=80.0,
        min_physical=0.0,  # 0% empty tank
        max_physical=100.0,  # 100% full tank
        unit="%",
    ),
}

STATUS_NORMAL = "NORMAL"
STATUS_HIGH_FAULT = "HIGH_FAULT"
STATUS_LOW_FAULT = "LOW_FAULT"
STATUS_INVALID = "INVALID_READING"


class FaultDetector:
    """Evaluates sensor readings against operational thresholds and physical constraints."""

    def __init__(
        self,
        limits: Optional[Dict[str, OperatingLimits]] = None,
    ) -> None:
        """Initialize the Fault Detector with customizable thresholds.

        Args:
            limits: Dictionary mapping sensor_id to OperatingLimits instance.
        """
        self.limits = limits if limits is not None else DEFAULT_OPERATING_LIMITS.copy()

    def set_sensor_limits(
        self,
        sensor_id: str,
        lower_limit: float,
        upper_limit: float,
        min_physical: Optional[float] = None,
        max_physical: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> None:
        """Configure or update operating limits for a specific sensor.

        Args:
            sensor_id: Target sensor identifier.
            lower_limit: Lower threshold below which a LOW_FAULT is triggered.
            upper_limit: Upper threshold above which a HIGH_FAULT is triggered.
            min_physical: Optional physical lower boundary.
            max_physical: Optional physical upper boundary.
            unit: Unit of measurement.

        Raises:
            ValueError: If lower_limit >= upper_limit.
        """
        if lower_limit >= upper_limit:
            raise ValueError(
                f"Lower limit ({lower_limit}) must be strictly less than upper limit ({upper_limit})."
            )

        existing = self.limits.get(
            sensor_id,
            OperatingLimits(lower_limit=lower_limit, upper_limit=upper_limit),
        )
        self.limits[sensor_id] = OperatingLimits(
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            min_physical=min_physical if min_physical is not None else existing.min_physical,
            max_physical=max_physical if max_physical is not None else existing.max_physical,
            unit=unit if unit is not None else existing.unit,
        )

    def classify_reading(
        self, sensor_id: str, value: Any
    ) -> Tuple[str, Optional[str], Optional[float]]:
        """Classify a single reading value.

        Args:
            sensor_id: Sensor identifier to lookup limits.
            value: Sensor reading value (numeric or potential NaN/None/String).

        Returns:
            Tuple of (fault_status, fault_description, threshold_violated).
        """
        if sensor_id not in self.limits:
            return STATUS_NORMAL, None, None

        lim = self.limits[sensor_id]

        # Check for missing, None, or non-numeric values
        if value is None or pd.isna(value):
            return STATUS_INVALID, "Telemetry Dropout / Missing Value", None

        try:
            numeric_val = float(value)
        except (ValueError, TypeError):
            return STATUS_INVALID, f"Non-numeric reading: '{value}'", None

        # Check physical feasibility bounds
        if numeric_val < lim.min_physical:
            return (
                STATUS_INVALID,
                f"Value {numeric_val} below physical lower limit ({lim.min_physical} {lim.unit})",
                lim.min_physical,
            )
        if numeric_val > lim.max_physical:
            return (
                STATUS_INVALID,
                f"Value {numeric_val} exceeds physical upper limit ({lim.max_physical} {lim.unit})",
                lim.max_physical,
            )

        # Check operational limits
        if numeric_val > lim.upper_limit:
            return (
                STATUS_HIGH_FAULT,
                f"Exceeded Upper Threshold ({lim.upper_limit} {lim.unit}) by {round(numeric_val - lim.upper_limit, 2)}",
                lim.upper_limit,
            )
        if numeric_val < lim.lower_limit:
            return (
                STATUS_LOW_FAULT,
                f"Subceeded Lower Threshold ({lim.lower_limit} {lim.unit}) by {round(lim.lower_limit - numeric_val, 2)}",
                lim.lower_limit,
            )

        return STATUS_NORMAL, None, None

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame of sensor readings and annotate fault classification.

        Args:
            df: DataFrame containing at least ['sensor_id', 'reading_value'].

        Returns:
            New DataFrame with added columns:
                ['fault_status', 'fault_description', 'threshold_violated', 'is_fault']
        """
        if df.empty:
            result = df.copy()
            result["fault_status"] = pd.Series(dtype="object")
            result["fault_description"] = pd.Series(dtype="object")
            result["threshold_violated"] = pd.Series(dtype="float64")
            result["is_fault"] = pd.Series(dtype="bool")
            return result

        result = df.copy()
        statuses: List[str] = []
        descriptions: List[Optional[str]] = []
        thresholds: List[Optional[float]] = []
        is_fault_list: List[bool] = []

        for _, row in result.iterrows():
            sid = str(row.get("sensor_id", ""))
            val = row.get("reading_value", np.nan)
            status, desc, thresh = self.classify_reading(sid, val)
            statuses.append(status)
            descriptions.append(desc)
            thresholds.append(thresh)
            is_fault_list.append(status != STATUS_NORMAL)

        result["fault_status"] = statuses
        result["fault_description"] = descriptions
        result["threshold_violated"] = thresholds
        result["is_fault"] = is_fault_list
        return result

    def get_fault_summary(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        """Extract a filtered summary table containing only fault and invalid events.

        Args:
            annotated_df: DataFrame returned by `analyze_dataframe`.

        Returns:
            Filtered DataFrame of anomalous events.
        """
        if annotated_df.empty or "fault_status" not in annotated_df.columns:
            return pd.DataFrame()

        faults_df = annotated_df[annotated_df["fault_status"] != STATUS_NORMAL].copy()
        return faults_df.reset_index(drop=True)

    def evaluate_system_health(self, annotated_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute system-wide and individual sensor health diagnostics.

        Args:
            annotated_df: DataFrame output from `analyze_dataframe`.

        Returns:
            Dictionary with overall system status, total readings, fault count,
            fault percentage, and sensor-wise status breakdown.
        """
        if annotated_df.empty:
            return {
                "overall_status": "NO_DATA",
                "total_readings": 0,
                "fault_count": 0,
                "fault_percentage": 0.0,
                "sensor_status": {},
            }

        total_readings = len(annotated_df)
        fault_readings = int(annotated_df["is_fault"].sum())
        fault_rate = (fault_readings / total_readings) * 100.0 if total_readings > 0 else 0.0

        # Assess individual sensor status
        sensor_status: Dict[str, Dict[str, Any]] = {}
        grouped = annotated_df.groupby("sensor_id")
        for sid, group in grouped:
            s_total = len(group)
            s_faults = int(group["is_fault"].sum())
            s_high = int((group["fault_status"] == STATUS_HIGH_FAULT).sum())
            s_low = int((group["fault_status"] == STATUS_LOW_FAULT).sum())
            s_invalid = int((group["fault_status"] == STATUS_INVALID).sum())
            s_rate = (s_faults / s_total) * 100.0 if s_total > 0 else 0.0

            latest_row = group.iloc[-1]
            latest_status = latest_row.get("fault_status", STATUS_NORMAL)

            sensor_status[str(sid)] = {
                "total": s_total,
                "faults": s_faults,
                "high_faults": s_high,
                "low_faults": s_low,
                "invalid_readings": s_invalid,
                "fault_percentage": round(s_rate, 2),
                "latest_status": latest_status,
                "latest_value": latest_row.get("reading_value"),
            }

        # Overall Plant / System Status Logic
        if fault_readings == 0:
            overall_status = "NORMAL"
        elif fault_rate < 5.0 and not any(
            s["invalid_readings"] > 0 for s in sensor_status.values()
        ):
            overall_status = "WARNING"
        else:
            overall_status = "CRITICAL"

        return {
            "overall_status": overall_status,
            "total_readings": total_readings,
            "fault_count": fault_readings,
            "fault_percentage": round(fault_rate, 2),
            "sensor_status": sensor_status,
        }
