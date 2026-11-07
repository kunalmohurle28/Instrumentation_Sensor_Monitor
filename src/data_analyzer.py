"""Industrial Process Data Analysis Module.

Performs rigorous descriptive statistical analysis on industrial time-series telemetry
using Pandas and NumPy:
- Central tendency metrics (Mean, Median).
- Dispersion metrics (Standard Deviation, Variance, Range, Interquartile Range IQR).
- Sample accounting (Total readings, Valid counts, Missing/NaN counts).
- Per-sensor channel summaries and system-wide aggregations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataAnalyzer:
    """Calculates descriptive and engineering statistics for sensor measurements."""

    @staticmethod
    def calculate_sensor_statistics(
        series: pd.Series, unit: str = ""
    ) -> Dict[str, Any]:
        """Compute statistical summary for a single numeric series.

        Args:
            series: Pandas Series of sensor readings.
            unit: Engineering unit string for display formatting.

        Returns:
            Dictionary containing descriptive statistics.
        """
        total_count = len(series)
        numeric_series = pd.to_numeric(series, errors="coerce")
        valid_series = numeric_series.dropna()
        valid_count = len(valid_series)
        missing_count = total_count - valid_count

        if valid_count == 0:
            return {
                "total_count": total_count,
                "valid_count": 0,
                "missing_count": missing_count,
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "std_dev": np.nan,
                "variance": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "iqr": np.nan,
                "unit": unit,
            }

        vals = valid_series.to_numpy()
        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        val_range = max_val - min_val
        std_val = float(np.std(vals, ddof=1)) if valid_count > 1 else 0.0
        var_val = float(np.var(vals, ddof=1)) if valid_count > 1 else 0.0
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        iqr = q75 - q25

        return {
            "total_count": total_count,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "range": round(val_range, 2),
            "std_dev": round(std_val, 2),
            "variance": round(var_val, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "unit": unit,
        }

    def generate_sensor_summary_table(
        self,
        df: pd.DataFrame,
        value_col: str = "reading_value",
        group_col: str = "sensor_id",
    ) -> pd.DataFrame:
        """Generate a structured multi-sensor comparative statistical summary table.

        Args:
            df: Input DataFrame of sensor measurements.
            value_col: Column containing reading values.
            group_col: Column to group by (typically sensor_id).

        Returns:
            DataFrame where rows represent sensors and columns represent metrics.
        """
        if df.empty or group_col not in df.columns or value_col not in df.columns:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for sid, group in df.groupby(group_col):
            unit = str(group["unit"].iloc[0]) if "unit" in group.columns else ""
            m_type = (
                str(group["measurement_type"].iloc[0])
                if "measurement_type" in group.columns
                else ""
            )

            stats = self.calculate_sensor_statistics(group[value_col], unit=unit)
            stats["sensor_id"] = str(sid)
            stats["measurement_type"] = m_type
            rows.append(stats)

        summary_df = pd.DataFrame(rows)
        # Reorder columns for professional engineering presentation
        preferred_order = [
            "sensor_id",
            "measurement_type",
            "unit",
            "total_count",
            "valid_count",
            "missing_count",
            "mean",
            "median",
            "min",
            "max",
            "range",
            "std_dev",
            "variance",
            "iqr",
        ]
        cols = [c for c in preferred_order if c in summary_df.columns] + [
            c for c in summary_df.columns if c not in preferred_order
        ]
        return summary_df[cols].reset_index(drop=True)

    def calculate_system_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall high-level system metrics across all channels.

        Args:
            df: Full telemetry DataFrame.

        Returns:
            Dictionary with aggregated statistics.
        """
        if df.empty:
            return {
                "total_readings": 0,
                "sensor_count": 0,
                "sensors": [],
                "time_span_start": None,
                "time_span_end": None,
                "missing_total": 0,
            }

        total_readings = len(df)
        sensor_ids = df["sensor_id"].unique().tolist() if "sensor_id" in df.columns else []
        sensor_count = len(sensor_ids)

        start_time = None
        end_time = None
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if not ts.empty:
                start_time = ts.min().isoformat()
                end_time = ts.max().isoformat()

        val_col = "reading_value" if "reading_value" in df.columns else None
        missing_count = 0
        if val_col:
            numeric_vals = pd.to_numeric(df[val_col], errors="coerce")
            missing_count = int(numeric_vals.isna().sum())

        return {
            "total_readings": total_readings,
            "sensor_count": sensor_count,
            "sensors": sensor_ids,
            "time_span_start": start_time,
            "time_span_end": end_time,
            "missing_total": missing_count,
        }
