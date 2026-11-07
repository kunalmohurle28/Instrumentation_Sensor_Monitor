"""Statistical Anomaly Detection Module.

Implements univariate statistical anomaly detection using the Z-Score method:
    Z = (X - μ) / σ

Identifies statistical outliers that significantly deviate from the baseline empirical
mean of the sample distribution.

Handles edge cases including zero variance (constant signals), missing data (NaN),
and non-numeric sensor readings.

DISCLAIMER: This module utilizes classical parametric statistical dispersion metrics.
It is NOT an Artificial Intelligence or Machine Learning algorithm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class AnomalyDetector:
    """Computes Z-scores and detects statistical outliers per sensor channel."""

    def __init__(self, threshold: float = 3.0) -> None:
        """Initialize the Anomaly Detector with a Z-score cutoff.

        Args:
            threshold: Absolute Z-score cutoff for outlier classification (must be > 0).

        Raises:
            ValueError: If threshold <= 0.
        """
        if threshold <= 0:
            raise ValueError(f"Anomaly threshold must be strictly positive, got: {threshold}")
        self.threshold = float(threshold)

    def set_threshold(self, threshold: float) -> None:
        """Update the Z-score detection threshold.

        Args:
            threshold: New positive Z-score threshold.
        """
        if threshold <= 0:
            raise ValueError(f"Anomaly threshold must be strictly positive, got: {threshold}")
        self.threshold = float(threshold)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        value_col: str = "reading_value",
        group_col: str = "sensor_id",
    ) -> pd.DataFrame:
        """Compute grouped Z-scores and annotate anomalies per sensor channel.

        Args:
            df: Input DataFrame containing sensor records.
            value_col: Name of the numeric measurement column.
            group_col: Column to partition data for per-sensor statistics.

        Returns:
            DataFrame augmented with ['mean', 'std_dev', 'z_score', 'is_anomaly', 'anomaly_reason'].
        """
        if df.empty:
            res = df.copy()
            res["mean"] = pd.Series(dtype="float64")
            res["std_dev"] = pd.Series(dtype="float64")
            res["z_score"] = pd.Series(dtype="float64")
            res["is_anomaly"] = pd.Series(dtype="bool")
            res["anomaly_reason"] = pd.Series(dtype="object")
            return res

        result = df.copy()

        # Coerce numeric values cleanly
        numeric_series = pd.to_numeric(result[value_col], errors="coerce")

        means = np.full(len(result), np.nan)
        stds = np.full(len(result), np.nan)
        z_scores = np.full(len(result), np.nan)
        is_anomaly = np.zeros(len(result), dtype=bool)
        reasons: List[Optional[str]] = [None] * len(result)

        # Process per sensor group to evaluate relative deviation accurately
        grouped = result.groupby(group_col)

        for _, group_indices in grouped.groups.items():
            idx = group_indices.to_numpy()
            vals = numeric_series.iloc[idx].dropna().to_numpy()

            if len(vals) < 2:
                # Insufficient points to evaluate standard deviation
                for i in idx:
                    reasons[i] = "Insufficient sample points for Z-Score"
                continue

            grp_mean = float(np.mean(vals))
            grp_std = float(np.std(vals, ddof=1))

            for i in idx:
                val = numeric_series.iloc[i]
                means[i] = grp_mean
                stds[i] = grp_std

                if pd.isna(val):
                    reasons[i] = "Missing / NaN reading"
                    continue

                if grp_std == 0.0 or np.isclose(grp_std, 0.0):
                    # Zero standard deviation: all valid samples in group are identical
                    if not np.isclose(val, grp_mean):
                        z_scores[i] = np.inf
                        is_anomaly[i] = True
                        reasons[i] = f"Deviation from constant signal (mean={grp_mean})"
                    else:
                        z_scores[i] = 0.0
                else:
                    z = (val - grp_mean) / grp_std
                    z_scores[i] = round(z, 3)

                    if abs(z) > self.threshold:
                        is_anomaly[i] = True
                        reasons[i] = (
                            f"Z-score |{round(z, 2)}| > {self.threshold} (mean={round(grp_mean, 2)}, std={round(grp_std, 2)})"
                        )

        result["mean"] = means
        result["std_dev"] = stds
        result["z_score"] = z_scores
        result["is_anomaly"] = is_anomaly
        result["anomaly_reason"] = reasons
        return result

    def get_anomaly_summary(self, annotated_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out normal readings and return only detected statistical anomalies.

        Args:
            annotated_df: DataFrame output from `detect_anomalies`.

        Returns:
            Filtered DataFrame of anomalous records.
        """
        if annotated_df.empty or "is_anomaly" not in annotated_df.columns:
            return pd.DataFrame()

        anomalies_df = annotated_df[annotated_df["is_anomaly"]].copy()
        return anomalies_df.reset_index(drop=True)

    def compute_anomaly_metrics(self, annotated_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical anomaly count, rates, and distribution across sensors.

        Args:
            annotated_df: DataFrame output from `detect_anomalies`.

        Returns:
            Dictionary with anomaly statistics.
        """
        if annotated_df.empty:
            return {
                "total_readings": 0,
                "anomaly_count": 0,
                "anomaly_rate_pct": 0.0,
                "sensor_anomalies": {},
                "threshold_used": self.threshold,
            }

        total_readings = len(annotated_df)
        anomaly_count = int(annotated_df["is_anomaly"].sum())
        anomaly_rate = (anomaly_count / total_readings) * 100.0 if total_readings > 0 else 0.0

        sensor_anomalies: Dict[str, Dict[str, Any]] = {}
        grouped = annotated_df.groupby("sensor_id")
        for sid, group in grouped:
            g_total = len(group)
            g_anom = int(group["is_anomaly"].sum())
            g_rate = (g_anom / g_total) * 100.0 if g_total > 0 else 0.0
            sensor_anomalies[str(sid)] = {
                "total": g_total,
                "anomalies": g_anom,
                "anomaly_rate_pct": round(g_rate, 2),
            }

        return {
            "total_readings": total_readings,
            "anomaly_count": anomaly_count,
            "anomaly_rate_pct": round(anomaly_rate, 2),
            "sensor_anomalies": sensor_anomalies,
            "threshold_used": self.threshold,
        }
