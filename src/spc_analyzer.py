"""Statistical Process Control (SPC) Analysis Module.

Implements Shewhart individual control chart metrics (X-bar / I-chart):
- Center Line (CL) = Process Sample Mean (μ)
- Upper Control Limit (UCL) = μ + 3σ
- Lower Control Limit (LCL) = μ - 3σ
- Upper Warning Limit (UWL) = μ + 2σ
- Lower Warning Limit (LWL) = μ - 2σ

Evaluates standard industrial out-of-control rules:
- Rule 1 (Outlier): 1 point beyond Zone A (> UCL or < LCL).
- Rule 2 (Shift/Trend): 8 consecutive points on one side of the Center Line.
- Rule 3 (Zone Alert): 2 out of 3 consecutive points in Zone A or beyond.

Optional capability indices: Cp and Cpk when specification limits are provided.

DISCLAIMER: In this educational prototype, control limits are dynamically estimated
from the available sample set. In production industrial automation (Phase II SPC),
control limits must be calculated from a validated, stable baseline reference period
and frozen before live monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SPCControlLimits:
    """Calculated statistical control limits for a process parameter."""

    center_line: float
    std_dev: float
    ucl: float
    lcl: float
    uwl: float
    lwl: float
    sample_size: int


@dataclass
class SPCSummary:
    """Summary of SPC evaluation for a sensor channel."""

    sensor_id: str
    measurement_type: str
    unit: str
    limits: SPCControlLimits
    out_of_control_count: int
    out_of_control_pct: float
    process_status: str  # "IN_CONTROL" or "OUT_OF_CONTROL"
    violations: List[Dict[str, Any]]
    cp: Optional[float] = None
    cpk: Optional[float] = None


class SPCAnalyzer:
    """Calculates statistical process control limits and identifies out-of-control signals."""

    def __init__(self, sigma_multiplier: float = 3.0) -> None:
        """Initialize the SPC Analyzer.

        Args:
            sigma_multiplier: Multiplier for control limits (standard is 3.0 for 99.73% coverage).
        """
        if sigma_multiplier <= 0:
            raise ValueError(f"Sigma multiplier must be positive, got: {sigma_multiplier}")
        self.sigma_multiplier = float(sigma_multiplier)

    def calculate_control_limits(
        self, values: pd.Series | np.ndarray | List[float]
    ) -> Optional[SPCControlLimits]:
        """Compute Shewhart control limits from numeric sample values.

        Args:
            values: Series, array, or list of numeric process readings.

        Returns:
            SPCControlLimits dataclass or None if sample size < 2.
        """
        clean_vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
        n = len(clean_vals)
        if n < 2:
            return None

        mean = float(np.mean(clean_vals))
        std = float(np.std(clean_vals, ddof=1))

        ucl = mean + self.sigma_multiplier * std
        lcl = mean - self.sigma_multiplier * std
        uwl = mean + 2.0 * std
        lwl = mean - 2.0 * std

        return SPCControlLimits(
            center_line=round(mean, 3),
            std_dev=round(std, 3),
            ucl=round(ucl, 3),
            lcl=round(lcl, 3),
            uwl=round(uwl, 3),
            lwl=round(lwl, 3),
            sample_size=n,
        )

    def calculate_capability_indices(
        self,
        mean: float,
        std: float,
        usl: Optional[float],
        lsl: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate process capability indices Cp and Cpk.

        Cp  = (USL - LSL) / (6 * σ)
        Cpk = min((USL - μ) / (3σ), (μ - LSL) / (3σ))

        Args:
            mean: Process mean (μ).
            std: Process standard deviation (σ).
            usl: Upper Specification Limit.
            lsl: Lower Specification Limit.

        Returns:
            Tuple of (Cp, Cpk) or (None, None) if limits missing or std is 0.
        """
        if std <= 0.0 or usl is None or lsl is None:
            return None, None
        if usl <= lsl:
            return None, None

        cp = (usl - lsl) / (6.0 * std)
        cpu = (usl - mean) / (3.0 * std)
        cpl = (mean - lsl) / (3.0 * std)
        cpk = min(cpu, cpl)

        return round(float(cp), 3), round(float(cpk), 3)

    def evaluate_sensor_spc(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        value_col: str = "reading_value",
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
    ) -> Optional[SPCSummary]:
        """Evaluate a specific sensor's process data against SPC rules.

        Args:
            df: DataFrame containing the process data.
            sensor_id: Identifier of the sensor to evaluate.
            value_col: Name of column containing readings.
            usl: Optional Upper Specification Limit for capability evaluation.
            lsl: Optional Lower Specification Limit for capability evaluation.

        Returns:
            SPCSummary object or None if sensor not found or insufficient data.
        """
        sensor_df = df[df["sensor_id"] == sensor_id].copy()
        if sensor_df.empty:
            return None

        # Sort chronologically if timestamp exists
        if "timestamp" in sensor_df.columns:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            sensor_df = sensor_df.sort_values(by="timestamp").reset_index(drop=True)

        numeric_vals = pd.to_numeric(sensor_df[value_col], errors="coerce")
        valid_mask = numeric_vals.notna()
        valid_indices = sensor_df.index[valid_mask].tolist()
        vals = numeric_vals[valid_mask].to_numpy()

        limits = self.calculate_control_limits(vals)
        if limits is None:
            return None

        m_type = str(sensor_df["measurement_type"].iloc[0]) if "measurement_type" in sensor_df.columns else "unknown"
        unit = str(sensor_df["unit"].iloc[0]) if "unit" in sensor_df.columns else ""

        violations: List[Dict[str, Any]] = []
        out_of_control_indices = set()

        mean = limits.center_line
        ucl = limits.ucl
        lcl = limits.lcl
        uwl = limits.uwl
        lwl = limits.lwl

        # Rule 1: Point Beyond Control Limits (> UCL or < LCL)
        for i, val in enumerate(vals):
            orig_idx = valid_indices[i]
            ts = sensor_df.loc[orig_idx, "timestamp"] if "timestamp" in sensor_df.columns else str(orig_idx)

            if val > ucl:
                out_of_control_indices.add(orig_idx)
                violations.append(
                    {
                        "index": orig_idx,
                        "timestamp": ts,
                        "value": round(val, 2),
                        "rule": "Rule 1: Exceeds UCL",
                        "severity": "CRITICAL",
                        "limit_violated": ucl,
                    }
                )
            elif val < lcl:
                out_of_control_indices.add(orig_idx)
                violations.append(
                    {
                        "index": orig_idx,
                        "timestamp": ts,
                        "value": round(val, 2),
                        "rule": "Rule 1: Subceeds LCL",
                        "severity": "CRITICAL",
                        "limit_violated": lcl,
                    }
                )

        # Rule 2: 8 consecutive points on one side of center line (Run)
        run_count = 0
        current_side = 0  # 1 for above, -1 for below
        for i, val in enumerate(vals):
            orig_idx = valid_indices[i]
            side = 1 if val > mean else (-1 if val < mean else 0)
            if side == current_side and side != 0:
                run_count += 1
            else:
                current_side = side
                run_count = 1 if side != 0 else 0

            if run_count == 8:
                out_of_control_indices.add(orig_idx)
                ts = sensor_df.loc[orig_idx, "timestamp"] if "timestamp" in sensor_df.columns else str(orig_idx)
                direction = "above" if current_side == 1 else "below"
                violations.append(
                    {
                        "index": orig_idx,
                        "timestamp": ts,
                        "value": round(val, 2),
                        "rule": f"Rule 2: Process Shift (8 consecutive points {direction} mean)",
                        "severity": "WARNING",
                        "limit_violated": mean,
                    }
                )

        # Rule 3: 2 out of 3 consecutive points in Warning Zone A (> UWL or < LWL)
        for i in range(2, len(vals)):
            window = vals[i - 2 : i + 1]
            above_count = sum(1 for w in window if w > uwl)
            below_count = sum(1 for w in window if w < lwl)
            if above_count >= 2 or below_count >= 2:
                orig_idx = valid_indices[i]
                if orig_idx not in out_of_control_indices:
                    out_of_control_indices.add(orig_idx)
                    ts = sensor_df.loc[orig_idx, "timestamp"] if "timestamp" in sensor_df.columns else str(orig_idx)
                    violations.append(
                        {
                            "index": orig_idx,
                            "timestamp": ts,
                            "value": round(vals[i], 2),
                            "rule": "Rule 3: Warning Zone Alert (2 of 3 points beyond 2-sigma)",
                            "severity": "WARNING",
                            "limit_violated": uwl if above_count >= 2 else lwl,
                        }
                    )

        ooc_count = len(out_of_control_indices)
        total_valid = len(vals)
        ooc_pct = round((ooc_count / total_valid) * 100.0, 2) if total_valid > 0 else 0.0
        status = "IN_CONTROL" if ooc_count == 0 else "OUT_OF_CONTROL"

        cp, cpk = self.calculate_capability_indices(limits.center_line, limits.std_dev, usl, lsl)

        return SPCSummary(
            sensor_id=sensor_id,
            measurement_type=m_type,
            unit=unit,
            limits=limits,
            out_of_control_count=ooc_count,
            out_of_control_pct=ooc_pct,
            process_status=status,
            violations=violations,
            cp=cp,
            cpk=cpk,
        )

    def annotate_spc_dataframe(
        self,
        df: pd.DataFrame,
        sensor_id: str,
        value_col: str = "reading_value",
    ) -> pd.DataFrame:
        """Annotate sensor DataFrame with SPC limit columns and out-of-control flag.

        Args:
            df: Full DataFrame.
            sensor_id: Sensor ID to evaluate.
            value_col: Reading value column.

        Returns:
            DataFrame containing only the sensor's rows, annotated with:
                ['ucl', 'lcl', 'uwl', 'lwl', 'center_line', 'is_ooc', 'spc_rule_violated']
        """
        sensor_df = df[df["sensor_id"] == sensor_id].copy()
        if sensor_df.empty:
            return sensor_df

        if "timestamp" in sensor_df.columns:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            sensor_df = sensor_df.sort_values(by="timestamp").reset_index(drop=True)

        spc_res = self.evaluate_sensor_spc(sensor_df, sensor_id, value_col=value_col)
        if spc_res is None:
            return sensor_df

        lims = spc_res.limits
        sensor_df["center_line"] = lims.center_line
        sensor_df["ucl"] = lims.ucl
        sensor_df["lcl"] = lims.lcl
        sensor_df["uwl"] = lims.uwl
        sensor_df["lwl"] = lims.lwl

        sensor_df["is_ooc"] = False
        sensor_df["spc_rule_violated"] = None

        violation_map = {v["index"]: v["rule"] for v in spc_res.violations}
        for idx, rule in violation_map.items():
            if idx in sensor_df.index:
                sensor_df.loc[idx, "is_ooc"] = True
                sensor_df.loc[idx, "spc_rule_violated"] = rule

        return sensor_df
