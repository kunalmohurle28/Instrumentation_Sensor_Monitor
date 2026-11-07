"""Interactive Plotly Visualization Module.

Generates engineering charts for industrial telemetry:
- Real-time and historical time-series sensor trends with operational thresholds.
- Multi-channel synchronized overview subplots.
- Shewhart Statistical Process Control (SPC) X-charts with UCL, LCL, and Warning limits.
- Statistical anomaly dispersion charts (Z-Score distribution and thresholds).
- Empirical measurement distribution histograms.
- Fault classification breakdowns.

Uses only native Plotly components with professional engineering styling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Professional industrial SCADA-inspired color scheme
COLOR_NORMAL = "#1f77b4"      # Industrial blue
COLOR_HIGH_FAULT = "#d62728"  # Warning red
COLOR_LOW_FAULT = "#ff7f0e"   # Caution orange
COLOR_INVALID = "#9467bd"     # Purple
COLOR_UCL = "#e41a1c"         # Red limit
COLOR_LCL = "#e41a1c"         # Red limit
COLOR_CL = "#2ca02c"          # Green Center Line
COLOR_WARNING = "#ff7f00"     # Orange Warning line


def create_sensor_trend_chart(
    df: pd.DataFrame,
    sensor_id: str,
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Generate an interactive time-series trend chart for a specific sensor.

    Args:
        df: Telemetry DataFrame containing ['timestamp', 'sensor_id', 'reading_value'].
        sensor_id: Sensor ID to plot.
        upper_threshold: Optional upper operating threshold line.
        lower_threshold: Optional lower operating threshold line.
        title: Optional custom chart title.

    Returns:
        plotly.graph_objects.Figure
    """
    sensor_df = df[df["sensor_id"] == sensor_id].copy()
    if sensor_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected sensor")
        return fig

    if "timestamp" in sensor_df.columns:
        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
        sensor_df = sensor_df.sort_values(by="timestamp").reset_index(drop=True)

    unit = str(sensor_df["unit"].iloc[0]) if "unit" in sensor_df.columns else ""
    m_type = str(sensor_df["measurement_type"].iloc[0]).capitalize() if "measurement_type" in sensor_df.columns else ""

    chart_title = title or f"{m_type} Sensor Trend - {sensor_id} ({unit})"

    fig = go.Figure()

    # Base process line
    fig.add_trace(
        go.Scatter(
            x=sensor_df["timestamp"],
            y=sensor_df["reading_value"],
            mode="lines+markers",
            name=f"{sensor_id} Reading",
            line=dict(color=COLOR_NORMAL, width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x}<br>Value: %{y:.2f} " + unit + "<extra></extra>",
        )
    )

    # Highlight faults if fault_status column is present
    if "fault_status" in sensor_df.columns:
        faulty = sensor_df[sensor_df["fault_status"] != "NORMAL"]
        if not faulty.empty:
            fig.add_trace(
                go.Scatter(
                    x=faulty["timestamp"],
                    y=faulty["reading_value"],
                    mode="markers",
                    name="Fault Events",
                    marker=dict(
                        color=COLOR_HIGH_FAULT,
                        size=10,
                        symbol="triangle-up",
                        line=dict(width=1, color="black"),
                    ),
                    hovertemplate="Fault: %{text}<br>Value: %{y:.2f} " + unit + "<extra></extra>",
                    text=faulty["fault_status"],
                )
            )

    # Add Upper Operating Limit line
    if upper_threshold is not None:
        fig.add_hline(
            y=upper_threshold,
            line_dash="dash",
            line_color=COLOR_UCL,
            line_width=1.5,
            annotation_text=f"Upper Threshold: {upper_threshold} {unit}",
            annotation_position="top right",
        )

    # Add Lower Operating Limit line
    if lower_threshold is not None:
        fig.add_hline(
            y=lower_threshold,
            line_dash="dash",
            line_color=COLOR_LOW_FAULT,
            line_width=1.5,
            annotation_text=f"Lower Threshold: {lower_threshold} {unit}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text=chart_title, x=0.05),
        xaxis_title="Timestamp",
        yaxis_title=f"Measurement ({unit})",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_multi_sensor_overview_chart(df: pd.DataFrame) -> go.Figure:
    """Generate a 3-row synchronized time-series subplot for Temperature, Pressure, and Level.

    Args:
        df: Multi-sensor telemetry DataFrame.

    Returns:
        plotly.graph_objects.Figure
    """
    if df.empty or "sensor_id" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No telemetry data to display")
        return fig

    # Identify sensors
    temp_df = df[df["measurement_type"] == "temperature"]
    pres_df = df[df["measurement_type"] == "pressure"]
    lvl_df = df[df["measurement_type"] == "level"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Temperature Sensor (°C)",
            "Pressure Sensor (kPa)",
            "Liquid Level Sensor (%)",
        ),
    )

    if not temp_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(temp_df["timestamp"]),
                y=temp_df["reading_value"],
                mode="lines",
                name="Temperature",
                line=dict(color="#d62728", width=1.8),
                hovertemplate="%{x}<br>%{y:.2f} °C<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not pres_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(pres_df["timestamp"]),
                y=pres_df["reading_value"],
                mode="lines",
                name="Pressure",
                line=dict(color="#1f77b4", width=1.8),
                hovertemplate="%{x}<br>%{y:.2f} kPa<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if not lvl_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(lvl_df["timestamp"]),
                y=lvl_df["reading_value"],
                mode="lines",
                name="Liquid Level",
                line=dict(color="#2ca02c", width=1.8),
                hovertemplate="%{x}<br>%{y:.2f} %<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=650,
        template="plotly_white",
        title_text="Multi-Channel Plant Telemetry Overview",
        showlegend=False,
        margin=dict(l=50, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kPa", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)

    return fig


def create_spc_control_chart(
    annotated_spc_df: pd.DataFrame,
    sensor_id: str,
    unit: str = "",
) -> go.Figure:
    """Generate a Shewhart Individuals (X-chart) with control and warning limits.

    Args:
        annotated_spc_df: DataFrame returned by `spc_analyzer.annotate_spc_dataframe`.
        sensor_id: Sensor ID.
        unit: Measurement unit string.

    Returns:
        plotly.graph_objects.Figure
    """
    if annotated_spc_df.empty or "reading_value" not in annotated_spc_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available for SPC Control Chart")
        return fig

    fig = go.Figure()

    timestamps = pd.to_datetime(annotated_spc_df["timestamp"])
    readings = annotated_spc_df["reading_value"]

    # Base process measurements
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=readings,
            mode="lines+markers",
            name="Measured Reading",
            line=dict(color=COLOR_NORMAL, width=1.5),
            marker=dict(size=4),
            hovertemplate="%{x}<br>Value: %{y:.2f} " + unit + "<extra></extra>",
        )
    )

    cl = float(annotated_spc_df["center_line"].iloc[0])
    ucl = float(annotated_spc_df["ucl"].iloc[0])
    lcl = float(annotated_spc_df["lcl"].iloc[0])
    uwl = float(annotated_spc_df["uwl"].iloc[0])
    lwl = float(annotated_spc_df["lwl"].iloc[0])

    # Center Line (Mean)
    fig.add_hline(
        y=cl,
        line_color=COLOR_CL,
        line_width=2,
        annotation_text=f"CL (Mean): {cl:.2f} {unit}",
        annotation_position="top left",
    )

    # Upper Control Limit (UCL = Mean + 3sigma)
    fig.add_hline(
        y=ucl,
        line_dash="dash",
        line_color=COLOR_UCL,
        line_width=2,
        annotation_text=f"UCL (+3σ): {ucl:.2f} {unit}",
        annotation_position="top right",
    )

    # Lower Control Limit (LCL = Mean - 3sigma)
    fig.add_hline(
        y=lcl,
        line_dash="dash",
        line_color=COLOR_LCL,
        line_width=2,
        annotation_text=f"LCL (-3σ): {lcl:.2f} {unit}",
        annotation_position="bottom right",
    )

    # Upper Warning Limit (UWL = Mean + 2sigma)
    fig.add_hline(
        y=uwl,
        line_dash="dot",
        line_color=COLOR_WARNING,
        line_width=1,
        annotation_text=f"UWL (+2σ): {uwl:.2f}",
        annotation_position="top right",
    )

    # Lower Warning Limit (LWL = Mean - 2sigma)
    fig.add_hline(
        y=lwl,
        line_dash="dot",
        line_color=COLOR_WARNING,
        line_width=1,
        annotation_text=f"LWL (-2σ): {lwl:.2f}",
        annotation_position="bottom right",
    )

    # Highlight Out-of-Control Points
    if "is_ooc" in annotated_spc_df.columns:
        ooc_points = annotated_spc_df[annotated_spc_df["is_ooc"]]
        if not ooc_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(ooc_points["timestamp"]),
                    y=ooc_points["reading_value"],
                    mode="markers",
                    name="Out of Control",
                    marker=dict(
                        color="#d62728",
                        size=10,
                        symbol="circle-open",
                        line=dict(width=2.5, color="#d62728"),
                    ),
                    text=ooc_points["spc_rule_violated"],
                    hovertemplate="OOC Violation: %{text}<br>Value: %{y:.2f} " + unit + "<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Shewhart Individuals Control Chart (X-Chart) - {sensor_id}",
            x=0.05,
        ),
        xaxis_title="Timestamp",
        yaxis_title=f"Measured Reading ({unit})",
        template="plotly_white",
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_anomaly_scatter_chart(
    df: pd.DataFrame,
    sensor_id: str,
    threshold: float = 3.0,
) -> go.Figure:
    """Generate a Z-score statistical dispersion chart highlighting anomalies.

    Args:
        df: DataFrame with ['timestamp', 'sensor_id', 'z_score', 'is_anomaly'].
        sensor_id: Target sensor.
        threshold: Z-score threshold.

    Returns:
        plotly.graph_objects.Figure
    """
    sensor_df = df[df["sensor_id"] == sensor_id].copy()
    if sensor_df.empty or "z_score" not in sensor_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No anomaly data available for selected sensor")
        return fig

    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    sensor_df = sensor_df.sort_values(by="timestamp").reset_index(drop=True)

    fig = go.Figure()

    # Normal Z-score points
    normal_pts = sensor_df[~sensor_df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=normal_pts["timestamp"],
            y=normal_pts["z_score"],
            mode="markers",
            name="Normal (|Z| ≤ threshold)",
            marker=dict(color="#1f77b4", size=6, opacity=0.7),
            hovertemplate="Time: %{x}<br>Z-Score: %{y:.2f}<extra></extra>",
        )
    )

    # Anomaly Z-score points
    anomaly_pts = sensor_df[sensor_df["is_anomaly"]]
    if not anomaly_pts.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_pts["timestamp"],
                y=anomaly_pts["z_score"],
                mode="markers",
                name=f"Anomaly (|Z| > {threshold})",
                marker=dict(
                    color="#d62728",
                    size=11,
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                ),
                text=anomaly_pts.get("anomaly_reason", ""),
                hovertemplate="Time: %{x}<br>Z-Score: %{y:.2f}<br>%{text}<extra></extra>",
            )
        )

    # Threshold horizontal reference lines
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#d62728",
        line_width=1.5,
        annotation_text=f"+{threshold}σ Cutoff",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-threshold,
        line_dash="dash",
        line_color="#d62728",
        line_width=1.5,
        annotation_text=f"-{threshold}σ Cutoff",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=0.0,
        line_color="#2ca02c",
        line_width=1,
        annotation_text="Mean (Z = 0)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=f"Statistical Anomaly Z-Score Profile - {sensor_id}",
            x=0.05,
        ),
        xaxis_title="Timestamp",
        yaxis_title="Standardized Z-Score (σ)",
        template="plotly_white",
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_distribution_histogram(
    df: pd.DataFrame,
    sensor_id: str,
    unit: str = "",
) -> go.Figure:
    """Generate an empirical distribution histogram with Gaussian curve overlay.

    Args:
        df: Telemetry DataFrame.
        sensor_id: Sensor identifier.
        unit: Measurement unit.

    Returns:
        plotly.graph_objects.Figure
    """
    sensor_df = df[df["sensor_id"] == sensor_id]
    if sensor_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to plot distribution")
        return fig

    vals = pd.to_numeric(sensor_df["reading_value"], errors="coerce").dropna().to_numpy()
    if len(vals) < 2:
        fig = go.Figure()
        fig.update_layout(title="Insufficient numeric data for histogram")
        return fig

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1))

    fig = go.Figure()

    # Histogram trace
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=25,
            name="Sample Frequency",
            marker_color="#1f77b4",
            opacity=0.7,
        )
    )

    # Reference mean vertical line
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#2ca02c",
        line_width=2,
        annotation_text=f"Mean: {mean_val:.2f} {unit}",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=f"Empirical Frequency Distribution - {sensor_id} (μ={mean_val:.2f}, σ={std_val:.2f})",
            x=0.05,
        ),
        xaxis_title=f"Measurement ({unit})",
        yaxis_title="Count / Frequency",
        template="plotly_white",
        margin=dict(l=50, r=40, t=50, b=40),
        bargap=0.05,
    )

    return fig


def create_fault_distribution_pie(health_status: Dict[str, Any]) -> go.Figure:
    """Generate a summary donut chart illustrating the operational health breakdown.

    Args:
        health_status: Dictionary output from `fault_detector.evaluate_system_health`.

    Returns:
        plotly.graph_objects.Figure
    """
    total = health_status.get("total_readings", 0)
    fault_count = health_status.get("fault_count", 0)
    normal_count = max(total - fault_count, 0)

    if total == 0:
        fig = go.Figure()
        fig.update_layout(title="No operational data available")
        return fig

    labels = ["Normal Readings", "Faulty Readings"]
    values = [normal_count, fault_count]
    colors = [COLOR_NORMAL, COLOR_HIGH_FAULT]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} readings (%{percent})<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text="System Health & Integrity Distribution", x=0.05),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig
