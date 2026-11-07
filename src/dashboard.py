"""Streamlit Dashboard UI Presentation Module.

Renders the multi-page industrial instrumentation dashboard:
1. System Overview
2. Live Sensor Monitoring
3. Fault Detection
4. Anomaly Analysis
5. Statistical Process Control
6. Data Analysis
7. Historical Data
8. CSV Import and Export
9. System Information

Pure Streamlit + Plotly implementation without external paid APIs, HTML/CSS hacks,
or client-side scripts.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detector import AnomalyDetector
from src.charts import (
    create_anomaly_scatter_chart,
    create_distribution_histogram,
    create_fault_distribution_pie,
    create_multi_sensor_overview_chart,
    create_sensor_trend_chart,
    create_spc_control_chart,
)
from src.data_analyzer import DataAnalyzer
from src.data_logger import DataLogger
from src.data_validator import DataValidator
from src.database_manager import DatabaseManager
from src.fault_detector import (
    DEFAULT_OPERATING_LIMITS,
    STATUS_HIGH_FAULT,
    STATUS_INVALID,
    STATUS_LOW_FAULT,
    STATUS_NORMAL,
    FaultDetector,
)
from src.sensor_simulator import FAULT_MODES, SensorSimulator
from src.spc_analyzer import SPCAnalyzer


def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = SensorSimulator(seed=42)

    if "fault_detector" not in st.session_state:
        st.session_state.fault_detector = FaultDetector()

    if "anomaly_detector" not in st.session_state:
        st.session_state.anomaly_detector = AnomalyDetector(threshold=3.0)

    if "spc_analyzer" not in st.session_state:
        st.session_state.spc_analyzer = SPCAnalyzer(sigma_multiplier=3.0)

    if "data_analyzer" not in st.session_state:
        st.session_state.data_analyzer = DataAnalyzer()

    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    if "data_logger" not in st.session_state:
        st.session_state.data_logger = DataLogger(db_manager=st.session_state.db_manager)

    # Initial telemetry generation if not already generated
    if "telemetry_df" not in st.session_state:
        raw_df = st.session_state.simulator.generate_time_series(
            count=100, interval_seconds=5, fault_type="None (Normal)", seed=42
        )
        pipeline_res = st.session_state.data_logger.process_and_persist_batch(
            raw_df, st.session_state.fault_detector, st.session_state.anomaly_detector
        )
        st.session_state.telemetry_df = pipeline_res["annotated_df"]
        st.session_state.health_status = pipeline_res["health_status"]
        st.session_state.anomaly_metrics = pipeline_res["anomaly_metrics"]

    if "live_readings" not in st.session_state:
        st.session_state.live_readings = []


def run_pipeline_on_current_data() -> None:
    """Re-evaluate the current telemetry DataFrame through detection pipelines."""
    raw_df = st.session_state.telemetry_df[
        ["timestamp", "sensor_id", "measurement_type", "reading_value", "unit", "is_simulated", "injected_fault"]
    ].copy()
    fault_annotated = st.session_state.fault_detector.analyze_dataframe(raw_df)
    anomaly_annotated = st.session_state.anomaly_detector.detect_anomalies(fault_annotated)

    st.session_state.telemetry_df = anomaly_annotated
    st.session_state.health_status = st.session_state.fault_detector.evaluate_system_health(fault_annotated)
    st.session_state.anomaly_metrics = st.session_state.anomaly_detector.compute_anomaly_metrics(anomaly_annotated)


def render_sidebar() -> str:
    """Render the sidebar navigation and simulation configuration controls."""
    st.sidebar.title("Industrial Monitor")
    st.sidebar.caption("Educational Engineering Prototype v1.0")

    pages = [
        "1. System Overview",
        "2. Live Sensor Monitoring",
        "3. Fault Detection",
        "4. Anomaly Analysis",
        "5. Statistical Process Control",
        "6. Data Analysis",
        "7. Historical Data",
        "8. CSV Import and Export",
        "9. System Information",
    ]
    selected_page = st.sidebar.radio("Navigation Menu", pages)

    st.sidebar.divider()
    st.sidebar.subheader("Simulation Control Panel")

    with st.sidebar.expander("Telemetry Generation", expanded=False):
        sample_count = st.slider("Sample Count per Sensor", min_value=30, max_value=500, value=100, step=10)
        interval_sec = st.slider("Sampling Interval (s)", min_value=1, max_value=30, value=5, step=1)
        fault_mode = st.selectbox("Fault Mode to Inject", FAULT_MODES, index=0)
        fault_rate = st.slider("Fault Rate Probability", min_value=0.01, max_value=0.40, value=0.10, step=0.01)
        seed_val = st.number_input("Random Seed", value=42, step=1)

        if st.button("Generate & Process Telemetry", use_container_width=True):
            st.session_state.simulator.set_seed(int(seed_val))
            raw_df = st.session_state.simulator.generate_time_series(
                count=sample_count,
                interval_seconds=interval_sec,
                fault_type=fault_mode,
                fault_rate=fault_rate,
                seed=int(seed_val),
            )
            pipeline_res = st.session_state.data_logger.process_and_persist_batch(
                raw_df, st.session_state.fault_detector, st.session_state.anomaly_detector
            )
            st.session_state.telemetry_df = pipeline_res["annotated_df"]
            st.session_state.health_status = pipeline_res["health_status"]
            st.session_state.anomaly_metrics = pipeline_res["anomaly_metrics"]
            st.success(f"Generated {len(raw_df)} readings & saved to database!")
            st.rerun()

    with st.sidebar.expander("Operating Thresholds", expanded=False):
        st.caption("Adjust engineering threshold limits:")
        for sid, lim in st.session_state.fault_detector.limits.items():
            st.markdown(f"**{sid} ({lim.unit})**")
            col_l, col_u = st.columns(2)
            with col_l:
                new_lower = st.number_input(f"{sid} Lower", value=float(lim.lower_limit), key=f"lim_low_{sid}")
            with col_u:
                new_upper = st.number_input(f"{sid} Upper", value=float(lim.upper_limit), key=f"lim_upp_{sid}")

            if new_lower != lim.lower_limit or new_upper != lim.upper_limit:
                if new_lower < new_upper:
                    st.session_state.fault_detector.set_sensor_limits(
                        sensor_id=sid, lower_limit=new_lower, upper_limit=new_upper
                    )
                else:
                    st.error("Lower limit must be < upper limit")

        if st.button("Apply Thresholds", use_container_width=True):
            run_pipeline_on_current_data()
            st.success("Thresholds applied!")
            st.rerun()

    with st.sidebar.expander("Anomaly Cutoff", expanded=False):
        z_thresh = st.slider(
            "Z-Score Cutoff (|Z|)",
            min_value=1.5,
            max_value=4.0,
            value=float(st.session_state.anomaly_detector.threshold),
            step=0.1,
        )
        if z_thresh != st.session_state.anomaly_detector.threshold:
            st.session_state.anomaly_detector.set_threshold(z_thresh)
            run_pipeline_on_current_data()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.info(
        "Simulation Mode Active.\n"
        "Data is mathematically simulated for instrumentation training."
    )
    return selected_page


def page_system_overview() -> None:
    """Render Page 1: System Overview."""
    st.header("1. Industrial System Overview")
    st.caption("Plant-wide telemetry status and real-time instrumentation summary")

    st.warning(
        "SIMULATED DATA (Educational Prototype) — Readings generated using stochastic processes; "
        "not connected to physical field transmitters."
    )

    df = st.session_state.telemetry_df
    health = st.session_state.health_status
    anom_metrics = st.session_state.anomaly_metrics

    # Top KPI Metrics
    total_readings = health.get("total_readings", 0)
    fault_count = health.get("fault_count", 0)
    fault_rate = health.get("fault_percentage", 0.0)
    anomaly_count = anom_metrics.get("anomaly_count", 0)
    sys_status = health.get("overall_status", "NORMAL")

    normal_count = max(total_readings - fault_count, 0)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Readings", f"{total_readings:,}")
    m2.metric("Active Sensors", "3")
    m3.metric("Normal Readings", f"{normal_count:,}")
    m4.metric("Fault Events", f"{fault_count:,}", delta=f"{fault_rate:.1f}%", delta_color="inverse")
    m5.metric("Anomalies (|Z|>3)", f"{anomaly_count:,}")

    status_color = "normal"
    if sys_status == "NORMAL":
        m6.metric("System Health", "NORMAL")
    elif sys_status == "WARNING":
        m6.metric("System Health", "WARNING", delta="Attention", delta_color="off")
    else:
        m6.metric("System Health", "CRITICAL", delta="Action Required", delta_color="inverse")

    st.divider()

    col_chart, col_donut = st.columns([2, 1])
    with col_chart:
        st.subheader("Multi-Channel Plant Telemetry")
        overview_fig = create_multi_sensor_overview_chart(df)
        st.plotly_chart(overview_fig, use_container_width=True)

    with col_donut:
        st.subheader("Integrity Distribution")
        donut_fig = create_fault_distribution_pie(health)
        st.plotly_chart(donut_fig, use_container_width=True)

    st.subheader("Sensor Channel Health Cards")
    sensor_status_dict = health.get("sensor_status", {})

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    sensor_ids = ["TEMP_001", "PRES_001", "LVL_001"]
    sensor_names = {
        "TEMP_001": ("Temperature Transmitter", "°C"),
        "PRES_001": ("Pressure Transmitter", "kPa"),
        "LVL_001": ("Liquid Level Transmitter", "%"),
    }

    for i, sid in enumerate(sensor_ids):
        s_info = sensor_status_dict.get(sid, {})
        name, unit = sensor_names[sid]
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{sid}** — *{name}*")
                lat_val = s_info.get("latest_value")
                val_str = f"{lat_val:.2f} {unit}" if lat_val is not None and pd.notna(lat_val) else "N/A"
                st.markdown(f"### {val_str}")

                st_status = s_info.get("latest_status", "NORMAL")
                if st_status == STATUS_NORMAL:
                    st.success(f"Channel Status: {st_status}")
                elif "FAULT" in st_status:
                    st.error(f"Channel Status: {st_status}")
                else:
                    st.warning(f"Channel Status: {st_status}")

                f_pct = s_info.get("fault_percentage", 0.0)
                st.write(f"Faults: **{s_info.get('faults', 0)}** ({f_pct}%)")
                st.write(f"High: {s_info.get('high_faults', 0)} | Low: {s_info.get('low_faults', 0)}")


def page_live_monitoring() -> None:
    """Render Page 2: Live Sensor Monitoring."""
    st.header("2. Live Sensor Telemetry Stream")
    st.caption("Real-time loop acquisition simulation with instant threshold checking")

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        tick_fault_mode = st.selectbox(
            "Fault Injection for Next Tick",
            ["None", "High Spike", "Low Drop", "Missing Data", "Invalid Reading"],
        )
        if st.button("Acquire Telemetry Tick (Sample)", use_container_width=True):
            now_ts = datetime.datetime.now(datetime.timezone.utc)
            new_records = []
            for sid in ["TEMP_001", "PRES_001", "LVL_001"]:
                rec = st.session_state.simulator.generate_single_reading(
                    sensor_id=sid, timestamp=now_ts, fault_type=tick_fault_mode
                )
                new_records.append(rec)

            new_df = pd.DataFrame(new_records)
            st.session_state.data_logger.process_and_persist_batch(
                new_df, st.session_state.fault_detector, st.session_state.anomaly_detector
            )

            # Prepend or append to live readings list
            st.session_state.live_readings.extend(new_records)
            if len(st.session_state.live_readings) > 150:
                st.session_state.live_readings = st.session_state.live_readings[-150:]
            st.success("New telemetry packet acquired and logged to SQLite!")

    with col_btn2:
        st.info(
            "Click **Acquire Telemetry Tick** to simulate receiving a new Modbus/HART/OPC-UA "
            "telemetry frame from field transmitters. Each reading is evaluated against operating thresholds "
            "and immediately saved to the historical database."
        )

    st.divider()

    # Gauge cards for latest readings
    df = st.session_state.telemetry_df
    k1, k2, k3 = st.columns(3)

    for i, (sid, col, name, unit) in enumerate(
        [
            ("TEMP_001", k1, "Temperature Sensor", "°C"),
            ("PRES_001", k2, "Pressure Sensor", "kPa"),
            ("LVL_001", k3, "Liquid Level Sensor", "%"),
        ]
    ):
        s_data = df[df["sensor_id"] == sid]
        with col:
            with st.container(border=True):
                st.markdown(f"**{name} ({sid})**")
                if not s_data.empty:
                    latest = s_data.iloc[-1]
                    prev_val = s_data.iloc[-2]["reading_value"] if len(s_data) > 1 else None
                    cur_val = latest["reading_value"]

                    delta_val = None
                    if cur_val is not None and prev_val is not None and pd.notna(cur_val) and pd.notna(prev_val):
                        delta_val = round(cur_val - prev_val, 2)

                    disp_val = f"{cur_val:.2f} {unit}" if pd.notna(cur_val) else "DROPOUT (NaN)"
                    st.metric("Latest Reading", disp_val, delta=delta_val)

                    lim = st.session_state.fault_detector.limits[sid]
                    st.caption(f"Limits: {lim.lower_limit} - {lim.upper_limit} {unit}")

                    c_status = latest.get("fault_status", "NORMAL")
                    if c_status == STATUS_NORMAL:
                        st.success("Status: NORMAL")
                    elif "FAULT" in c_status:
                        st.error(f"Status: {c_status}")
                    else:
                        st.warning(f"Status: {c_status}")
                else:
                    st.write("No telemetry recorded")

    st.subheader("Rolling Process Trend (Selected Channel)")
    selected_sid = st.selectbox("Select Sensor Channel to Inspect", ["TEMP_001", "PRES_001", "LVL_001"], key="live_sid")
    lim = st.session_state.fault_detector.limits[selected_sid]
    chart = create_sensor_trend_chart(
        df,
        selected_sid,
        upper_threshold=lim.upper_limit,
        lower_threshold=lim.lower_limit,
    )
    st.plotly_chart(chart, use_container_width=True)


def page_fault_detection() -> None:
    """Render Page 3: Fault Detection."""
    st.header("3. Automatic Sensor Fault Detection")
    st.caption("Configurable Upper/Lower Threshold violations & Physical Feasibility Checks")

    df = st.session_state.telemetry_df

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        s_filter = st.multiselect(
            "Filter by Sensor ID",
            options=["TEMP_001", "PRES_001", "LVL_001"],
            default=["TEMP_001", "PRES_001", "LVL_001"],
        )
    with col_filter2:
        status_filter = st.multiselect(
            "Filter by Fault Status",
            options=[STATUS_NORMAL, STATUS_HIGH_FAULT, STATUS_LOW_FAULT, STATUS_INVALID],
            default=[STATUS_NORMAL, STATUS_HIGH_FAULT, STATUS_LOW_FAULT, STATUS_INVALID],
        )

    filtered_df = df[
        (df["sensor_id"].isin(s_filter)) & (df["fault_status"].isin(status_filter))
    ]

    # Metrics
    total_f = len(filtered_df)
    fault_count = int(filtered_df["is_fault"].sum()) if not filtered_df.empty else 0
    high_count = int((filtered_df["fault_status"] == STATUS_HIGH_FAULT).sum()) if not filtered_df.empty else 0
    low_count = int((filtered_df["fault_status"] == STATUS_LOW_FAULT).sum()) if not filtered_df.empty else 0
    inv_count = int((filtered_df["fault_status"] == STATUS_INVALID).sum()) if not filtered_df.empty else 0

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Filtered Records", total_f)
    f2.metric("Total Faults", fault_count)
    f3.metric("High Faults", high_count)
    f4.metric("Low Faults", low_count)
    f5.metric("Invalid Readings", inv_count)

    st.divider()

    # Visual trend with threshold markers for a chosen sensor
    chosen_sensor = st.selectbox("View Sensor Threshold Graph", s_filter if s_filter else ["TEMP_001"])
    lim = st.session_state.fault_detector.limits[chosen_sensor]
    fig = create_sensor_trend_chart(
        df,
        chosen_sensor,
        upper_threshold=lim.upper_limit,
        lower_threshold=lim.lower_limit,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fault Event Log & Diagnostic Table")
    faults_only = df[df["fault_status"] != STATUS_NORMAL]
    if faults_only.empty:
        st.success("No fault conditions detected in current telemetry sequence.")
    else:
        display_cols = [
            "timestamp",
            "sensor_id",
            "measurement_type",
            "reading_value",
            "unit",
            "fault_status",
            "fault_description",
            "threshold_violated",
            "injected_fault",
        ]
        valid_cols = [c for c in display_cols if c in faults_only.columns]
        st.dataframe(faults_only[valid_cols], use_container_width=True)


def page_anomaly_analysis() -> None:
    """Render Page 4: Statistical Anomaly Analysis."""
    st.header("4. Statistical Anomaly Detection (Z-Score)")
    st.caption("Standardized dispersion analysis: Z = (X - μ) / σ")

    st.info(
        "METHODOLOGY NOTE: Statistical anomaly detection is a classical parametric mathematical model "
        "evaluating deviation from the sample mean. It does NOT employ Machine Learning (ML) or Artificial "
        "Intelligence (AI) models."
    )

    df = st.session_state.telemetry_df
    anom_det = st.session_state.anomaly_detector

    col_sid, col_thresh = st.columns(2)
    with col_sid:
        selected_sensor = st.selectbox(
            "Select Sensor Channel to Inspect",
            ["TEMP_001", "PRES_001", "LVL_001"],
            key="anom_sensor_sel",
        )
    with col_thresh:
        st.write(f"Active Z-Score Threshold: **|Z| > {anom_det.threshold:.1f}σ**")

    sensor_data = df[df["sensor_id"] == selected_sensor]
    if sensor_data.empty:
        st.warning("No data available for this sensor.")
        return

    # Metrics for selected sensor
    s_total = len(sensor_data)
    s_anom = int(sensor_data["is_anomaly"].sum())
    s_rate = (s_anom / s_total) * 100.0 if s_total > 0 else 0.0

    valid_vals = pd.to_numeric(sensor_data["reading_value"], errors="coerce").dropna()
    mean_val = float(np.mean(valid_vals)) if not valid_vals.empty else np.nan
    std_val = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else np.nan

    z_scores_abs = sensor_data["z_score"].dropna().abs()
    max_z = float(z_scores_abs.max()) if not z_scores_abs.empty else 0.0

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Sample Count", s_total)
    a2.metric("Sample Mean (μ)", f"{mean_val:.2f}")
    a3.metric("Std Dev (σ)", f"{std_val:.2f}")
    a4.metric("Anomalies Found", f"{s_anom} ({s_rate:.1f}%)")
    a5.metric("Peak |Z|-Score", f"{max_z:.2f}σ")

    st.divider()

    st.subheader(f"Standardized Z-Score Scatter Profile ({selected_sensor})")
    anom_fig = create_anomaly_scatter_chart(df, selected_sensor, threshold=anom_det.threshold)
    st.plotly_chart(anom_fig, use_container_width=True)

    st.subheader("Statistical Outlier Event Records")
    anom_only = sensor_data[sensor_data["is_anomaly"]]
    if anom_only.empty:
        st.success(f"No statistical anomalies detected beyond |Z| > {anom_det.threshold} for {selected_sensor}.")
    else:
        disp_cols = [
            "timestamp",
            "sensor_id",
            "reading_value",
            "mean",
            "std_dev",
            "z_score",
            "anomaly_reason",
            "injected_fault",
        ]
        valid_cols = [c for c in disp_cols if c in anom_only.columns]
        st.dataframe(anom_only[valid_cols], use_container_width=True)


def page_spc_analysis() -> None:
    """Render Page 5: Statistical Process Control (SPC)."""
    st.header("5. Statistical Process Control (SPC)")
    st.caption("Shewhart Individuals Control Chart (X-Chart) & Process Capability Analysis")

    st.warning(
        "EDUCATIONAL DISCLAIMER: Initial control limits are dynamically estimated from the current sample set. "
        "In industrial manufacturing (Phase II SPC), baseline control limits are established from a stable historical "
        "reference period and locked prior to monitoring production."
    )

    df = st.session_state.telemetry_df
    spc = st.session_state.spc_analyzer

    selected_sid = st.selectbox("Select Sensor for SPC Analysis", ["TEMP_001", "PRES_001", "LVL_001"], key="spc_sensor_sel")

    # Specification limits for Cp / Cpk evaluation
    lim = st.session_state.fault_detector.limits[selected_sid]
    with st.expander("Specification Limits for Process Capability (Cp / Cpk)", expanded=False):
        c_usl, c_lsl = st.columns(2)
        with c_usl:
            usl_val = st.number_input("Upper Specification Limit (USL)", value=float(lim.upper_limit), key=f"spc_usl_{selected_sid}")
        with c_lsl:
            lsl_val = st.number_input("Lower Specification Limit (LSL)", value=float(lim.lower_limit), key=f"spc_lsl_{selected_sid}")

    spc_summary = spc.evaluate_sensor_spc(df, selected_sid, usl=usl_val, lsl=lsl_val)
    if spc_summary is None:
        st.error("Insufficient sample points to perform SPC analysis.")
        return

    lims = spc_summary.limits
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Center Line (μ)", f"{lims.center_line:.2f} {spc_summary.unit}")
    s2.metric("UCL (+3σ)", f"{lims.ucl:.2f}")
    s3.metric("LCL (-3σ)", f"{lims.lcl:.2f}")
    s4.metric("OOC Violations", f"{spc_summary.out_of_control_count} ({spc_summary.out_of_control_pct}%)")

    cp_disp = f"{spc_summary.cp:.2f}" if spc_summary.cp is not None else "N/A"
    cpk_disp = f"{spc_summary.cpk:.2f}" if spc_summary.cpk is not None else "N/A"
    s5.metric("Cp Index", cp_disp)
    s6.metric("Cpk Index", cpk_disp)

    # Process status banner
    if spc_summary.process_status == "IN_CONTROL":
        st.success("Process Status: IN STATISTICAL CONTROL (All sample points within 3-sigma boundaries)")
    else:
        st.error(
            f"Process Status: OUT OF CONTROL — {spc_summary.out_of_control_count} violation(s) identified!"
        )

    st.divider()

    annotated_spc = spc.annotate_spc_dataframe(df, selected_sid)
    spc_fig = create_spc_control_chart(annotated_spc, selected_sid, unit=spc_summary.unit)
    st.plotly_chart(spc_fig, use_container_width=True)

    st.subheader("SPC Rule Violations Log")
    if not spc_summary.violations:
        st.info("No out-of-control rules violated.")
    else:
        v_df = pd.DataFrame(spc_summary.violations)
        st.dataframe(v_df[["timestamp", "value", "rule", "severity", "limit_violated"]], use_container_width=True)


def page_data_analysis() -> None:
    """Render Page 6: Data Analysis."""
    st.header("6. Industrial Data Analysis & Descriptive Statistics")
    st.caption("Parametric central tendency, sample dispersion, and correlation analysis")

    df = st.session_state.telemetry_df
    analyzer = st.session_state.data_analyzer

    st.subheader("Multi-Channel Statistical Summary Table")
    summary_table = analyzer.generate_sensor_summary_table(df)
    st.dataframe(summary_table, use_container_width=True)

    st.divider()

    st.subheader("Empirical Distribution & Normality Check")
    selected_sid = st.selectbox("Select Sensor for Distribution Plot", ["TEMP_001", "PRES_001", "LVL_001"], key="dist_sid")
    unit = str(df[df["sensor_id"] == selected_sid]["unit"].iloc[0]) if not df[df["sensor_id"] == selected_sid].empty else ""
    hist_fig = create_distribution_histogram(df, selected_sid, unit=unit)
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Telemetry Correlation Matrix")
    pivoted = df.pivot_table(index="timestamp", columns="sensor_id", values="reading_value", aggfunc="first")
    if not pivoted.empty and len(pivoted.columns) > 1:
        corr = pivoted.corr()
        st.dataframe(corr.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.write("Insufficient data columns to calculate correlation matrix.")


def page_historical_data() -> None:
    """Render Page 7: Historical Data (SQLite Storage)."""
    st.header("7. Historical Data Storage (SQLite)")
    st.caption("Relational telemetry persistence, indexed queries, and audit logs")

    db = st.session_state.db_manager
    stats = db.get_database_statistics()

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Stored Readings", f"{stats['total_readings']:,}")
    h2.metric("Recorded Faults", f"{stats['total_faults']:,}")
    h3.metric("Recorded Anomalies", f"{stats['total_anomalies']:,}")
    h4.metric("Database Mode", "SQLite (WAL)")

    st.caption(f"Database File: `{stats['database_path']}`")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Historical Readings", "Fault Records", "Anomaly Records"])

    with tab1:
        st.subheader("Query Sensor Readings")
        limit_rows = st.select_slider("Record Limit", options=[25, 50, 100, 250, 500, 1000], value=100)
        hist_readings = db.query_readings(limit=limit_rows)
        if hist_readings.empty:
            st.info("No readings found in database.")
        else:
            st.dataframe(hist_readings, use_container_width=True)

    with tab2:
        st.subheader("Historical Fault Event Log")
        hist_faults = db.query_faults(limit=100)
        if hist_faults.empty:
            st.info("No fault records logged in database.")
        else:
            st.dataframe(hist_faults, use_container_width=True)

    with tab3:
        st.subheader("Historical Anomaly Event Log")
        hist_anom = db.query_anomalies(limit=100)
        if hist_anom.empty:
            st.info("No anomaly records logged in database.")
        else:
            st.dataframe(hist_anom, use_container_width=True)

    st.divider()
    with st.expander("Database Maintenance", expanded=False):
        st.warning("Clearing database removes all persisted historical readings permanently.")
        if st.button("Clear Historical Database", type="secondary"):
            db.clear_database()
            st.success("Historical database tables successfully cleared!")
            st.rerun()


def page_csv_import_export() -> None:
    """Render Page 8: CSV Import and Export."""
    st.header("8. CSV Telemetry Ingestion & Export")
    st.caption("Compliant schema validation, batch ingestion, and report generation")

    tab_import, tab_export = st.tabs(["Import CSV Dataset", "Export Reports"])

    with tab_import:
        st.subheader("Upload Industrial Sensor Telemetry (CSV)")
        st.markdown(
            "Upload a comma-separated values (CSV) file containing telemetry. "
            "Required columns: `timestamp`, `sensor_id`, `measurement_type`, `reading_value`."
        )

        uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            is_valid, errors, cleaned_df = DataValidator.validate_csv(file_bytes)

            if is_valid:
                st.success(f"Validation Successful: Schema compliant with {len(cleaned_df)} valid records.")
                st.dataframe(cleaned_df.head(10), use_container_width=True)

                if st.button("Ingest Uploaded Dataset into System", use_container_width=True):
                    pipeline_res = st.session_state.data_logger.process_and_persist_batch(
                        cleaned_df,
                        st.session_state.fault_detector,
                        st.session_state.anomaly_detector,
                    )
                    st.session_state.telemetry_df = pipeline_res["annotated_df"]
                    st.session_state.health_status = pipeline_res["health_status"]
                    st.session_state.anomaly_metrics = pipeline_res["anomaly_metrics"]
                    st.success("Uploaded dataset processed and loaded as current telemetry!")
                    st.rerun()
            else:
                st.error("Validation Failed. Please review errors:")
                for err in errors:
                    st.write(f"- {err}")

    with tab_export:
        st.subheader("Download Formatted Reports")
        df = st.session_state.telemetry_df
        analyzer = st.session_state.data_analyzer

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.markdown("#### Full Processed Telemetry")
            st.caption("Includes timestamps, readings, fault flags, and Z-scores.")
            full_csv = DataValidator.export_to_csv(df)
            st.download_button(
                label="Download Full Telemetry (CSV)",
                data=full_csv,
                file_name="processed_sensor_telemetry.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("#### Statistical Analysis Summary")
            st.caption("Descriptive statistics across all sensor channels.")
            summary_df = analyzer.generate_sensor_summary_table(df)
            summary_csv = DataValidator.export_to_csv(summary_df)
            st.download_button(
                label="Download Statistical Summary (CSV)",
                data=summary_csv,
                file_name="sensor_statistical_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c_exp2:
            st.markdown("#### Fault Event Report")
            st.caption("Filtered list of all threshold and feasibility violations.")
            fault_summary = st.session_state.fault_detector.get_fault_summary(df)
            fault_csv = DataValidator.export_to_csv(fault_summary)
            st.download_button(
                label="Download Fault Report (CSV)",
                data=fault_csv,
                file_name="sensor_fault_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("#### Anomaly Event Report")
            st.caption("Filtered list of all points exceeding Z-score threshold.")
            anomaly_summary = st.session_state.anomaly_detector.get_anomaly_summary(df)
            anom_csv = DataValidator.export_to_csv(anomaly_summary)
            st.download_button(
                label="Download Anomaly Report (CSV)",
                data=anom_csv,
                file_name="sensor_anomaly_report.csv",
                mime="text/csv",
                use_container_width=True,
            )


def page_system_information() -> None:
    """Render Page 9: System Information & Engineering Documentation."""
    st.header("9. Engineering Specifications & Instrumentation Architecture")
    st.caption("Technical reference documentation for educational and industrial evaluation")

    with st.expander("1. Sensor Instrumentation Specifications", expanded=True):
        st.markdown(
            """
            | Sensor Tag | Measurement | Physical Principle | Operating Range | Nominal Target | Standard Deviation | Default Limits |
            | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
            | **TEMP_001** | Temperature | RTD Pt100 (4-wire Bridge) | -50.0 to 300.0 °C | 70.0 °C | ±2.0 °C | 50.0 to 90.0 °C |
            | **PRES_001** | Pressure | Piezoresistive Diaphragm | 0.0 to 500.0 kPa | 150.0 kPa | ±4.0 kPa | 120.0 to 180.0 kPa |
            | **LVL_001** | Liquid Level | Hydrostatic Differential Pressure | 0.0 to 100.0 % | 55.0 % | ±3.0 % | 20.0 to 80.0 % |
            """
        )

    with st.expander("2. Fault Detection Methodology", expanded=True):
        st.markdown(
            """
            Threshold-based monitoring classifies each measurement sample deterministically:
            - **NORMAL**: Reading is within operating boundaries: $LOL \\le X \\le UOL$.
            - **HIGH_FAULT**: Reading exceeds upper operating limit: $X > UOL$.
            - **LOW_FAULT**: Reading drops below lower operating limit: $X < LOL$.
            - **INVALID_READING**: Missing reading ($NaN$), unparseable string, or physical impossibility ($X < Min_{phys}$ or $X > Max_{phys}$).
            """
        )

    with st.expander("3. Statistical Anomaly Detection (Z-Score)", expanded=True):
        st.markdown(
            """
            Univariate standardized anomaly score calculation:
            $$Z = \\frac{X - \\mu}{\\sigma}$$
            where:
            - $\\mu$ is the sample mean of the channel.
            - $\\sigma$ is the sample standard deviation ($ddof = 1$).
            - An outlier is flagged when $|Z| > Z_{cutoff}$ (Default $Z_{cutoff} = 3.0$, representing 99.73% confidence interval under normal distribution).
            
            *Note: This is a classical statistical dispersion technique, not an Artificial Intelligence or Machine Learning algorithm.*
            """
        )

    with st.expander("4. Statistical Process Control (SPC) Architecture", expanded=True):
        st.markdown(
            """
            Shewhart Individual Control Chart ($X$-chart) limits:
            - **Center Line (CL)**: $\\bar{X} = \\frac{1}{n} \\sum_{i=1}^{n} X_i$
            - **Upper Control Limit (UCL)**: $\\bar{X} + 3\\sigma$
            - **Lower Control Limit (LCL)**: $\\bar{X} - 3\\sigma$
            - **Warning Limits**: $\\bar{X} \\pm 2\\sigma$
            
            Process capability metrics:
            $$C_p = \\frac{USL - LSL}{6\\sigma}$$
            $$C_{pk} = \\min\\left(\\frac{USL - \\mu}{3\\sigma}, \\frac{\\mu - LSL}{3\\sigma}\\right)$$
            """
        )

    with st.expander("5. System Architecture & Relational Database Design", expanded=True):
        st.markdown(
            """
            - **Database Engine**: Embedded SQLite 3 with Write-Ahead Logging (WAL) enabled.
            - **Tables**:
                1. `sensor_readings`: Raw telemetry records with composite `UNIQUE(timestamp, sensor_id)` constraint.
                2. `fault_records`: Event-driven log of threshold breaches and invalid measurements.
                3. `anomaly_records`: Event-driven log of statistical $Z$-score outliers.
            - **Storage Path**: `data/sensor_monitoring.db` (auto-created via `pathlib.Path`).
            """
        )

    with st.expander("6. Academic & Professional Engineering Prototype Disclaimer", expanded=True):
        st.info(
            "This project is developed as an educational engineering prototype for industrial automation "
            "and semiconductor instrumentation portfolios. Telemetry data is mathematically generated using "
            "stochastic Gaussian and sinusoidal processes with simulated physical fault modes. The system does not "
            "connect to physical plant hardware."
        )


def render_dashboard() -> None:
    """Main dashboard entry dispatcher."""
    init_session_state()
    selected_page = render_sidebar()

    if selected_page == "1. System Overview":
        page_system_overview()
    elif selected_page == "2. Live Sensor Monitoring":
        page_live_monitoring()
    elif selected_page == "3. Fault Detection":
        page_fault_detection()
    elif selected_page == "4. Anomaly Analysis":
        page_anomaly_analysis()
    elif selected_page == "5. Statistical Process Control":
        page_spc_analysis()
    elif selected_page == "6. Data Analysis":
        page_data_analysis()
    elif selected_page == "7. Historical Data":
        page_historical_data()
    elif selected_page == "8. CSV Import and Export":
        page_csv_import_export()
    elif selected_page == "9. System Information":
        page_system_information()
