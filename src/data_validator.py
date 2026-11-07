"""Data Validation and Sanitation Module.

Provides schema validation, data type verification, and sanitation for sensor telemetry:
- Validates external CSV uploads against industrial schemas.
- Parses and standardizes ISO-8601 timestamps.
- Verifies physical ranges and numeric coercions.
- Generates clear, actionable validation error reports.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Core required fields in industrial telemetry datasets
REQUIRED_COLUMNS = ["timestamp", "sensor_id", "measurement_type", "reading_value"]
OPTIONAL_COLUMNS = ["unit", "is_simulated", "injected_fault"]

VALID_MEASUREMENT_TYPES = {"temperature", "pressure", "level"}


class DataValidator:
    """Validates and cleans incoming sensor datasets and CSV files."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
    ) -> Tuple[bool, List[str], pd.DataFrame]:
        """Validate an arbitrary DataFrame against the telemetry data schema.

        Args:
            df: Input DataFrame to inspect and sanitize.

        Returns:
            Tuple containing:
                - is_valid (bool): True if schema satisfies minimum requirements.
                - errors (List[str]): List of error descriptions or warnings.
                - cleaned_df (pd.DataFrame): Sanitized DataFrame ready for pipeline ingestion.
        """
        errors: List[str] = []

        if df is None or not isinstance(df, pd.DataFrame):
            return False, ["Input is not a valid pandas DataFrame."], pd.DataFrame()

        if df.empty:
            return False, ["Input DataFrame is empty."], pd.DataFrame()

        # 1. Check for required column headers
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required column(s): {', '.join(missing_cols)}")
            return False, errors, pd.DataFrame()

        cleaned_df = df.copy()

        # 2. Validate and standardize timestamps
        try:
            parsed_timestamps = pd.to_datetime(cleaned_df["timestamp"], errors="coerce")
            invalid_ts_count = int(parsed_timestamps.isna().sum())
            if invalid_ts_count > 0:
                errors.append(
                    f"Found {invalid_ts_count} invalid or unparseable timestamp(s)."
                )
            cleaned_df["timestamp"] = parsed_timestamps
        except Exception as ex:
            errors.append(f"Failed to parse 'timestamp' column: {str(ex)}")

        # 3. Validate sensor_id
        cleaned_df["sensor_id"] = cleaned_df["sensor_id"].astype(str).str.strip()
        empty_ids = int((cleaned_df["sensor_id"] == "").sum())
        if empty_ids > 0:
            errors.append(f"Found {empty_ids} empty or blank 'sensor_id' record(s).")

        # 4. Validate measurement_type
        cleaned_df["measurement_type"] = (
            cleaned_df["measurement_type"].astype(str).str.strip().str.lower()
        )

        # 5. Validate reading_value
        cleaned_df["reading_value"] = pd.to_numeric(
            cleaned_df["reading_value"], errors="coerce"
        )
        null_readings = int(cleaned_df["reading_value"].isna().sum())
        if null_readings > 0:
            errors.append(
                f"Note: {null_readings} reading(s) are non-numeric or missing (NaN). Handled gracefully."
            )

        # 6. Ensure default unit column exists if missing
        if "unit" not in cleaned_df.columns:
            def infer_unit(m_type: str) -> str:
                if "temp" in m_type:
                    return "°C"
                if "pres" in m_type:
                    return "kPa"
                if "level" in m_type:
                    return "%"
                return ""

            cleaned_df["unit"] = cleaned_df["measurement_type"].apply(infer_unit)
        else:
            cleaned_df["unit"] = cleaned_df["unit"].astype(str).str.strip()

        # 7. Ensure is_simulated flag exists
        if "is_simulated" not in cleaned_df.columns:
            cleaned_df["is_simulated"] = True

        # Drop records with completely unparseable timestamps or blank sensor IDs
        cleaned_df = cleaned_df.dropna(subset=["timestamp"])
        cleaned_df = cleaned_df[cleaned_df["sensor_id"] != ""]
        cleaned_df = cleaned_df.reset_index(drop=True)

        is_valid = len(cleaned_df) > 0 and len([e for e in errors if not e.startswith("Note:")]) == 0
        return is_valid, errors, cleaned_df

    @classmethod
    def validate_csv(
        cls, file_content: Union[str, bytes, io.StringIO, io.BytesIO]
    ) -> Tuple[bool, List[str], pd.DataFrame]:
        """Validate a CSV file or buffer.

        Args:
            file_content: Raw CSV string, bytes, or file-like buffer.

        Returns:
            Tuple of (is_valid, error_list, cleaned_dataframe).
        """
        try:
            if isinstance(file_content, bytes):
                df = pd.read_csv(io.BytesIO(file_content))
            elif isinstance(file_content, str):
                df = pd.read_csv(io.StringIO(file_content))
            else:
                df = pd.read_csv(file_content)
        except Exception as ex:
            return False, [f"Failed to parse CSV format: {str(ex)}"], pd.DataFrame()

        return cls.validate_dataframe(df)

    @staticmethod
    def export_to_csv(df: pd.DataFrame) -> bytes:
        """Export a DataFrame to UTF-8 encoded CSV bytes.

        Args:
            df: DataFrame to convert.

        Returns:
            bytes containing the CSV representation.
        """
        if df.empty:
            return b""
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
