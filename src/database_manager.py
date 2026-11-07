"""SQLite Historical Database Management Module.

Provides robust, persistent relational storage for industrial sensor telemetry,
fault events, and statistical anomalies:
- Auto-initialization of schema and tables.
- Parameterized SQL execution to prevent injection.
- Duplicate record avoidance via composite unique constraints (timestamp, sensor_id).
- Flexible historical filtering (time ranges, sensor IDs, fault status).
- Portable path resolution using pathlib.Path.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sensor_monitoring.db"


class DatabaseManager:
    """Manages SQLite persistent storage for telemetry, faults, and anomalies."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the DatabaseManager with a database file path.

        Args:
            db_path: Path to the SQLite file. If None or default, places in project data/ directory.
        """
        self._is_memory = str(db_path) == ":memory:"
        if db_path is None:
            self.db_path: Union[str, Path] = DEFAULT_DB_PATH
        elif self._is_memory:
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        # Persistent connection for in-memory databases
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
        else:
            assert isinstance(self.db_path, Path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        if self._is_memory and self._memory_conn is not None:
            return self._memory_conn

        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_database(self) -> None:
        """Create database tables and performance indexes if they do not exist."""
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()

                # Table 1: Sensor Readings
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sensor_readings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        sensor_id TEXT NOT NULL,
                        measurement_type TEXT NOT NULL,
                        reading_value REAL,
                        unit TEXT,
                        is_simulated INTEGER DEFAULT 1,
                        injected_fault TEXT DEFAULT 'None',
                        created_at TEXT DEFAULT (datetime('now', 'utc')),
                        CONSTRAINT unq_reading UNIQUE (timestamp, sensor_id)
                    );
                    """
                )

                # Table 2: Fault Records
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fault_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        sensor_id TEXT NOT NULL,
                        measurement_type TEXT,
                        reading_value REAL,
                        unit TEXT,
                        fault_status TEXT NOT NULL,
                        fault_description TEXT,
                        threshold_violated REAL,
                        created_at TEXT DEFAULT (datetime('now', 'utc'))
                    );
                    """
                )

                # Table 3: Anomaly Records
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS anomaly_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        sensor_id TEXT NOT NULL,
                        measurement_type TEXT,
                        reading_value REAL,
                        sample_mean REAL,
                        sample_std REAL,
                        z_score REAL,
                        threshold REAL,
                        anomaly_reason TEXT,
                        created_at TEXT DEFAULT (datetime('now', 'utc'))
                    );
                    """
                )

                # Performance Indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_readings_ts ON sensor_readings(timestamp);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_readings_sensor ON sensor_readings(sensor_id);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_faults_ts ON fault_records(timestamp);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_anomalies_ts ON anomaly_records(timestamp);"
                )
            if not self._is_memory:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite database at {self.db_path}: {e}")
            raise

    def save_readings(self, df: pd.DataFrame) -> int:
        """Persist a DataFrame of sensor readings to the database.

        Uses INSERT OR IGNORE to prevent duplicate entries based on (timestamp, sensor_id).

        Args:
            df: DataFrame containing at least ['timestamp', 'sensor_id', 'measurement_type', 'reading_value'].

        Returns:
            Number of newly inserted records.
        """
        if df.empty:
            return 0

        insert_sql = """
            INSERT OR IGNORE INTO sensor_readings (
                timestamp, sensor_id, measurement_type, reading_value, unit, is_simulated, injected_fault
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """

        records: List[Tuple[Any, ...]] = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            if isinstance(ts, (pd.Timestamp, pd.DatetimeIndex)):
                ts_str = ts.isoformat()
            else:
                ts_str = str(ts)

            sid = str(row.get("sensor_id", ""))
            mtype = str(row.get("measurement_type", ""))
            val = row.get("reading_value")
            val_flt = None if pd.isna(val) else float(val)
            unit = str(row.get("unit", ""))
            is_sim = 1 if bool(row.get("is_simulated", True)) else 0
            inj = str(row.get("injected_fault", "None"))

            records.append((ts_str, sid, mtype, val_flt, unit, is_sim, inj))

        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.executemany(insert_sql, records)
                inserted = cursor.rowcount
            if not self._is_memory:
                conn.close()
            return max(inserted, 0)
        except sqlite3.Error as e:
            logger.error(f"Error saving readings to database: {e}")
            raise

    def save_faults(self, df: pd.DataFrame) -> int:
        """Persist detected fault events to the fault_records table.

        Args:
            df: DataFrame of fault events.

        Returns:
            Number of fault records persisted.
        """
        if df.empty:
            return 0

        insert_sql = """
            INSERT INTO fault_records (
                timestamp, sensor_id, measurement_type, reading_value, unit,
                fault_status, fault_description, threshold_violated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """

        records: List[Tuple[Any, ...]] = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            ts_str = ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
            sid = str(row.get("sensor_id", ""))
            mtype = str(row.get("measurement_type", ""))
            val = row.get("reading_value")
            val_flt = None if pd.isna(val) else float(val)
            unit = str(row.get("unit", ""))
            status = str(row.get("fault_status", "UNKNOWN"))
            desc = str(row.get("fault_description", ""))
            thresh = row.get("threshold_violated")
            thresh_flt = None if pd.isna(thresh) else float(thresh)

            records.append((ts_str, sid, mtype, val_flt, unit, status, desc, thresh_flt))

        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.executemany(insert_sql, records)
                inserted = cursor.rowcount
            if not self._is_memory:
                conn.close()
            return max(inserted, 0)
        except sqlite3.Error as e:
            logger.error(f"Error saving faults to database: {e}")
            raise

    def save_anomalies(self, df: pd.DataFrame) -> int:
        """Persist statistical anomaly events to the anomaly_records table.

        Args:
            df: DataFrame containing detected anomaly events.

        Returns:
            Number of anomaly records persisted.
        """
        if df.empty:
            return 0

        insert_sql = """
            INSERT INTO anomaly_records (
                timestamp, sensor_id, measurement_type, reading_value,
                sample_mean, sample_std, z_score, threshold, anomaly_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        records: List[Tuple[Any, ...]] = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            ts_str = ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
            sid = str(row.get("sensor_id", ""))
            mtype = str(row.get("measurement_type", ""))
            val = row.get("reading_value")
            val_flt = None if pd.isna(val) else float(val)
            mean_val = row.get("mean")
            std_val = row.get("std_dev")
            z_score = row.get("z_score")
            thresh = row.get("threshold", 3.0)
            reason = str(row.get("anomaly_reason", ""))

            records.append(
                (
                    ts_str,
                    sid,
                    mtype,
                    val_flt,
                    float(mean_val) if pd.notna(mean_val) else None,
                    float(std_val) if pd.notna(std_val) else None,
                    float(z_score) if pd.notna(z_score) else None,
                    float(thresh) if pd.notna(thresh) else None,
                    reason,
                )
            )

        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.executemany(insert_sql, records)
                inserted = cursor.rowcount
            if not self._is_memory:
                conn.close()
            return max(inserted, 0)
        except sqlite3.Error as e:
            logger.error(f"Error saving anomalies to database: {e}")
            raise

    def query_readings(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sensor_ids: Optional[List[str]] = None,
        measurement_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve historical telemetry with parameterized filtering.

        Args:
            start_time: Optional ISO timestamp lower bound.
            end_time: Optional ISO timestamp upper bound.
            sensor_ids: Optional list of sensor identifiers.
            measurement_type: Optional measurement type filter.
            limit: Maximum number of rows to return.

        Returns:
            pd.DataFrame of queried readings.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if sensor_ids:
            placeholders = ", ".join(["?"] * len(sensor_ids))
            conditions.append(f"sensor_id IN ({placeholders})")
            params.extend(sensor_ids)
        if measurement_type:
            conditions.append("measurement_type = ?")
            params.append(measurement_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None and limit > 0 else ""

        query = f"""
            SELECT id, timestamp, sensor_id, measurement_type, reading_value, unit, is_simulated, injected_fault
            FROM sensor_readings
            {where_clause}
            ORDER BY timestamp ASC
            {limit_clause};
        """

        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            if not self._is_memory:
                conn.close()
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to query readings: {e}")
            raise

    def query_faults(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sensor_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve historical fault event records.

        Args:
            start_time: Optional ISO timestamp lower bound.
            end_time: Optional ISO timestamp upper bound.
            sensor_ids: Optional sensor filter.
            limit: Maximum records.

        Returns:
            pd.DataFrame of historical faults.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if sensor_ids:
            placeholders = ", ".join(["?"] * len(sensor_ids))
            conditions.append(f"sensor_id IN ({placeholders})")
            params.extend(sensor_ids)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None and limit > 0 else ""

        query = f"""
            SELECT id, timestamp, sensor_id, measurement_type, reading_value, unit,
                   fault_status, fault_description, threshold_violated
            FROM fault_records
            {where_clause}
            ORDER BY timestamp DESC
            {limit_clause};
        """

        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            if not self._is_memory:
                conn.close()
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to query faults: {e}")
            raise

    def query_anomalies(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        sensor_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve historical anomaly event records.

        Args:
            start_time: Optional ISO timestamp lower bound.
            end_time: Optional ISO timestamp upper bound.
            sensor_ids: Optional sensor filter.
            limit: Maximum records.

        Returns:
            pd.DataFrame of historical anomalies.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if sensor_ids:
            placeholders = ", ".join(["?"] * len(sensor_ids))
            conditions.append(f"sensor_id IN ({placeholders})")
            params.extend(sensor_ids)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None and limit > 0 else ""

        query = f"""
            SELECT id, timestamp, sensor_id, measurement_type, reading_value,
                   sample_mean, sample_std, z_score, threshold, anomaly_reason
            FROM anomaly_records
            {where_clause}
            ORDER BY timestamp DESC
            {limit_clause};
        """

        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            if not self._is_memory:
                conn.close()
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except sqlite3.Error as e:
            logger.error(f"Failed to query anomalies: {e}")
            raise

    def get_database_statistics(self) -> Dict[str, Any]:
        """Return counts of records in all database tables."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sensor_readings;")
            readings_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM fault_records;")
            faults_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM anomaly_records;")
            anomalies_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM sensor_readings;"
            )
            ts_row = cursor.fetchone()
            min_ts, max_ts = ts_row[0], ts_row[1]

            if not self._is_memory:
                conn.close()

            return {
                "database_path": str(self.db_path),
                "total_readings": readings_count,
                "total_faults": faults_count,
                "total_anomalies": anomalies_count,
                "earliest_timestamp": min_ts,
                "latest_timestamp": max_ts,
            }
        except sqlite3.Error as e:
            logger.error(f"Failed to query database statistics: {e}")
            return {
                "database_path": str(self.db_path),
                "total_readings": 0,
                "total_faults": 0,
                "total_anomalies": 0,
                "earliest_timestamp": None,
                "latest_timestamp": None,
            }

    def clear_database(self) -> None:
        """Clear all records from all tables (useful for unit testing or resets)."""
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sensor_readings;")
                cursor.execute("DELETE FROM fault_records;")
                cursor.execute("DELETE FROM anomaly_records;")
            if not self._is_memory:
                conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to clear database: {e}")
            raise
