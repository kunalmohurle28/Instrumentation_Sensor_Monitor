# Testing & Verification Guide

## Industrial Sensor Monitoring & Predictive Fault Analysis System

### 1. Overview & Test Strategy

The verification strategy employs **Pytest** to ensure that every algorithmic calculation, threshold check, data parsing step, and persistent storage operation satisfies instrumentation engineering constraints:
- **Zero unhandled exceptions**: All edge cases (empty data, missing values, non-numerics, zero standard deviation) must degrade gracefully.
- **Deterministic reproducibility**: Stochastic generators must replicate identical sequences when seeded.
- **Data integrity**: Ingestion and validation must protect against schema violations and corrupt sensor inputs.

---

### 2. Test Suite Structure

The test suite is organized into 7 modular test modules matching the core package structure:

| Test Module | Target System Component | Test Cases |
| :--- | :--- | :--- |
| `test_sensor_simulator.py` | `SensorSimulator` | Initialization, seed reproducibility, boundary arguments, schema verification, fault mode parametrization, single telemetry ticks. |
| `test_fault_detector.py` | `FaultDetector` | Operating limits configuration, Normal classification, High fault detection, Low fault detection, physical boundary checks, invalid/NaN handling, health summary synthesis. |
| `test_anomaly_detector.py` | `AnomalyDetector` | Positive threshold constraints, empty data handling, Z-score accuracy on synthetic distributions, zero-variance signals, missing/corrupt data handling. |
| `test_data_analyzer.py` | `DataAnalyzer` | Analytical central tendency (mean, median), dispersion (std, variance, range, IQR), multi-sensor summary tables, system-wide metrics. |
| `test_spc_analyzer.py` | `SPCAnalyzer` | Control limit formulation (CL, UCL, LCL, UWL, LWL), sample size boundary validation, capability indices ($C_p, C_{pk}$), Shewhart rule violations. |
| `test_database_manager.py` | `DatabaseManager` | Schema initialization, parameterized reading persistence, duplicate prevention (`INSERT OR IGNORE`), fault logging, anomaly logging, database clearing. |
| `test_data_validator.py` | `DataValidator` | Schema validation, missing required columns, empty inputs, raw CSV byte decoding, export to CSV serialization. |

---

### 3. How to Execute Tests

#### 3.1 Run All Unit Tests
From the project root (`Industrial_Sensor_Monitoring_System`):

```bash
python -m pytest tests/ -v
```

#### 3.2 Run Specific Test Module
```bash
python -m pytest tests/test_fault_detector.py -v
python -m pytest tests/test_spc_analyzer.py -v
```

#### 3.3 Run with Code Coverage
```bash
python -m pytest --cov=src tests/ --cov-report=term-missing
```

---

### 4. Edge Cases and Boundary Condition Verification

1. **Zero Standard Deviation ($\sigma = 0$)**:
   - Tested in `test_anomaly_detector_zero_standard_deviation`: When all values are identical, the detector ensures no zero-division error occurs and flags $Z = 0.0$ for identical points.
2. **Missing / Telemetry Dropouts ($NaN$ / None / Non-numeric)**:
   - Handled in `test_fault_detector_classify_reading_invalid_and_missing` and `test_data_analyzer_calculate_sensor_statistics_with_missing_and_invalids`.
3. **Physical Boundary Impossibilities**:
   - Liquid Level $< 0\%$ or $> 100\%$ and absolute temperature violations are classified as `INVALID_READING` rather than legitimate operational faults.
4. **Duplicate Record Prevention**:
   - Parameterized SQL with `INSERT OR IGNORE` ensures re-ingesting duplicate timestamps does not inflate sample counts or corrupt historical metrics.
