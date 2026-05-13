# Alpha Mining System - Integration Specification

Version: 1.0
Date: 2026-05-12

## Overview

This document specifies the interface contract between the **Alpha Mining Multi-Agent System** and the **Backtesting Framework**. The backtesting team should implement the interfaces defined here to enable end-to-end alpha factor evaluation.

---

## 1. Evaluator Interface

### 1.1 HTTP Endpoint

```
POST /evaluate
Content-Type: application/json
```

### 1.2 Request Schema

```json
{
  "alpha_id": "string (optional, internal tracking)",
  "alpha_description": "string (mathematical/economic description)",
  "alpha_code": "string (Python/numpy expression)",
  "parameters": {
    "window": 20,
    "threshold": 0.5,
    ...
  },
  "eval_config": {
    "symbols": ["A-share", "SHSE.600000", ...],
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "rebalance_freq": "daily|weekly|monthly",
    "universe": "A-share|HS300|ZZ500|...",
    "commission_rate": 0.0003,
    "risk_free_rate": 0.03
  }
}
```

### 1.3 Response Schema - Success

```json
{
  "status": "success",
  "alpha_id": "string (echoed or generated)",
  "metrics": {
    "ic_mean": 0.0512,
    "ic_std": 0.0893,
    "ir": 0.5734,
    "sharpe": 1.3842,
    "max_drawdown": 0.1523,
    "turnover": 0.3512,
    "long_short_return": 0.0823,
    "win_rate": 0.5421
  },
  "period_stats": {
    "ic_by_year": {
      "2020": 0.048,
      "2021": 0.055,
      "2022": 0.041,
      "2023": 0.061
    }
  },
  "error_message": null
}
```

### 1.4 Response Schema - Compliance Error

When the alpha code fails compliance checks (syntax, invalid operations, look-ahead bias, etc.), return:

```json
{
  "status": "error",
  "alpha_id": "string",
  "metrics": null,
  "error_code": "COMPLIANCE_ERROR",
  "error_message": "Division by zero in expression: ts_mean(close, 0)",
  "error_details": {
    "type": "DIVISION_BY_ZERO",
    "location": "line 1, position 45-48",
    "suggestion": "Ensure window parameter > 0"
  },
  "recoverable": true
}
```

### 1.5 Response Schema - Evaluation Error (non-recoverable)

```json
{
  "status": "error",
  "alpha_id": "string",
  "metrics": null,
  "error_code": "EVAL_ERROR",
  "error_message": "Insufficient data coverage for given universe",
  "error_details": {
    "type": "DATA_INSUFFICIENT",
    "coverage": 0.45,
    "required": 0.90
  },
  "recoverable": false
}
```

---

## 2. Error Code Reference

### 2.1 Compliance Errors (recoverable, retry allowed)

| Error Code            | Type     | Description                                  | Suggestion                     |
| --------------------- | -------- | -------------------------------------------- | ------------------------------ |
| `COMPLIANCE_ERROR`  | General  | Code failed compliance checks                | Regenerate with corrected code |
| `SYNTAX_ERROR`      | Parsing  | Invalid Python syntax                        | Fix syntax errors              |
| `INVALID_OPERATION` | Runtime  | Division by zero, log of negative, etc.      | Add bounds checks              |
| `LOOK_AHEAD_BIAS`   | Risk     | Factor uses future data                      | Remove look-ahead              |
| `FUTURE_FUNC_USAGE` | Risk     | Unauthorized future-looking functions        | Use only allowed functions     |
| `UNAUTHORIZED_FUNC` | Security | Prohibited function calls (eval, exec, etc.) | Remove prohibited calls        |

### 2.2 Evaluation Errors (non-recoverable)

| Error Code              | Type    | Description                     |
| ----------------------- | ------- | ------------------------------- |
| `EVAL_ERROR`          | General | Evaluation failed               |
| `DATA_INSUFFICIENT`   | Data    | Not enough data coverage        |
| `COMPUTATION_TIMEOUT` | System  | Computation exceeded time limit |
| `UNIVERSE_EMPTY`      | Data    | No matching securities          |

---

## 3. Compliance Check Requirements

The backtesting framework **MUST** perform the following compliance checks before evaluation:

1. **Syntax Validation**: Verify alpha_code is valid Python/numpy
2. **Division by Zero**: Ensure no division by zero (e.g., `ts_mean(close, 0)`)
3. **Log/Negative Check**: Ensure log arguments are positive
4. **Look-Ahead Bias**: Ensure no use of future data (e.g., `close.shift(-1)`)
5. **Authorized Functions**: Only allow predefined functions (see Section 4)
6. **Data Availability**: Ensure sufficient data coverage (e.g., >90% non-null)
7. **Computation Limits**: Timeout after reasonable duration (e.g., 30s)
8. **Security**: No code injection (eval, exec, import, etc.)

---

## 4. Authorized Factor Functions

Only the following functions should be allowed in `alpha_code`:

### Price Data

- `open`, `high`, `low`, `close`, `volume`, `amount`
- `vwap`, `limit_up`, `limit_down`

### Time-Series Operations

- `ts_mean(x, window)`, `ts_sum(x, window)`, `ts_std(x, window)`
- `ts_max(x, window)`, `ts_min(x, window)`
- `ts_rank(x, window)`, `ts_skew(x, window)`, `ts_kurt(x, window)`
- `ts_argmax(x, window)`, `ts_argmin(x, window)`
- `ts_delay(x, n)`, `ts_corr(x, y, window)`, `ts_cov(x, y, window)`
- `ts_zscore(x, window)`, `ts_normalize(x, window)`

### Cross-Sectional Operations

- `cs_rank(x)`, `cs_zscore(x)`, `cs_percentile(x)`
- `cs_corr(x, y)`, `cs_mean(x)`, `cs_std(x)`

### Mathematical Functions

- `log(x)`, `abs(x)`, `sign(x)`, `sqrt(x)`
- `exp(x)`, `power(x, n)`
- `max(x, y)`, `min(x, y)`
- `clip(x, min, max)`, `winsorize(x, limits)`

### Factor Composition

- `+`, `-`, `*`, `/`, `**`
- All numpy functions (`np.` prefix allowed)

---

## 5. Metrics Definition

### 5.1 IC (Information Coefficient)

- **ic_mean**: Mean Pearson correlation between factor and forward returns
- **ic_std**: Standard deviation of IC across time
- **ir**: Information Ratio = ic_mean / ic_std

### 5.2 Performance

- **sharpe**: Annualized Sharpe ratio of long-short portfolio
- **max_drawdown**: Maximum drawdown of cumulative returns
- **long_short_return**: Annualized return of long-short portfolio
- **turnover**: Average daily portfolio turnover
- **win_rate**: Fraction of periods with positive long-side return

---

## 6. Integration Notes

### 6.1 Recovery from Compliance Errors

When `COMPLIANCE_ERROR` is returned:

1. The alpha mining system will regenerate the factor code
2. Up to `max_proposer_retries` attempts (default: 3) are allowed
3. The `error_details.suggestion` field should be included to guide regeneration
4. After max retries, the factor is abandoned with status "skipped"

### 6.2 Performance Expectations

- Evaluation latency: <5s for typical factors, <30s for complex ones
- Data coverage: >90% non-null values required
- Time range: Minimum 1 year of data recommended

### 6.3 Data Format

- All prices should be **unadjusted** (split/dividend adjusted separately if needed)
- Volume should be in original units (shares or money, clarify in config)
- Use `suspend` flag to handle suspend-trading days

---

## 7. Example Integration Test

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "alpha_code": "(close - ts_mean(close, 20)) / ts_std(close, 20)",
    "alpha_description": "Z-score of price relative to 20-day moving average",
    "parameters": {"window": 20},
    "eval_config": {
      "universe": "A-share",
      "start_date": "2020-01-01",
      "end_date": "2023-12-31"
    }
  }'
```

Expected: Returns `status: "success"` with metrics or `status: "error"` with error details.

---

## 8. Contact

For questions about this specification, contact the Alpha Mining team.
