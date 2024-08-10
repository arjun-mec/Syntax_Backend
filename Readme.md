## ML-Powered Education Outcome Prediction API

This FastAPI application predicts educational outcomes using time series forecasting. It leverages the Facebook Prophet library to forecast scores for various educational categories based on historical data from the PGI-D dataset.

**Features:**

- **Predictive Analytics:** Forecasts scores for future years.
- **Time Series Forecasting:** Employs Facebook Prophet for accurate predictions.
- **RESTful API:** Easy integration with other applications.
- **CORS Enabled:** Permits requests from specified origins (e.g., your frontend).

### Getting Started:

1. **Clone:** `git clone https://github.com/arjun-mec/Syntax_backend.git`
2. **Virtual Environment (Recommended):**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. **Install:** `pip install -r requirements.txt`
4. **Run:** `uvicorn backend:app --reload`

### API Usage:

**POST /predict/**

**Request Body (JSON):**

```json
{
  "state": "State Name",
  "district": "District Name",
  "year": 2024
}
```

**Response Body (JSON):**

```json
{
  "State/UT": "State Name",
  "District": "District Name",
  "Year": 2024,
  "Outcome": 85.67,
  // ... category scores ...
  "Overall": 547.98
}
```
