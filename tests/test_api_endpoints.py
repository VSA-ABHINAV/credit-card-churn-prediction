"""Test suite to verify FastAPI endpoints in src/api.py."""

import io
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

# Sample valid customer payload
VALID_CUSTOMER = {
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "High School",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 39,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 1,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 12691.0,
    "Total_Revolving_Bal": 777,
    "Avg_Open_To_Buy": 11914.0,
    "Total_Amt_Chng_Q4_Q1": 1.335,
    "Total_Trans_Amt": 1144,
    "Total_Trans_Ct": 42,
    "Total_Ct_Chng_Q4_Q1": 1.625,
    "Avg_Utilization_Ratio": 0.061
}


def test_health():
    """Verify GET /health returns 200 and model_loaded=true."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] is not None
    print("[PASS] test_health passed:", data)


def test_model_info():
    """Verify GET /model-info returns metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["model_type"] == "RandomForestClassifier"
    assert data["decision_threshold"] == 0.3
    assert len(data["expected_input_columns"]) == 19
    print("[PASS] test_model_info passed:", data["model_type"], "threshold:", data["decision_threshold"])


def test_predict_single():
    """Verify POST /predict with a single customer payload."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert "churn_probability" in pred
    assert "churn_risk_flag" in pred
    assert pred["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["threshold_used"] == 0.3
    print("[PASS] test_predict_single passed:", pred)


def test_predict_batch_array():
    """Verify POST /predict with an array of multiple customers."""
    customers = [
        VALID_CUSTOMER,
        {**VALID_CUSTOMER, "Customer_Age": 55, "Gender": "F", "Total_Trans_Ct": 15, "Total_Revolving_Bal": 0},
        {**VALID_CUSTOMER, "Customer_Age": 32, "Gender": "M", "Total_Trans_Ct": 80, "Total_Revolving_Bal": 2200}
    ]
    response = client.post("/predict", json=customers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert len(data["predictions"]) == 3
    print("[PASS] test_predict_batch_array passed: 3 predictions returned")


def test_predict_validation_missing_field():
    """Verify POST /predict returns 422 when required fields are missing."""
    invalid_customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "Customer_Age"}
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("[PASS] test_predict_validation_missing_field passed: 422 received")


def test_predict_validation_invalid_category():
    """Verify POST /predict returns 422 on invalid categorical value."""
    invalid_customer = {**VALID_CUSTOMER, "Education_Level": "Astronaut"}
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("[PASS] test_predict_validation_invalid_category passed: 422 received")


def test_predict_batch_csv():
    """Verify POST /predict/batch accepts CSV upload and returns downloadable CSV."""
    df_sample = pd.DataFrame([VALID_CUSTOMER, {**VALID_CUSTOMER, "Gender": "F", "Total_Trans_Amt": 500}])
    csv_bytes = df_sample.to_csv(index=False).encode("utf-8")

    files = {"file": ("test_customers.csv", csv_bytes, "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert "text/csv" in response.headers.get("content-type", "")
    assert "attachment; filename=\"predictions_test_customers.csv\"" in response.headers.get("content-disposition", "")

    result_df = pd.read_csv(io.StringIO(response.text))
    assert len(result_df) == 2
    assert "churn_probability" in result_df.columns
    assert "churn_risk_flag" in result_df.columns
    assert "risk_level" in result_df.columns
    print("[PASS] test_predict_batch_csv passed: Scored CSV shape", result_df.shape)


def test_frontend_root():
    """Verify GET / serves static/index.html."""
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "text/html" in response.headers.get("content-type", "")
    assert "Customer Churn Risk Intelligence Engine" in response.text
    assert "Customer_Age" in response.text
    print("[PASS] test_frontend_root passed: Served index.html at /")


def test_frontend_static():
    """Verify GET /static/index.html serves the frontend."""
    response = client.get("/static/index.html")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "text/html" in response.headers.get("content-type", "")
    assert "Customer Churn Risk Intelligence Engine" in response.text
    print("[PASS] test_frontend_static passed: Served index.html at /static/index.html")


if __name__ == "__main__":
    test_health()
    test_model_info()
    test_predict_single()
    test_predict_batch_array()
    test_predict_validation_missing_field()
    test_predict_validation_invalid_category()
    test_predict_batch_csv()
    test_frontend_root()
    test_frontend_static()
    print("\nALL 9 TESTS PASSED SUCCESSFULLY!")
