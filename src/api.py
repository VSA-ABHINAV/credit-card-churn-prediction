"""FastAPI application for serving the Credit Card Churn Prediction production model.

Provides endpoints for health checks, model metadata inspection, single/batch JSON prediction,
and bulk CSV batch scoring.
"""

from contextlib import asynccontextmanager
import io
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_api")

# Model and config locations
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "final_model_config.json"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Global runtime artifacts
model = None
config = None

# Allowed categorical sets for input validation
ALLOWED_EDUCATION = {
    "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"
}
ALLOWED_MARITAL = {
    "Married", "Single", "Divorced", "Unknown"
}
ALLOWED_INCOME = {
    "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"
}
ALLOWED_CARD = {
    "Blue", "Silver", "Gold", "Platinum"
}


def load_artifacts() -> None:
    """Load model pipeline and configuration file into memory."""
    global model, config
    if not MODEL_PATH.exists():
        logger.error(f"Model file missing at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        logger.error(f"Config file missing at: {CONFIG_PATH}")
        raise FileNotFoundError(f"Config file missing at: {CONFIG_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info(
            f"Production model loaded successfully: type={config.get('model_type')}, "
            f"threshold={config.get('decision_threshold')}, version={config.get('created_utc')}"
        )
    except Exception as exc:
        logger.error(f"Error loading model artifacts: {exc}")
        raise


def ensure_artifacts_loaded() -> None:
    """Ensure artifacts are loaded in memory, loading on demand if not yet initialized."""
    global model, config
    if model is None or config is None:
        load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load artifacts once during application startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Credit Card Churn Prediction API",
    description=(
        "Production REST API serving the tuned Random Forest model to predict credit card "
        "customer churn probability and support proactive customer retention campaigns."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the single-page HTML frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found in static directory"
        )
    return FileResponse(index_file)


# Pydantic Schemas
class CustomerInput(BaseModel):
    """Customer features schema representing the 19 input columns expected by the model."""
    Customer_Age: int = Field(..., ge=18, le=100, description="Customer age in years (18-100)")
    Gender: str = Field(..., pattern="^[MF]$", description="Gender of customer ('M' or 'F')")
    Dependent_count: int = Field(..., ge=0, le=10, description="Number of dependents (0-10)")
    Education_Level: str = Field(..., description="Educational qualification")
    Marital_Status: str = Field(..., description="Marital status")
    Income_Category: str = Field(..., description="Annual income bracket")
    Card_Category: str = Field(..., description="Card tier")
    Months_on_book: int = Field(..., ge=0, description="Account tenure in months")
    Total_Relationship_Count: int = Field(..., ge=1, le=10, description="Total bank products held")
    Months_Inactive_12_mon: int = Field(..., ge=0, le=12, description="Months inactive in past 12 months")
    Contacts_Count_12_mon: int = Field(..., ge=0, le=50, description="Inquiries/contacts in past 12 months")
    Credit_Limit: float = Field(..., ge=0, description="Total credit limit")
    Total_Revolving_Bal: int = Field(..., ge=0, description="Total revolving balance")
    Avg_Open_To_Buy: float = Field(..., description="Open to buy credit line")
    Total_Amt_Chng_Q4_Q1: float = Field(..., description="Transaction amount change ratio (Q4 over Q1)")
    Total_Trans_Amt: int = Field(..., ge=0, description="Total transaction amount in last 12 months")
    Total_Trans_Ct: int = Field(..., ge=0, description="Total transaction count in last 12 months")
    Total_Ct_Chng_Q4_Q1: float = Field(..., description="Transaction count change ratio (Q4 over Q1)")
    Avg_Utilization_Ratio: float = Field(..., ge=0.0, le=1.0, description="Average credit card utilization ratio")

    @field_validator("Education_Level")
    @classmethod
    def validate_education(cls, value: str) -> str:
        if value not in ALLOWED_EDUCATION:
            raise ValueError(f"Education_Level must be one of: {sorted(list(ALLOWED_EDUCATION))}")
        return value

    @field_validator("Marital_Status")
    @classmethod
    def validate_marital(cls, value: str) -> str:
        if value not in ALLOWED_MARITAL:
            raise ValueError(f"Marital_Status must be one of: {sorted(list(ALLOWED_MARITAL))}")
        return value

    @field_validator("Income_Category")
    @classmethod
    def validate_income(cls, value: str) -> str:
        if value not in ALLOWED_INCOME:
            raise ValueError(f"Income_Category must be one of: {sorted(list(ALLOWED_INCOME))}")
        return value

    @field_validator("Card_Category")
    @classmethod
    def validate_card(cls, value: str) -> str:
        if value not in ALLOWED_CARD:
            raise ValueError(f"Card_Category must be one of: {sorted(list(ALLOWED_CARD))}")
        return value


class PredictionResponse(BaseModel):
    """Individual prediction outcome."""
    churn_probability: float
    churn_risk_flag: int
    risk_level: str


class PredictResponse(BaseModel):
    """Enclosing response model for /predict endpoint."""
    predictions: List[PredictionResponse]
    threshold_used: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: Optional[str]


def assign_risk_level(probability: float) -> str:
    """Classify churn risk level based on predicted probability.

    - HIGH: >= 0.70
    - MEDIUM: >= 0.30 and < 0.70
    - LOW: < 0.30
    """
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    return "LOW"


def preprocess_input(data: Union[CustomerInput, List[CustomerInput]]) -> pd.DataFrame:
    """Convert validated Pydantic customer inputs to an aligned DataFrame."""
    if isinstance(data, CustomerInput):
        data = [data]

    records = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in data]
    df = pd.DataFrame(records)

    # Encode Gender: 'M' -> 1, 'F' -> 0
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).astype(int)

    # Order strictly by expected schema
    expected_cols = config["expected_input_columns"]
    df = df[expected_cols]
    return df


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness/readiness probe reporting API health and model status."""
    ensure_artifacts_loaded()
    is_loaded = (model is not None and config is not None)
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        version=config.get("created_utc") if config else None
    )


@app.get("/model-info")
async def model_info():
    """Inspect model metadata, training configuration, and threshold parameters."""
    ensure_artifacts_loaded()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model configuration not loaded"
        )
    return {
        "model_type": config.get("model_type"),
        "model_params": config.get("model_params"),
        "decision_threshold": config.get("decision_threshold"),
        "expected_input_columns": config.get("expected_input_columns"),
        "categorical_columns": config.get("categorical_columns"),
        "numeric_columns": config.get("numeric_columns"),
        "training_rows": config.get("training_rows"),
        "sklearn_version": config.get("sklearn_version"),
        "created_utc": config.get("created_utc")
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(customers: Union[CustomerInput, List[CustomerInput]]):
    """Score one or more customer profiles and return churn probabilities with decision flags."""
    ensure_artifacts_loaded()
    if model is None or config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference"
        )

    try:
        df = preprocess_input(customers)
        probas = model.predict_proba(df)[:, 1]
        threshold = float(config.get("decision_threshold", 0.30))
        flags = (probas >= threshold).astype(int)

        predictions = [
            PredictionResponse(
                churn_probability=round(float(p), 4),
                churn_risk_flag=int(f),
                risk_level=assign_risk_level(float(p))
            )
            for p, f in zip(probas, flags)
        ]

        return PredictResponse(
            predictions=predictions,
            threshold_used=threshold,
            model_version=str(config.get("created_utc", "unknown"))
        )
    except Exception as exc:
        logger.error(f"Prediction error encountered: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while computing customer churn prediction"
        )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Bulk score customers via uploaded CSV and return a downloadable CSV with scored outputs."""
    ensure_artifacts_loaded()
    if model is None or config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: Only CSV files are accepted"
        )

    try:
        raw_bytes = await file.read()
        csv_text = raw_bytes.decode("utf-8")
        df_input = pd.read_csv(io.StringIO(csv_text))

        expected_cols = config["expected_input_columns"]
        missing_cols = set(expected_cols) - set(df_input.columns)
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded CSV is missing required columns: {sorted(list(missing_cols))}"
            )

        # Prepare processing copy
        proc_df = df_input.copy()
        if "Gender" in proc_df.columns:
            proc_df["Gender"] = proc_df["Gender"].replace({"M": 1, "F": 0}).astype(int)
        proc_df = proc_df[expected_cols]

        probas = model.predict_proba(proc_df)[:, 1]
        threshold = float(config.get("decision_threshold", 0.30))
        flags = (probas >= threshold).astype(int)
        risk_levels = [assign_risk_level(p) for p in probas]

        # Append predictions to original dataframe
        output_df = df_input.copy()
        output_df["churn_probability"] = probas.round(4)
        output_df["churn_risk_flag"] = flags
        output_df["risk_level"] = risk_levels

        out_buffer = io.StringIO()
        output_df.to_csv(out_buffer, index=False)
        out_buffer.seek(0)

        filename = f"predictions_{Path(file.filename).name}"
        return StreamingResponse(
            iter([out_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing bulk CSV batch"
        )


if __name__ == "__main__":
    import uvicorn
    # Load artifacts ahead of server start when running standalone script
    load_artifacts()
    uvicorn.run(app, host="0.0.0.0", port=8000)
