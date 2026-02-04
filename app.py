from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

app = FastAPI(
    title="Smart Nudge System – Intelligent Decision Engine",
    version="1.0.0"
)

# ======================================================
# DATA MODELS
# ======================================================

class PredictRequest(BaseModel):
    credit_score: int
    debt_ratio: float
    outstanding_payment_amount: float
    annoyance_level: int


class Customer(BaseModel):
    id: Optional[str] = None
    name: str
    phone: str
    credit_score: int
    debt_ratio: float
    outstanding_payment_amount: float
    annoyance_level: int


# ======================================================
# IN-MEMORY DATABASE (FOR DISSERTATION / DEMO)
# ======================================================

CUSTOMERS_DB: List[Customer] = []


# ======================================================
# ROOT
# ======================================================

@app.get("/")
def root():
    return {"status": "FastAPI is running 🚀"}


# ======================================================
# CUSTOMER MANAGEMENT ENDPOINTS
# ======================================================

@app.post("/customers")
def add_customer(customer: Customer):
    customer.id = str(uuid4())
    CUSTOMERS_DB.append(customer)
    return customer


@app.get("/customers")
def get_customers():
    return CUSTOMERS_DB


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    customer = next((c for c in CUSTOMERS_DB if c.id == customer_id), None)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


# ======================================================
# INTELLIGENT SCORING ENGINE
# ======================================================

def compute_scores(
    credit_score: int,
    debt_ratio: float,
    outstanding: float,
    annoyance: int
):
    # Financial risk (payment default probability proxy)
    financial_risk = (
        (1 - credit_score / 900) * 0.4 +
        debt_ratio * 0.35 +
        min(outstanding / 5000, 1) * 0.25
    )

    # Engagement probability
    engagement_score = (
        (credit_score / 900) * 0.5 +
        (1 - debt_ratio) * 0.3 +
        (1 - annoyance / 5) * 0.2
    )

    # Communication fatigue & uninstall risk
    fatigue_risk = (
        annoyance * 0.6 +
        financial_risk * 0.4
    )

    return financial_risk, engagement_score, fatigue_risk


# ======================================================
# POLICY DECISION ENGINE
# ======================================================

def generate_recommendation(
    credit_score: int,
    debt_ratio: float,
    outstanding: float,
    annoyance: int
):
    financial_risk, engagement, fatigue = compute_scores(
        credit_score, debt_ratio, outstanding, annoyance
    )

    # ---------------- CHANNEL SELECTION ----------------
    if fatigue > 0.65:
        channel = "Email"            # least intrusive
    elif financial_risk > 0.7:
        channel = "SMS"              # urgency dominates
    else:
        channel = "WhatsApp"         # balanced engagement

    # ---------------- MESSAGE TYPE ----------------
    if engagement < 0.4:
        message_type = "Text"
    elif engagement < 0.7:
        message_type = "Image"
    else:
        message_type = "Video"

    # ---------------- MESSAGE TONE ----------------
    if financial_risk > 0.75 and annoyance < 2:
        tone = "Firm"
    elif fatigue > 0.6:
        tone = "Soft"
    else:
        tone = "Empathetic"

    # ---------------- MESSAGE TIMING ----------------
    if channel == "SMS":
        message_time = "Morning"
    elif channel == "WhatsApp":
        message_time = "Evening"
    else:
        message_time = "Afternoon"

    # ---------------- RISK LABEL ----------------
    if financial_risk >= 0.7:
        risk_level = "High"
    elif financial_risk >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_level": risk_level,
        "financial_risk_score": round(financial_risk, 2),
        "engagement_score": round(engagement, 2),
        "fatigue_risk_score": round(fatigue, 2),
        "recommended_channel": channel,
        "recommended_message_time": message_time,
        "recommended_message_tone": tone,
        "recommended_message_type": message_type
    }


# ======================================================
# GENERIC PREDICT ENDPOINT
# ======================================================

@app.post("/predict")
def predict(request: PredictRequest):
    return generate_recommendation(
        credit_score=request.credit_score,
        debt_ratio=request.debt_ratio,
        outstanding=request.outstanding_payment_amount,
        annoyance=request.annoyance_level
    )


# ======================================================
# CUSTOMER-BASED PREDICT ENDPOINT
# ======================================================

@app.post("/customers/{customer_id}/predict")
def predict_for_customer(customer_id: str):
    customer = next((c for c in CUSTOMERS_DB if c.id == customer_id), None)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    return generate_recommendation(
        credit_score=customer.credit_score,
        debt_ratio=customer.debt_ratio,
        outstanding=customer.outstanding_payment_amount,
        annoyance=customer.annoyance_level
    )
