from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# =========================
# FLEXIBLE INPUT SCHEMA
# =========================
class PredictRequest(BaseModel):
    debt_ratio: Optional[float] = 0.5
    credit_utilization: Optional[float] = 0.5
    credit_score: Optional[int] = 650
    days_overdue: Optional[int] = 0
    app_launches_30d: Optional[int] = 1
    last_login_days_ago: Optional[int] = 7
    annoyance_level: Optional[int] = 1
    outstanding_payment_amount: Optional[float] = 0.0


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "FastAPI is running 🚀"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: PredictRequest):
    """
    Flexible AI logic:
    Works even if some fields are missing
    """

    # -------- Risk logic (simple + explainable) --------
    risk_score = 0

    if data.credit_score < 600:
        risk_score += 2

    if data.debt_ratio > 0.6:
        risk_score += 2

    if data.days_overdue > 7:
        risk_score += 2

    if data.outstanding_payment_amount > 500:
        risk_score += 2

    # -------- Recommendation policy --------
    if risk_score >= 5:
        return {
            "recommended_channel": "SMS",
            "recommended_message_tone": "Firm",
            "recommended_message_type": "Text",
            "recommended_message_time": "Morning",
            "risk_level": "High"
        }

    elif risk_score >= 3:
        return {
            "recommended_channel": "WhatsApp",
            "recommended_message_tone": "Friendly",
            "recommended_message_type": "Image",
            "recommended_message_time": "Afternoon",
            "risk_level": "Medium"
        }

    else:
        return {
            "recommended_channel": "Email",
            "recommended_message_tone": "Informative",
            "recommended_message_type": "Video",
            "recommended_message_time": "Evening",
            "risk_level": "Low"
        }
