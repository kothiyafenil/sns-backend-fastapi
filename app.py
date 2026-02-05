from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# ======================================================
# DATABASE SETUP
# ======================================================
DATABASE_URL = "sqlite:///./customers.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class CustomerDB(Base):
    __tablename__ = "customers"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    phone = Column(String)
    credit_score = Column(Integer)
    debt_ratio = Column(Float)
    outstanding_payment_amount = Column(Float)
    annoyance_level = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ======================================================
# ML ENGINE SETUP
# ======================================================
MODEL_PATH = "nudge_model.joblib"
FEATURES = ["credit_score", "debt_ratio", "outstanding_payment_amount", "annoyance_level"]


def train_initial_model():
    data = pd.DataFrame([
        [800, 0.1, 100, 1, 0], [750, 0.2, 500, 1, 0],
        [620, 0.4, 2000, 3, 1], [580, 0.5, 3000, 2, 1],
        [450, 0.8, 6000, 5, 2], [350, 0.9, 8000, 4, 2],
    ], columns=FEATURES + ["target"])
    X, y = data[FEATURES], data["target"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


ml_brain = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_initial_model()


# ======================================================
# SCHEMAS
# ======================================================
class CustomerCreate(BaseModel):
    name: str
    phone: str
    credit_score: int
    debt_ratio: float
    outstanding_payment_amount: float
    annoyance_level: int


class CustomerResponse(CustomerCreate):
    id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ======================================================
# API ENDPOINTS
# ======================================================
app = FastAPI(title="Smart Nudge AI Engine", version="3.0.0")


@app.get("/")
def health():
    return {"status": "AI Engine is Live 🚀", "timestamp": datetime.now()}


# ADD CUSTOMER
@app.post("/customers", response_model=CustomerResponse)
def add_customer(customer: CustomerCreate):
    db = SessionLocal()
    try:
        db_customer = CustomerDB(id=str(uuid4()), **customer.model_dump())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        return db_customer
    finally:
        db.close()


# VIEW ALL CUSTOMERS
@app.get("/customers", response_model=List[CustomerResponse])
def get_all_customers():
    db = SessionLocal()
    try:
        return db.query(CustomerDB).all()
    finally:
        db.close()


# PREDICT ALL (Now with Date & Time!)
@app.get("/customers/predict_all")
def predict_all():
    db = SessionLocal()
    try:
        customers = db.query(CustomerDB).all()
        if not customers: return []

        results = []
        for c in customers:
            input_df = pd.DataFrame([[c.credit_score, c.debt_ratio, c.outstanding_payment_amount, c.annoyance_level]],
                                    columns=FEATURES)
            risk_class = int(ml_brain.predict(input_df)[0])
            prob_dist = ml_brain.predict_proba(input_df)[0]
            confidence = round(max(prob_dist) * 100, 2)

            mapping = {
                2: {"level": "High", "chan": "SMS", "tone": "Firm"},
                1: {"level": "Medium", "chan": "WhatsApp", "tone": "Empathetic"},
                0: {"level": "Low", "chan": "Email", "tone": "Friendly"}
            }
            policy = mapping[risk_class]

            results.append({
                "customer_id": c.id,
                "name": c.name,
                "added_on": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),  # Formatted Date
                "ml_risk_level": policy["level"],
                "ml_confidence": f"{confidence}%",
                "recommendation": policy
            })
        return results
    finally:
        db.close()