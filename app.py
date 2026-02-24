from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List
from uuid import uuid4
from datetime import datetime, timedelta
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# ======================================================
# DATABASE SETUP (Persistent)
# ======================================================
DATABASE_URL = "sqlite:///./financial_wellness.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class CustomerDB(Base):
    __tablename__ = "customers"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)  # Name is included
    phone = Column(String)
    credit_score = Column(Integer)
    debt_ratio = Column(Float)
    outstanding_payment_amount = Column(Float)
    annoyance_level = Column(Integer)
    due_date = Column(DateTime)
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
    name: str  # Name is required
    phone: str
    credit_score: int
    debt_ratio: float
    outstanding_payment_amount: float
    annoyance_level: int
    due_date: datetime


# ======================================================
# API ENDPOINTS
# ======================================================
app = FastAPI(title="Financial Wellness AI", version="4.2.0")


# 1. ADD CUSTOMER
@app.post("/customers")
def add_customer(customer: CustomerCreate):
    db = SessionLocal()
    try:
        db_customer = CustomerDB(id=str(uuid4()), **customer.model_dump())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        return {"message": "Customer Added", "id": db_customer.id}
    finally:
        db.close()


# 2. GET ALL CUSTOMERS (So your data is never lost)
@app.get("/customers")
def get_all_customers():
    db = SessionLocal()
    try:
        return db.query(CustomerDB).all()
    finally:
        db.close()


# 3. PREDICT ALL (With ML Confidence and Risk Level)
@app.get("/customers/predict_all")
def predict_all():
    db = SessionLocal()
    try:
        customers = db.query(CustomerDB).all()
        results = []
        now = datetime.utcnow()

        for c in customers:
            input_df = pd.DataFrame([[c.credit_score, c.debt_ratio, c.outstanding_payment_amount, c.annoyance_level]],
                                    columns=FEATURES)
            risk_class = int(ml_brain.predict(input_df)[0])
            prob = ml_brain.predict_proba(input_df)[0]

            mapping = {
                2: {"level": "High", "chan": "SMS", "tone": "Firm", "days": 2},
                1: {"level": "Medium", "chan": "WhatsApp", "tone": "Empathetic", "days": 5},
                0: {"level": "Low", "chan": "Email", "tone": "Friendly", "days": 10}
            }
            policy = mapping[risk_class]
            sched_time = (c.due_date - timedelta(days=policy["days"])).replace(hour=9, minute=0)

            results.append({
                "customer_id": c.id,
                "name": c.name,
                "added_on": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "ml_risk_level": policy["level"],
                "ml_confidence": f"{round(max(prob) * 100, 1)}%",
                "recommendation": {
                    "level": policy["level"],
                    "chan": policy["chan"],
                    "tone": policy["tone"],
                    "scheduled_for": sched_time.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
        return results
    finally:
        db.close()