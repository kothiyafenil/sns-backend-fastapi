from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
# MACHINE LEARNING ENGINE (True ML Logic)
# ======================================================
MODEL_PATH = "nudge_model.joblib"
FEATURES = ["credit_score", "debt_ratio", "outstanding_payment_amount", "annoyance_level"]


def train_system():
    """Trains the model on behavioral patterns instead of hard-coded IF statements"""
    print("🤖 Machine Learning Engine: Analyzing patterns and training...")

    # Historical training data (Synthetic)
    # [credit, debt, amount, annoyance, target_risk]
    data = pd.DataFrame([
        [800, 0.1, 100, 1, 0],  # Low Risk (0)
        [720, 0.2, 500, 2, 0],  # Low Risk
        [650, 0.4, 2000, 3, 1],  # Medium Risk (1)
        [580, 0.5, 3000, 4, 1],  # Medium Risk
        [400, 0.8, 6000, 5, 2],  # High Risk (2)
        [350, 0.9, 8500, 4, 2],  # High Risk
    ], columns=FEATURES + ["risk_class"])

    X = data[FEATURES]
    y = data["risk_class"]

    # Random Forest uses multiple decision trees to find the 'strongest' path
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf


# Load or Train on startup
if os.path.exists(MODEL_PATH):
    ml_brain = joblib.load(MODEL_PATH)
else:
    ml_brain = train_system()


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


# ======================================================
# API ENDPOINTS
# ======================================================
app = FastAPI(title="Smart Nudge AI Engine", version="2.0.0")


@app.get("/")
def health():
    return {"status": "AI Engine Online 🧠"}


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


@app.get("/customers/predict_all")
def predict_all():
    db = SessionLocal()
    try:
        customers = db.query(CustomerDB).all()
        if not customers:
            return []

        results = []
        for c in customers:
            # 1. Transform DB data for the AI
            input_data = pd.DataFrame([[
                c.credit_score, c.debt_ratio, c.outstanding_payment_amount, c.annoyance_level
            ]], columns=FEATURES)

            # 2. Strong Logic: Get Probability Distribution
            # This shows how 'sure' the model is about each risk level
            prob_dist = ml_brain.predict_proba(input_data)[0]
            risk_class = int(ml_brain.predict(input_data)[0])
            confidence = round(max(prob_dist) * 100, 2)

            # 3. Dynamic Policy Mapping
            mapping = {
                2: {"risk": "High", "channel": "Direct SMS", "tone": "Formal/Firm", "content": "Payment Link"},
                1: {"risk": "Medium", "channel": "WhatsApp", "tone": "Empathetic", "content": "Educational Video"},
                0: {"risk": "Low", "channel": "Email", "tone": "Friendly", "content": "Soft Reminder"}
            }
            policy = mapping[risk_class]

            results.append({
                "customer_id": c.id,
                "name": c.name,
                "ai_decision": {
                    "risk_level": policy["risk"],
                    "confidence_score": f"{confidence}%",
                    "recommended_action": {
                        "channel": policy["channel"],
                        "tone": policy["tone"],
                        "content_type": policy["content"]
                    },
                    "scheduled_time": "Morning" if datetime.now().hour < 12 else "Evening"
                }
            })
        return results
    finally:
        db.close()


@app.get("/customers/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: str):
    db = SessionLocal()
    try:
        customer = db.query(CustomerDB).filter(CustomerDB.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Not found")
        return customer
    finally:
        db.close()