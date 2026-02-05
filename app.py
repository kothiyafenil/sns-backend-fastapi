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
# ML ENGINE SETUP
# ======================================================
MODEL_PATH = "nudge_model.joblib"
FEATURES = ["credit_score", "debt_ratio", "outstanding_payment_amount", "annoyance_level"]


def train_initial_model():
    print("🤖 AI Engine: Training model with initial patterns...")
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


if os.path.exists(MODEL_PATH):
    ml_brain = joblib.load(MODEL_PATH)
else:
    ml_brain = train_initial_model()


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
app = FastAPI(title="Smart Nudge AI Engine", version="2.1.0")


@app.get("/")
def health():
    return {"status": "AI Engine is Live 🚀"}


# 1. ADD CUSTOMER (POST)
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


# 2. VIEW ALL CUSTOMERS (GET) - THIS WAS MISSING
@app.get("/customers", response_model=List[CustomerResponse])
def get_all_customers():
    db = SessionLocal()
    try:
        customers = db.query(CustomerDB).all()
        return customers
    finally:
        db.close()


# 3. PREDICT ALL CUSTOMERS (GET)
@app.get("/customers/predict_all")
def predict_all():
    db = SessionLocal()
    try:
        customers = db.query(CustomerDB).all()
        if not customers:
            return []

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
                "name": c.name,
                "ml_risk_level": policy["level"],
                "ml_confidence": f"{confidence}%",
                "action": policy
            })
        return results
    finally:
        db.close()


# 4. GET SINGLE CUSTOMER (GET)
@app.get("/customers/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: str):
    db = SessionLocal()
    try:
        customer = db.query(CustomerDB).filter(CustomerDB.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        return customer
    finally:
        db.close()