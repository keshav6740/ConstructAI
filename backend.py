# main.py
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, Session, declarative_base  # Updated import
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
from typing import Optional, List
import uvicorn

# Database setup
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:root@localhost:5432/construction_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()  # This is now imported directly

# Database Models remain the same
class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    project_type = Column(String)
    square_footage = Column(Float)
    num_floors = Column(Integer)
    soil_condition_score = Column(Float)
    complexity_score = Column(Float)
    initial_estimated_duration = Column(Integer)
    initial_estimated_cost = Column(Float)
    avg_daily_workers = Column(Integer)
    equipment_utilization_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, index=True)
    predicted_delay = Column(Float)
    predicted_cost_overrun = Column(Float)
    actual_delay = Column(Float, nullable=True)
    actual_cost_overrun = Column(Float, nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Risk(Base):
    __tablename__ = "risks"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, index=True)
    risk_type = Column(String)
    probability = Column(Float)
    impact = Column(Float)
    mitigation_status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models updated for V2
class ProjectCreate(BaseModel):
    name: str
    project_type: str
    square_footage: float
    num_floors: int
    soil_condition_score: float
    complexity_score: float
    initial_estimated_duration: int
    initial_estimated_cost: float
    avg_daily_workers: int
    equipment_utilization_rate: float
    status: str = "Active"

class ProjectResponse(ProjectCreate):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True  # Updated from orm_mode

class PredictionResponse(BaseModel):
    project_id: int
    predicted_delay: float
    predicted_cost_overrun: float
    risk_factors: List[dict]
    
    class Config:
        from_attributes = True  # Updated from orm_mode

# ML Model Wrapper remains the same
class EnhancedConstructionPredictor:
    def __init__(self):
        self.delay_model = joblib.load('construction_models_delay.joblib')
        self.cost_model = joblib.load('construction_models_cost.joblib')
        self.scaler = joblib.load('construction_models_scaler.joblib')
        
    def predict_with_risks(self, project_data: dict) -> dict:
        # Standard predictions
        scaled_data = self.scaler.transform([list(project_data.values())])
        delay_pred = self.delay_model.predict(scaled_data)[0]
        cost_pred = self.cost_model.predict(scaled_data)[0]
        
        # Risk analysis
        risks = self._analyze_risks(project_data, delay_pred, cost_pred)
        
        return {
            "predicted_delay": delay_pred,
            "predicted_cost_overrun": cost_pred,
            "risk_factors": risks
        }
    
    def _analyze_risks(self, project_data: dict, delay_pred: float, cost_pred: float) -> List[dict]:
        risks = []
        
        # Complexity risk
        if project_data['complexity_score'] > 7:
            risks.append({
                "type": "High Complexity",
                "probability": 0.8,
                "impact": "High",
                "suggestion": "Consider breaking down into sub-projects"
            })
        
        # Resource risk
        if project_data['avg_daily_workers'] < 50 and project_data['square_footage'] > 50000:
            risks.append({
                "type": "Resource Shortage",
                "probability": 0.7,
                "impact": "Medium",
                "suggestion": "Consider increasing workforce"
            })
        
        # Cost risk
        if cost_pred > 20:  # If predicted cost overrun is more than 20%
            risks.append({
                "type": "High Cost Risk",
                "probability": 0.75,
                "impact": "High",
                "suggestion": "Review cost estimates and add contingency"
            })
        
        return risks

# FastAPI app and routes remain the same
app = FastAPI(title="Construction Project Management API")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

predictor = EnhancedConstructionPredictor()

@app.post("/projects/", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = Project(**project.dict())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/projects/", response_model=List[ProjectResponse])
def read_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = db.query(Project).offset(skip).limit(limit).all()
    return projects

@app.post("/projects/{project_id}/predict", response_model=PredictionResponse)
def create_prediction(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_data = {
        'square_footage': project.square_footage,
        'num_floors': project.num_floors,
        'soil_condition_score': project.soil_condition_score,
        'complexity_score': project.complexity_score,
        'initial_estimated_duration': project.initial_estimated_duration,
        'initial_estimated_cost': project.initial_estimated_cost,
        'avg_daily_workers': project.avg_daily_workers,
        'equipment_utilization_rate': project.equipment_utilization_rate
    }
    
    prediction_result = predictor.predict_with_risks(project_data)
    
    # Save prediction to database
    db_prediction = Prediction(
        project_id=project_id,
        predicted_delay=prediction_result['predicted_delay'],
        predicted_cost_overrun=prediction_result['predicted_cost_overrun']
    )
    db.add(db_prediction)
    
    # Save risks to database
    for risk in prediction_result['risk_factors']:
        db_risk = Risk(
            project_id=project_id,
            risk_type=risk['type'],
            probability=risk['probability'],
            impact=0.0 if risk['impact'] == "Low" else 0.5 if risk['impact'] == "Medium" else 1.0,
            mitigation_status="Identified"
        )
        db.add(db_risk)
    
    db.commit()
    
    return {
        "project_id": project_id,
        "predicted_delay": prediction_result['predicted_delay'],
        "predicted_cost_overrun": prediction_result['predicted_cost_overrun'],
        "risk_factors": prediction_result['risk_factors']
    }

@app.put("/predictions/{project_id}/actuals")
def update_prediction_actuals(
    project_id: int,
    actual_delay: float,
    actual_cost_overrun: float,
    db: Session = Depends(get_db)
):
    prediction = db.query(Prediction).filter(
        Prediction.project_id == project_id,
        Prediction.is_active == True
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Active prediction not found")
    
    prediction.actual_delay = actual_delay
    prediction.actual_cost_overrun = actual_cost_overrun
    db.commit()
    
    return {"message": "Actuals updated successfully"}

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="127.0.0.1", port=8000)  # Updated host address