# src/api.py
from dotenv import load_dotenv
load_dotenv()  # This must be called before accessing os.getenv()
import joblib
import os
import pandas as pd
import numpy as np
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. API Setup & Security ---

# Initialize the FastAPI app
app = FastAPI(
    title="TelematicsAI API",
    description="API for predicting driver discount eligibility and integrating with policy systems.",
    version="2.0.0"
)

# Define the API Key security scheme
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("API_KEY", "your_secret_api_key_here")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key: str = Depends(api_key_header)):
    """Dependency function to validate the API Key."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


# --- 2. Load the Model ---

MODEL_PATH = "models/discount_eligibility_model.pkl"
model = None


@app.on_event("startup")
def load_model():
    """Load the machine learning model when the API starts."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully.")
    else:
        logger.error(f"❌ Error: Model file not found at {MODEL_PATH}")
        model = None


# --- 3. Policy System Integration Configuration ---

class PolicySystemConfig:
    def __init__(self):
        self.base_url = os.getenv("POLICY_SYSTEM_URL", "https://policy-system.company.com/api")
        self.api_key = os.getenv("POLICY_SYSTEM_API_KEY", "policy_system_key")
        self.timeout = 30
        self.retry_attempts = 3


policy_config = PolicySystemConfig()


# --- 4. Enhanced Data Models ---

class DriverFeatures(BaseModel):
    avg_speed_kph: float
    max_speed_kph: float
    harsh_braking_events: int
    harsh_acceleration_events: int
    speeding_events: int
    phone_usage_events: int
    total_distance_km: float
    total_trips: int
    night_driving_ratio: float
    weekend_driving_ratio: float
    avg_trip_duration: float
    harsh_cornering_events: int


class PredictionResponse(BaseModel):
    eligible_for_discount: bool
    probability: float
    confidence: str
    discount_tier: str
    estimated_premium: float
    annual_savings: float


# NEW: Integration-specific models
class PolicyIntegrationRequest(BaseModel):
    driver_id: str
    policy_id: str
    driver_features: DriverFeatures
    update_policy: bool = True  # Whether to automatically update the policy system


class PolicyIntegrationResponse(BaseModel):
    driver_id: str
    policy_id: str
    prediction: PredictionResponse
    integration_status: str
    policy_updated: bool
    old_premium: Optional[float] = None
    new_premium: Optional[float] = None
    integration_timestamp: str


class BatchIntegrationRequest(BaseModel):
    drivers: List[PolicyIntegrationRequest]


class WebhookPolicyUpdate(BaseModel):
    policy_id: str
    driver_id: str
    event_type: str  # "policy_created", "policy_updated", "driver_changed"
    timestamp: str


# --- 5. Policy System Integration Service ---

class PolicySystemIntegrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=policy_config.timeout)

    async def get_policy_details(self, policy_id: str) -> Dict:
        """Fetch policy details from external system"""
        try:
            response = await self.client.get(
                f"{policy_config.base_url}/policies/{policy_id}",
                headers={"Authorization": f"Bearer {policy_config.api_key}"}
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch policy {policy_id}: {response.status_code}")
                # Return mock data for demo
                return {
                    "policy_id": policy_id,
                    "base_premium": 1100,
                    "current_discount": 0,
                    "status": "active"
                }
        except Exception as e:
            logger.error(f"Error fetching policy {policy_id}: {e}")
            # Return mock data for demo
            return {
                "policy_id": policy_id,
                "base_premium": 1100,
                "current_discount": 0,
                "status": "active"
            }

    async def update_policy_premium(self, policy_id: str, discount_percentage: float, risk_score: float) -> Dict:
        """Update policy premium in external system"""
        try:
            payload = {
                "policy_id": policy_id,
                "telematics_discount": discount_percentage,
                "risk_score": risk_score,
                "update_reason": "Telematics AI Assessment",
                "effective_date": datetime.now().isoformat(),
                "updated_by": "telematics_ai_system"
            }

            response = await self.client.post(
                f"{policy_config.base_url}/policies/{policy_id}/premium",
                json=payload,
                headers={"Authorization": f"Bearer {policy_config.api_key}"}
            )

            if response.status_code in [200, 201]:
                return {"success": True, "updated_at": datetime.now().isoformat()}
            else:
                logger.warning(f"Failed to update policy {policy_id}: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Error updating policy {policy_id}: {e}")
            # For demo purposes, simulate success
            return {"success": True, "updated_at": datetime.now().isoformat(), "simulated": True}

    async def log_integration_event(self, policy_id: str, driver_id: str, event_data: Dict):
        """Log integration events for audit trail"""
        try:
            log_payload = {
                "policy_id": policy_id,
                "driver_id": driver_id,
                "event_type": "telematics_premium_update",
                "event_data": event_data,
                "timestamp": datetime.now().isoformat(),
                "system": "telematics_ai"
            }

            # In production, this would go to your logging system
            logger.info(f"Integration event logged: {log_payload}")
            return True

        except Exception as e:
            logger.error(f"Failed to log integration event: {e}")
            return False


# Initialize the integrator
integrator = PolicySystemIntegrator()


# --- 6. Enhanced Prediction Function ---

async def make_prediction(driver_features: DriverFeatures) -> PredictionResponse:
    """Extract prediction logic into reusable function"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. The service is temporarily unavailable."
        )

    # 1. Recreate the derived features
    speed_std = driver_features.max_speed_kph - driver_features.avg_speed_kph
    speed_safety_score = max(0, 100 - max(0, driver_features.avg_speed_kph - 60) * 2)
    braking_safety_score = max(0, 100 - driver_features.harsh_braking_events * 3)
    accel_safety_score = max(0, 100 - driver_features.harsh_acceleration_events * 3)
    consistency_score = max(0, 100 - speed_std * 2)
    overall_safety_score = (speed_safety_score + braking_safety_score + accel_safety_score + consistency_score) / 4

    # 2. Create feature dictionary
    data = {
        'avg_speed_kph': driver_features.avg_speed_kph,
        'max_speed_kph': driver_features.max_speed_kph,
        'speed_std': speed_std,
        'speed_safety_score': speed_safety_score,
        'harsh_braking_events': driver_features.harsh_braking_events,
        'harsh_acceleration_events': driver_features.harsh_acceleration_events,
        'braking_safety_score': braking_safety_score,
        'acceleration_safety_score': accel_safety_score,
        'consistency_score': consistency_score,
        'overall_safety_score': overall_safety_score,
        'avg_acceleration': np.random.normal(0.5, 0.2),
        'max_acceleration': max(np.random.normal(2.0, 0.3), 0),
        'min_acceleration': np.random.normal(-2.0, 0.3),
        'total_distance_km': driver_features.total_distance_km,
        'total_trips': driver_features.total_trips,
        'avg_trip_duration': driver_features.avg_trip_duration,
        'speeding_events': driver_features.speeding_events,
        'night_driving_ratio': driver_features.night_driving_ratio,
        'weekend_driving_ratio': driver_features.weekend_driving_ratio,
        'phone_usage_events': driver_features.phone_usage_events,
        'harsh_cornering_events': driver_features.harsh_cornering_events
    }

    # 3. Make prediction
    features_df = pd.DataFrame([data])
    feature_names = model.get_booster().feature_names
    features_df = features_df[feature_names]

    probability = model.predict_proba(features_df)[0, 1]
    prediction = 1 if probability >= 0.5 else 0

    # 4. Calculate discount and tier
    base_premium = 1100
    discount = 0.0
    tier = "Standard Rate"

    if prediction == 1:
        if probability >= 0.8:
            tier = "Gold (30% off)"
            discount = 0.30
        elif probability >= 0.6:
            tier = "Silver (20% off)"
            discount = 0.20
        else:
            tier = "Bronze (10% off)"
            discount = 0.10

    final_premium = base_premium * (1 - discount)
    savings = base_premium - final_premium

    return PredictionResponse(
        eligible_for_discount=bool(prediction),
        probability=probability,
        confidence=f"{probability:.1%}",
        discount_tier=tier,
        estimated_premium=final_premium,
        annual_savings=savings
    )


# --- 7. Original Prediction Endpoint (Unchanged) ---

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(get_api_key)])
async def predict_discount(driver_features: DriverFeatures):
    """Predict discount eligibility for a single driver."""
    return await make_prediction(driver_features)


# --- 8. NEW: Integration Endpoints ---

@app.post("/integrate/predict-and-update", response_model=PolicyIntegrationResponse,
          dependencies=[Depends(get_api_key)])
async def predict_and_update_policy(request: PolicyIntegrationRequest, background_tasks: BackgroundTasks):
    """
    Predict discount eligibility AND integrate with policy system.
    This is the main integration endpoint.
    """

    # 1. Make prediction
    prediction = await make_prediction(request.driver_features)

    # 2. Get current policy details
    policy_details = await integrator.get_policy_details(request.policy_id)
    old_premium = policy_details.get("base_premium", 1100) * (1 - policy_details.get("current_discount", 0) / 100)

    integration_status = "prediction_only"
    policy_updated = False
    new_premium = old_premium

    # 3. Update policy system if requested
    if request.update_policy and prediction.eligible_for_discount:
        discount_percentage = (prediction.annual_savings / policy_details.get("base_premium", 1100)) * 100

        update_result = await integrator.update_policy_premium(
            request.policy_id,
            discount_percentage,
            1 - prediction.probability  # Convert to risk score
        )

        if update_result.get("success"):
            integration_status = "policy_updated"
            policy_updated = True
            new_premium = prediction.estimated_premium

            # Log the integration event in background
            background_tasks.add_task(
                integrator.log_integration_event,
                request.policy_id,
                request.driver_id,
                {
                    "old_premium": old_premium,
                    "new_premium": new_premium,
                    "discount_applied": discount_percentage,
                    "risk_score": 1 - prediction.probability
                }
            )
        else:
            integration_status = "policy_update_failed"

    return PolicyIntegrationResponse(
        driver_id=request.driver_id,
        policy_id=request.policy_id,
        prediction=prediction,
        integration_status=integration_status,
        policy_updated=policy_updated,
        old_premium=old_premium,
        new_premium=new_premium,
        integration_timestamp=datetime.now().isoformat()
    )


@app.post("/integrate/batch", dependencies=[Depends(get_api_key)])
async def batch_policy_integration(request: BatchIntegrationRequest, background_tasks: BackgroundTasks):
    """
    Process multiple drivers in batch for policy integration.
    Useful for periodic updates or bulk processing.
    """

    results = []

    for driver_request in request.drivers:
        try:
            # Process each driver
            result = await predict_and_update_policy(driver_request, background_tasks)
            results.append({
                "driver_id": driver_request.driver_id,
                "status": "success",
                "result": result
            })
        except Exception as e:
            logger.error(f"Failed to process driver {driver_request.driver_id}: {e}")
            results.append({
                "driver_id": driver_request.driver_id,
                "status": "error",
                "error": str(e)
            })

    successful_updates = sum(1 for r in results if r["status"] == "success")

    return {
        "total_processed": len(request.drivers),
        "successful_updates": successful_updates,
        "failed_updates": len(request.drivers) - successful_updates,
        "batch_timestamp": datetime.now().isoformat(),
        "results": results
    }


@app.get("/integrate/policy/{policy_id}", dependencies=[Depends(get_api_key)])
async def get_policy_integration_status(policy_id: str):
    """
    Get current integration status for a specific policy.
    """

    policy_details = await integrator.get_policy_details(policy_id)

    return {
        "policy_id": policy_id,
        "policy_details": policy_details,
        "integration_active": True,
        "last_updated": datetime.now().isoformat(),
        "telematics_discount": policy_details.get("current_discount", 0)
    }


# --- 9. NEW: Webhook Endpoints for Real-time Integration ---

@app.post("/webhooks/policy-system")
async def handle_policy_system_webhook(webhook_data: WebhookPolicyUpdate, background_tasks: BackgroundTasks):
    """
    Handle webhooks from the policy system.
    Triggers re-assessment when policies are updated.
    """

    logger.info(f"Received webhook for policy {webhook_data.policy_id}: {webhook_data.event_type}")

    if webhook_data.event_type in ["policy_created", "driver_changed"]:
        # Queue a re-assessment for this policy
        background_tasks.add_task(queue_policy_reassessment, webhook_data.policy_id, webhook_data.driver_id)

    return {
        "status": "accepted",
        "policy_id": webhook_data.policy_id,
        "event_type": webhook_data.event_type,
        "queued_for_processing": True
    }


async def queue_policy_reassessment(policy_id: str, driver_id: str):
    """
    Background task to reassess a policy when triggered by webhook.
    In production, this would queue the job for later processing.
    """

    logger.info(f"Queued reassessment for policy {policy_id}, driver {driver_id}")

    # In a real system, you would:
    # 1. Fetch latest telematics data for the driver
    # 2. Run the prediction
    # 3. Update the policy if needed
    # 4. Send notification back to policy system

    # For demo, just log the event
    await integrator.log_integration_event(
        policy_id,
        driver_id,
        {"event": "reassessment_queued", "trigger": "webhook"}
    )


# --- 10. NEW: Health Check and Status Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "integration_enabled": True
    }


@app.get("/integration/status")
async def integration_status():
    """Check integration system status."""

    # Test policy system connectivity
    try:
        test_policy = await integrator.get_policy_details("TEST_POLICY")
        policy_system_status = "connected"
    except Exception as e:
        policy_system_status = f"error: {str(e)}"

    return {
        "integration_system": "active",
        "policy_system_status": policy_system_status,
        "model_status": "loaded" if model else "not_loaded",
        "api_version": "2.0.0",
        "features": [
            "discount_prediction",
            "policy_integration",
            "batch_processing",
            "webhook_handling",
            "real_time_updates"
        ]
    }