from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import jwt
from datetime import datetime, timedelta
import secrets
import logging
import os
import asyncio

import numpy as np
from sklearn.linear_model import LogisticRegression

import redis.asyncio as redis 

app = FastAPI()

SECRET_KEY = None
ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))  # Changed to default Redis port
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Initialize Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Pydantic models
class InitializeResponse(BaseModel):
    client_id: str
    timestamp: datetime

class GenerateTokenRequest(BaseModel):
    client_id: str

class GenerateTokenResponse(BaseModel):
    recaptcha_token: str

class VerifyTokenRequest(BaseModel):
    recaptcha_token: str

class AssessmentResponse(BaseModel):
    score: float
    reason_codes: List[str]

class ActionRequest(BaseModel):
    recaptcha_token: str

class ActionResponse(BaseModel):
    action: str
    message: str

class MousePosition(BaseModel):
    x: float
    y: float
    timestamp: datetime

class MouseData(BaseModel):
    mouse_movements: List[MousePosition]

class MouseAnalysisRequest(BaseModel):
    recaptcha_token: str
    mouse_data: MouseData

class MouseAnalysisResponse(BaseModel):
    is_human: bool
    confidence: float
    details: Optional[str] = None

class MouseMovementClassifier:
    def __init__(self):
        self.model = LogisticRegression()

        # Fit on dummy data to initialize classes_
        # We are using two samples and two classes: 0 (bot), 1 (human)
        dummy_X = np.array([[0, 0, 0, 0],
                            [1, 1, 1, 1]])
        dummy_y = np.array([0, 1])
        self.model.fit(dummy_X, dummy_y)

        # Now override coefficients and intercept if desired
        # These are arbitrary for demonstration
        self.model.coef_ = np.array([[1.5, -1.0, 0.8, 0.5]])
        self.model.intercept_ = np.array([-0.5])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> List[int]:
        return self.model.predict(X)

classifier = MouseMovementClassifier()

# Utility functions
def create_recaptcha_token(client_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    payload = {
        "client_id": client_id,
        "exp": expiration,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_recaptcha_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def assess_token(token: str) -> AssessmentResponse:
    payload = decode_recaptcha_token(token)
    if not payload:
        return AssessmentResponse(score=0.0, reason_codes=["invalid_token"])
    
    jti = payload.get("jti", "")
    if jti and int(uuid.UUID(jti).hex[-1], 16) % 2 == 0:
        score = 0.9  # High score indicating human
        reason_codes = []
    else:
        score = 0.1  # Low score indicating bot
        reason_codes = ["suspicious_behavior"]
    
    # Store assessment in Redis with 5 minutes TTL
    assessment_data = {
        "score": score,
        "reason_codes": ",".join(reason_codes)
    }
    await redis_client.hset(f"assessment:{token}", mapping=assessment_data)
    await redis_client.expire(f"assessment:{token}", TOKEN_EXPIRATION_MINUTES * 60)
    
    return AssessmentResponse(score=score, reason_codes=reason_codes)

# Mouse Movement Analysis Logic
def extract_features(mouse_movements: List[MousePosition]) -> np.ndarray:
    """
    Extract features from mouse movements.
    """
    if len(mouse_movements) < 2:
        return np.array([0, 0, 0, 0])

    times = [movement.timestamp.timestamp() for movement in mouse_movements]
    xs = [movement.x for movement in mouse_movements]
    ys = [movement.y for movement in mouse_movements]

    time_diffs = np.diff(times)
    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / time_diffs

    # Feature 1: Average speed
    avg_speed = np.mean(speeds) if len(speeds) > 0 else 0

    # Feature 2: Speed variance
    speed_variance = np.var(speeds) if len(speeds) > 0 else 0

    # Feature 3: Number of direction changes
    directions = np.arctan2(dy, dx)
    direction_changes = np.sum(np.abs(np.diff(directions)) > (np.pi / 4))  # Arbitrary threshold

    # Feature 4: Total distance
    total_distance = np.sum(distances)

    return np.array([avg_speed, speed_variance, direction_changes, total_distance])

async def analyze_mouse_movements(mouse_data: MouseData) -> MouseAnalysisResponse:
    features = extract_features(mouse_data.mouse_movements).reshape(1, -1)
    probabilities = classifier.predict_proba(features)
    human_prob = probabilities[0][1]  
    bot_prob = probabilities[0][0]    

    is_human = human_prob > 0.5
    confidence = human_prob if is_human else bot_prob
    details = "Human-like mouse movements detected." if is_human else "Bot-like mouse movements detected."

    return MouseAnalysisResponse(
        is_human=is_human,
        confidence=round(confidence, 2),
        details=details
    )

@app.on_event("startup")
async def startup_event():
    global SECRET_KEY
    SECRET_KEY = secrets.token_urlsafe(128)
    logging.info(f"SECRET_KEY: {SECRET_KEY}")
    try:
        await redis_client.ping()
        logging.info("Connected to Redis successfully.")
    except Exception as e:
        logging.error(f"Redis connection error: {e}")

@app.get("/initialize", response_model=InitializeResponse)
async def initialize():
    client_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    await redis_client.setex(f"client:{client_id}", TOKEN_EXPIRATION_MINUTES * 60, timestamp.isoformat())
    return InitializeResponse(client_id=client_id, timestamp=timestamp)

@app.post("/generate-token", response_model=GenerateTokenResponse)
async def generate_token(request: GenerateTokenRequest):
    """
    Step 2 & 3: Client requests a reCAPTCHA token.
    Generates and returns an encrypted reCAPTCHA token.
    """
    token = create_recaptcha_token(request.client_id)
    # Store token in Redis with 5 minutes TTL
    token_data = {
        "client_id": request.client_id,
        "created_at": datetime.utcnow().isoformat()
    }
    await redis_client.hset(f"token:{token}", mapping=token_data)
    await redis_client.expire(f"token:{token}", TOKEN_EXPIRATION_MINUTES * 60)
    return GenerateTokenResponse(recaptcha_token=token)

@app.post("/verify-token", response_model=AssessmentResponse)
async def verify_token(request: VerifyTokenRequest):
    """
    Step 6 & 7: Backend verifies the reCAPTCHA token by creating an assessment.
    Returns the assessment verdict.
    """
    # Check if token exists in Redis
    token_key = f"token:{request.recaptcha_token}"
    token_exists = await redis_client.exists(token_key)
    if not token_exists:
        raise HTTPException(status_code=400, detail="Invalid or expired reCAPTCHA token.")
    
    # Assess the token
    assessment = await assess_token(request.recaptcha_token)
    return assessment

@app.post("/action", response_model=ActionResponse)
async def perform_action(request: ActionRequest):
    """
    Step 8: Determine the next action based on mouse movement analysis.
    """
    # Check mouse movement analysis
    analysis_key = f"analysis:{request.recaptcha_token}"
    analysis_data = await redis_client.hgetall(analysis_key)
    
    if not analysis_data:
        raise HTTPException(status_code=400, detail="Mouse analysis not found. Please analyze mouse movements first.")
    
    try:
        is_human = analysis_data.get("is_human") == "true"
        confidence = float(analysis_data.get("confidence", 0.0))
    except ValueError:
        raise HTTPException(status_code=500, detail="Corrupted analysis data.")
    
    # Decide based solely on mouse movement analysis
    if is_human and confidence >= 0.5:
        action = "proceed"
        message = "Human verified. Proceeding with the action."
    else:
        action = "block"
        message = "Bot detected. Blocking the action."
    
    return ActionResponse(action=action, message=message)

@app.post("/analyze-mouse", response_model=MouseAnalysisResponse)
async def analyze_mouse(request: MouseAnalysisRequest):
    """
    Analyzes mouse movement data to determine if it's human or bot-like.
    """
    # Verify the token first
    token_key = f"token:{request.recaptcha_token}"
    token_exists = await redis_client.exists(token_key)
    if not token_exists:
        raise HTTPException(status_code=400, detail="Invalid or expired reCAPTCHA token.")
    
    # Analyze mouse movements
    analysis = await analyze_mouse_movements(request.mouse_data)
    # Bind the token to the analysis in Redis
    analysis_key = f"analysis:{request.recaptcha_token}"
    analysis_data = {
        "is_human": str(analysis.is_human).lower(),
        "confidence": str(analysis.confidence),
        "details": analysis.details or ""
    }
    await redis_client.hset(analysis_key, mapping=analysis_data)
    await redis_client.expire(analysis_key, TOKEN_EXPIRATION_MINUTES * 60)
    return analysis

@app.post("/clear")
async def clear_databases():
    """
    Optional: Endpoint to clear all tokens and assessments from Redis.
    Use with caution. Typically for testing purposes.
    """
    await redis_client.flushdb()
    return {"message": "Databases cleared."}
