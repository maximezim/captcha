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
REDIS_PORT = int(os.getenv("REDIS_PORT", 20001))  
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
    score: float
    accuracy: float
    mouse_movements: List[MousePosition]
    
def create_recaptcha_token(client_id: str) -> str:
    expiration = datetime.now() + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    payload = {
        "client_id": client_id,
        "exp": expiration,
        "iat": datetime.now(),
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

# Endpoints

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
    timestamp = datetime.now()
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
        "created_at": datetime.now().isoformat()
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
    Step 8: Determine the next action based on the assessment verdict.
    """
    assessment_key = f"assessment:{request.recaptcha_token}"
    assessment_data = await redis_client.hgetall(assessment_key)
    
    if not assessment_data:
        # If assessment not found, perform assessment
        assessment = await assess_token(request.recaptcha_token)
    else:
        # Retrieve existing assessment
        score = float(assessment_data.get("score", 0.0))
        reason_codes = assessment_data.get("reason_codes", "").split(",") if assessment_data.get("reason_codes") else []
        assessment = AssessmentResponse(score=score, reason_codes=reason_codes)
    
    if assessment.score >= 0.5:
        action = "proceed"
        message = "Human verified. Proceeding with the action."
    else:
        action = "block"
        message = "Bot detected. Blocking the action."
    
    return ActionResponse(action=action, message=message)

@app.post("/clear")
async def clear_databases():
    """
    Optional: Endpoint to clear all tokens and assessments from Redis.
    Use with caution. Typically for testing purposes.
    """
    # WARNING: This will delete all keys in the current Redis DB.
    await redis_client.flushdb()
    return {"message": "Databases cleared."}
