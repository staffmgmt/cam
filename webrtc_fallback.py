"""
Minimal fallback WebRTC router when aiortc isn't available.
Provides basic endpoints that return appropriate errors instead of 503.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter(prefix="/webrtc", tags=["webrtc-fallback"])

class OfferRequest(BaseModel):
    offer: Dict[str, Any]

@router.get("/token")
async def get_token():
    """Fallback token endpoint"""
    raise HTTPException(
        status_code=503, 
        detail="WebRTC not available: aiortc dependencies missing. Please check Docker build logs."
    )

@router.post("/offer") 
async def create_offer(request: OfferRequest):
    """Fallback offer endpoint"""
    raise HTTPException(
        status_code=503,
        detail="WebRTC not available: aiortc dependencies missing. Please check Docker build logs."
    )

@router.get("/ping")
async def ping():
    """Fallback ping endpoint"""
    return {"status": "WebRTC unavailable", "aiortc": False, "timestamp": __import__("time").time()}

@router.post("/cleanup")
async def cleanup():
    """Fallback cleanup endpoint"""
    return {"status": "no-op"}