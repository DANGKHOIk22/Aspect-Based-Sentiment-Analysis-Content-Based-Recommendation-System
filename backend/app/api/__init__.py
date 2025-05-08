from fastapi import APIRouter
from app.api.routes.recommend_system import router as btc_router

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}

# Include the v1 router
router.include_router(btc_router, prefix="/st", tags=["BTC"])

