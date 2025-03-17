from fastapi import APIRouter
from app.api.routes.classify_sentiment import router as btc_router
# from app.api.v1.routes.predict_gold import router as gold_router

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}

# Include the v1 router
router.include_router(btc_router, prefix="/st", tags=["BTC"])

