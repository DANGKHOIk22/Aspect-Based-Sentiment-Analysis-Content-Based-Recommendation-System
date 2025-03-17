from fastapi import APIRouter, HTTPException, status
from app.schemas.st import Model
from app.api.controller.classify_sentiment import predict_label

router = APIRouter()

@router.post("/predict",
             description="Classify sentiment")
async def classify_sentiment(input_data: Model):
    try:
        results = await predict_label(input_data)
        if results is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Sentiment classification failed"
            )
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in sentiment classification: {str(e)}"
        )