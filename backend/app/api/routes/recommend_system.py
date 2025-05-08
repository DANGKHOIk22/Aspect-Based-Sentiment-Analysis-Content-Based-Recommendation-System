from fastapi import APIRouter, HTTPException, status
from app.schemas.st import Model
from app.api.controller.recommend_system import predict_label

router = APIRouter()

@router.post("/predict",
             description="Recommend system")
async def recommend_system(input_data: Model):
    try:
        results = await predict_label(input_data)
        recommend_ids = results["recommend_ids"]
        sentiment_label = results["sentiment_label"]


        if recommend_ids is None or sentiment_label is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="recommend_system failed"
            )
        return {
            'recommend_ids': recommend_ids,
            'sentiment_label':  sentiment_label
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in recommend_system: {str(e)}"
        )