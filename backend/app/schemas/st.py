from pydantic import BaseModel, Field

class Model(BaseModel):
    book_id: int = Field(..., example=123)
    comment: str = Field(..., example="I loved the story and characters!")
