from pydantic import BaseModel, Field

class DocumentInput(BaseModel):
    text: str = Field(min_length=1)
    title: str = "Untitled"

class QueryInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
