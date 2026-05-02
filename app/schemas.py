from pydantic import BaseModel

class DocumentInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str