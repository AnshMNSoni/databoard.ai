from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    fields: List[str]

class SuggestRequest(BaseModel):
    fields: List[str]
