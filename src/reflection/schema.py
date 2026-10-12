from pydantic import BaseModel
from typing import Literal,Optional
class ReflectionResult(BaseModel):
    decision:Literal["accept","retry"]
    reasoning:str
    suggested_changes:Optional[str] = None