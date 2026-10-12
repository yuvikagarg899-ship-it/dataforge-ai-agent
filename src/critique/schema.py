from pydantic import BaseModel
from typing import Literal


class CritiqueResult(BaseModel):
    verdict: Literal["approve", "reject"]
    reasoning: str
    concerns: list[str]