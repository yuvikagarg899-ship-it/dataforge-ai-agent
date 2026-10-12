from pydantic import BaseModel
from typing import Literal
class CleaningAction(BaseModel):
    column:str
    action:Literal["impute_median","impute_mode","drop_rows","drop_column","none"]
    reason:str
class CleaningPlan(BaseModel):
    actions:list[CleaningAction]
    overall_reasoning:str


