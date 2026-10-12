from pydantic import BaseModel 
from typing import Literal,Optional
class ParsedObjective(BaseModel):
    task_type : Literal["classification","regression"]
    target_column:Optional[str] = None
    target_metric:Optional[str] = None
    is_satisfiable:bool
    reasoning:str
    constraints:Optional[str]=None