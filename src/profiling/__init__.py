from pydantic import BaseModel
from typing import Literal,Optional
class ColumnProfile(BaseModel):
    name:str
    dtype:str
    missing_pct:int
    unique_pct:int 
    unique_count:int

    mean:Optional[float] = None
    min:Optional[float] = None
    max:Optional[float] = None
    std:Optional[float] = None
    skew:Optional[float] = None

    top_values:Optional[dict[str,int]] = None
    flags: list[str] = []
class Dataset_profile(BaseModel):
    n_rows :int
    n_columns:int
    columns:list[ColumnProfile]
    dataset_level_flags:list[str] = None



