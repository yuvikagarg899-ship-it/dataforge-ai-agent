from typing import Optional,TypedDict
import pandas as pd 
from src.profiling.schema import DatasetProfile
from src.planning.schema import ParsedObjective
from src.cleaning.schema import CleaningPlan
from src.experiments.schema import ExperimentPlan
from src.reflection.schema import ReflectionResult
from src.critique.schema import CritiqueResult
class PipelineState(TypedDict):
    csv_path:str
    user_objective:str
    profile:Optional[DatasetProfile]
    parsed_objective:Optional[ParsedObjective]
    cleaning_plan :Optional[CleaningPlan]
    cleaned_df:Optional[pd.DataFrame]
    experiment_plan:Optional[ExperimentPlan]
    experiment_results:Optional[dict]
    reflection:Optional[ReflectionResult]
    iteration_count:int
    accepted:bool
    error_message:Optional[str]
    critique:Optional[CritiqueResult]
    critique_iteration_count:int
    trace:list[dict]
    