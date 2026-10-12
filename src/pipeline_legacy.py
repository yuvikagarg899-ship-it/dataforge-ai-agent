from dataclasses import dataclass
from typing import Optional
import pandas as pd
from src.profiling.schema import DatasetProfile
from src.profiling.profiler import profile_dataset
from src.planning.objective_parser import parse_objective
from src.planning.schema import ParsedObjective
from src.cleaning.schema import CleaningPlan
from src.cleaning.cleaning_planner import plan_cleaning
from src.cleaning.cleaning_executer import execute_cleaning_plan
from src.experiments.schema import ExperimentPlan
from src.experiments.experiment_planner import plan_experiment
from src.experiments.experiment_executer import run_experiment
from src.reflection.schema import ReflectionResult
from src.reflection.reflector import reflect_on_Result
@dataclass 
class PipelineResult :
    success :bool 
    profile:Optional[DatasetProfile] = None
    parsed_objective:  Optional[ParsedObjective] = None
    cleaning_plan: Optional[CleaningPlan] = None
    cleaned_df: Optional[pd.DataFrame] = None
    experiment_plan:Optional[ExperimentPlan] = None
    experiment_results:Optional[dict] = None
    accepted:bool = False
    error_message:Optional[str] = None
    reflection:Optional[ReflectionResult] = None
MAX_ITERATIONS = 3

def run_pipeline(csv_path:str,user_objective:str,client)->PipelineResult:
    profile = profile_dataset(csv_path)
    try:
        parsed = parse_objective(user_objective,profile,client)
    except ValueError as e:
        return PipelineResult(success = False,profile = profile,error_message=str(e))
    try:
        cleaning_plan = plan_cleaning(profile, parsed, client)
    except ValueError as e:
        return PipelineResult(
            success=False, profile=profile, parsed_objective=parsed, error_message=str(e)
        )
    df = pd.read_csv(csv_path)

    if cleaning_plan.actions:
        df = execute_cleaning_plan(df, cleaning_plan)
    prior_attempt = None
    experiment_plan = None
    results = None
    reflection = None
    accepted = False
    for iteration in range(MAX_ITERATIONS):
        try:
            experiment_plan = plan_experiment(profile,parsed,client,prior_attempt)
        except ValueError as e:
                return PipelineResult(
                    success = False,
                    profile= profile,
                    parsed_objective=parsed,
                    cleaning_plan= cleaning_plan,
                    cleaned_df=df,
                    error_message=str(e),
                )
        try:
            results = run_experiment(df, experiment_plan, parsed)
        except Exception as e:
                return PipelineResult(
                    success=False,
                    profile=profile,
                    parsed_objective=parsed,
                    cleaning_plan=cleaning_plan,
                    cleaned_df=df,
                    experiment_plan=experiment_plan,
                    error_message=f"Training/evaluation failed: {e}",
                )
        reflection = reflect_on_Result(parsed,results,client)
        if reflection.decision == "accept":
            accepted = True
            break
        prior_attempt = {
             "experiment_plan":experiment_plan,
             "results":results,
             "reflection_suggestion":reflection.suggested_changes or reflection.reasoning

        }

        
    return PipelineResult(
        success=True,
        profile=profile,
        parsed_objective=parsed,
        cleaning_plan=cleaning_plan,
        cleaned_df=df,
        experiment_plan=experiment_plan,
        experiment_results=results,
        reflection=reflection,
        accepted=accepted
        )
        