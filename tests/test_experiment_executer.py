import os 
from google import genai 
from src.profiling.profiler import profile_dataset
from src.planning.objective_parser import parse_objective
from src.cleaning.cleaning_planner import plan_cleaning
from src.cleaning.cleaning_executer import execute_cleaning_plan
from src.experiments.experiment_planner import plan_experiment
from src.experiments.experiment_executer import run_experiment
import pandas as pd 
if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    csv_path = "data/Social_Network_Ads (1).csv"

    profile = profile_dataset(csv_path)
    objective = parse_objective("predict whether someone will purchase,I care most about not missing actual buyers",profile,client)
    cleaning_plan = plan_cleaning(profile,objective,client)
    df = pd.read_csv(csv_path)
    if cleaning_plan.actions:
        df = execute_cleaning_plan(df,cleaning_plan)
    experiment_plan = plan_experiment(profile,objective,client)
    results = run_experiment(df,experiment_plan,objective)
    print("\nParsed Objective")
    print(objective.model_dump_json(indent=2))

    print("\nCleaning Plan")
    print(cleaning_plan.model_dump_json(indent=2))

    print("\nExperiment Plan")
    print(experiment_plan.model_dump_json(indent=2))

    print("\nExperiment Results")
    print(results)