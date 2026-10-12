import os 
from google import genai
from src.profiling.profiler import profile_dataset
from src.experiments.experiment_planner import plan_experiment
from src.planning.objective_parser import parse_objective
if __name__== "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    profile = profile_dataset("data/Social_Network_Ads (1).csv")
    objective = parse_objective("predict whether someone will purchase,I care most about not missing actual buyers",profile,client)
    plan = plan_experiment(profile,objective,client)
    print(plan.model_dump_json(indent = 2))
