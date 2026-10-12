import os 
from google import genai 
from src.planning.objective_parser import parse_objective
from src.profiling.profiler import profile_dataset
from src.cleaning.cleaning_planner import plan_cleaning
if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    profile = profile_dataset("data/Social_Network_Ads (1).csv")
    objective = parse_objective(
        "predict whether someone will purchase, I care most about not missing actual buyers",profile,client
    )
    plan = plan_cleaning(profile,objective,client)
    print(plan.model_dump_json(indent = 2))