import os 
from google import genai 
from unittest.mock import patch
from src.pipeline import run_pipeline
from src.reflection.reflector import reflect_on_Result
if __name__ == "__main__":
    client = genai.Client(api_key = os.environ["GEMINI_API_KEY"])
    result = run_pipeline(
        "data/Social_Network_Ads (1).csv",
        "predict whether someone will purchase, I care most about not missing actual buyers",
        client
    )
    print("success : ",result.success)
    print("error_message : ",result.error_message)
    if not result.success:
        raise SystemExit("Pipeline failed - nothing to reflect on ")
    print("Experiment Results : ",result.experiment_results)
    reflection = reflect_on_Result(result.parsed_objective,result.experiment_results,client)
    print(reflection.model_dump_json(indent = 2))