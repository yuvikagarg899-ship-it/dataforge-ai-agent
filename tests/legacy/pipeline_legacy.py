import os 
from google import genai 
from src.pipeline import run_pipeline
if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result1 = run_pipeline("data/Social_Network_Ads (1).csv","predict whether someone will purchase, I care most about not missing actual buyers",client)
    print("success : ",result1.success)
    print("error_message : ",result1.error_message)
    if result1.success:
        print("\n parsed objective")
        print(result1.parsed_objective.model_dump_json(indent = 2))
        print("\n Cleaning plan ")
        print(result1.cleaning_plan.model_dump_json(indent = 2))
        print("\n First 5 rows (cleaned data)")
        print(result1.cleaned_df.head())
        print("\n Cleaned data shape")
        print(result1.cleaned_df.shape)
        print("\nExperiment plan")
        print(result1.experiment_plan.model_dump_json(indent = 2))
        print("\nExperiment results")
        print(result1.experiment_results)