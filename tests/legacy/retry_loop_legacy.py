import os
from unittest.mock import patch
from google import genai
from src.pipeline import run_pipeline
from src.reflection.schema import ReflectionResult
from src.reflection.reflector import reflect_on_Result
from src.graph.graph_builder import build_graph

if __name__  == "__main__":
    client = genai.Client(api_key = os.environ["GEMINI_API_KEY"])
    call_count = {"n":0}
    def fake_reflect(objective,results,client):
        call_count["n"]+=1
        if call_count["n"] == 1:
            return ReflectionResult(
                decision="retry",
                reasoning= "Forced retry for testing",
                suggested_changes="Try a different model type."
            )
        return ReflectionResult(decision="accept",reasoning = "Accepting on retry")
    graph = build_graph()
    initial_state = {
        "csv_path": "data/Social_Network_Ads (1).csv",
        "user_objective": "predict whether someone will purchase, I care most about not missing actual buyers",
        "profile": None,
        "parsed_objective": None,
        "cleaning_plan": None,
        "cleaned_df": None,
        "experiment_plan": None,
        "experiment_results": None,
        "reflection": None,
        "iteration_count": 0,
        "accepted": False,
        "error_message": None,
    }
    with patch("src.graph.nodes.reflect_on_Result", side_effect=fake_reflect):
        final_state = graph.invoke(
            initial_state,
            config = {"configurable":{"client":client}}
        )
    print("accepted : ",final_state["accepted"])
    print("iteration_count : ",final_state["iteration_count"])
    print("number of reflect on Result calls :",call_count["n"])
    print("\n Final experiment plan")
    print(final_state["experiment_plan"].model_dump_json(indent = 2))