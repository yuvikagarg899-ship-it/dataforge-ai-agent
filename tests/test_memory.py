import os
from google import genai
from src.graph.graph_builder import build_graph
from src.memory.store import init_db
if __name__ == "__main__":
    init_db()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    graph = build_graph()

    def make_initial_state():
        return {
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
            "trace":[],
        }
    print("\n RUN1(First time seeing this dataset) ")
    result1 = graph.invoke(make_initial_state(),config = {"configurable":{"client":client}})
    print("accepted",result1["accepted"])
    print("experiment_plan:",result1["experiment_plan"].model_dump_json(indent = 2))
    print("\n RUN2(same dataset/objective - should see RUN1 in memory)")
    result2 = graph.invoke(make_initial_state(),config={"configurable":{"client":client}})
    print("accepted :",result2["accepted"])
    print("experiment_plan:",result2["experiment_plan"].model_dump_json(indent = 2)) 