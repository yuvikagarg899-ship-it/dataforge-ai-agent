
import os

from google import genai
from dotenv import load_dotenv

from src.graph.graph_builder import build_graph

load_dotenv(override=True)


if __name__ == "__main__":

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        vertexai=False
    )

    graph = build_graph()

    initial_state = {
        "csv_path": "data/house_prices_train.csv",
        "user_objective": "predict house sale prices, I care most about minimizing prediction error",
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

    try:
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"client": client}}
        )

    except Exception as e:
        print("GRAPH INVOKE FAILED:", repr(e))
        raise

    print("accepted:", final_state["accepted"])
    print("iteration_count:", final_state["iteration_count"])
    print("error_message:", final_state["error_message"])

    if final_state.get("parsed_objective"):
        print("\nParsed objective")
        print(
            final_state["parsed_objective"].model_dump_json(indent=2)
        )

    if final_state.get("cleaning_plan"):
        print("\nCleaning Plan")
        print(
            final_state["cleaning_plan"].model_dump_json(indent=2)
        )

    if final_state.get("experiment_plan"):
        print("\nExperiment Plan")
        print(
            final_state["experiment_plan"].model_dump_json(indent=2)
        )

    if final_state.get("experiment_results"):
        print("\nExperiment Results")
        print(final_state["experiment_results"])

    if final_state.get("reflection"):
        print("\nReflection")
        print(
            final_state["reflection"].model_dump_json(indent=2)
        )
print("\n--- Trace ---")
for e in final_state["trace"]:
    print(f'{e["at"]}  {e["node"]:28} {e["summary"]}')
