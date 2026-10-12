import os 
from google import genai 
from src.memory.store import init_db
from src.graph.graph_builder import build_graph
if __name__ == "__main__":
    init_db()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    graph = build_graph()
    initial_state = {
        "csv_path":"data/house_prices_train.csv",
        "user_objective":"predict the sale price of a house,I want accurate predictions and I especially want to avoid big mistakes on expensive homes",
        "profile":None,
        "parsed_objective":None,
        "cleaning_plan":None,
        "cleaned_df":None,
        "experiment_plan":None,
        "critique":None,
        "experiment_results":None,
        "reflection":None,
        "iteration_count":0,
        "critique_iteration_count":0,
        "accepted":False,
        "error_message":None,
        "trace":[],
    }
    final_state = graph.invoke(
        initial_state,
        config = {"configurable":{"client":client}}
    )

    print("accepted:",final_state["accepted"])
    print("error_message:",final_state["error_message"])
    obj = final_state.get("parsed_objective")
    if obj:
        print("\nobjective:",obj.task_type,obj.target_column,obj.target_metric)
    plan = final_state.get("cleaning_plan")
    if plan:
        print("\ncleaning actions :",[(a.column,a.action) for a in plan.actions])
    exp = final_state.get("experiment_plan")
    if exp:
        print("\nexperiment:",exp.model_type,exp.features)
        print("Hyperparameters:",[(h.name,h.value) for h in exp.hyperparameters])
    if final_state.get("experiment_results"):
        print("\nresults:",final_state["experiment_results"])
    print("\n----Trace---\n")
    for e in final_state["trace"]:
        print(f'{e["at"]}{e["node"]:28}{e["summary"]}')
    refl = final_state.get("reflection")
    if refl:
        print("\nreflection:", refl.decision, "-", refl.reasoning)