import os
from unittest.mock import patch
from google import genai
from src.graph.graph_builder import build_graph
from src.critique.schema import CritiqueResult
from src.experiments.experiment_planner import plan_experiment as real_plan_experiment
from src.memory.store import init_db

if __name__ == "__main__":
    init_db()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    call_count = {"n": 0}

    def fake_critique(profile, objective, experiment_plan, client):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CritiqueResult(
                verdict="reject",
                reasoning="Forced rejection for testing.",
                concerns=[
                    "No class_weight is set despite class imbalance, which risks "
                    "missing actual buyers."
                ],
            )
        return CritiqueResult(
            verdict="approve",
            reasoning="Approving on second look.",
            concerns=[],
        )

    graph = build_graph()

    initial_state = {
        "csv_path": "data/Social_Network_Ads (1).csv",
        "user_objective": "predict whether someone will purchase, I care most about not missing actual buyers",
        "profile": None,
        "parsed_objective": None,
        "cleaning_plan": None,
        "cleaned_df": None,
        "experiment_plan": None,
        "critique": None,
        "experiment_results": None,
        "reflection": None,
        "iteration_count": 0,
        "critique_iteration_count": 0,
        "accepted": False,
        "error_message": None,
        "trace":[],
    }

    with patch("src.graph.nodes.critique_experiment", side_effect=fake_critique), \
         patch("src.graph.nodes.plan_experiment", wraps=real_plan_experiment) as spy_plan:
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"client": client}},
        )

    print("critique calls:", call_count["n"])
    print("accepted:", final_state["accepted"])
    print("error_message:", final_state["error_message"])

    for i, call in enumerate(spy_plan.call_args_list, start=1):
        print(f"plan_experiment call {i}: critique_feedback =", call.kwargs.get("critique_feedback"))