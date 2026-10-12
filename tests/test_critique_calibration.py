import os
from google import genai
from src.profiling.profiler import profile_dataset
from src.planning.schema import ParsedObjective
from src.experiments.schema import ExperimentPlan, Hyperparameters
from src.critique.critic import critique_experiment
if __name__ == "__main__":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    profile = profile_dataset("data/Social_Network_Ads (1).csv")

    objective = ParsedObjective(
        task_type="classification",
        target_column="Purchased",
        target_metric="recall",
        is_satisfiable=True,
        reasoning="Binary purchase prediction, recall matters most.",
        constraints=None,
    )
    good_plan = ExperimentPlan(
        model_type = "RandomForestClassifier",
        features = ["Age","EstimatedSalary"],
        hyperparameters=[
            Hyperparameters(name = "n_estimators",value = "150"),
            Hyperparameters(name = "class_weight",value = "balanced")
        ],
        reasoning=(
    "Random forest handles non-linear boundaries on two numeric features, "
    "and class_weight='balanced' helps recall on the minority class."
),

    )
    bad_plan = ExperimentPlan(
        model_type="GradientBoostingClassifier",
        features = ["Age","EstimatedSalary"],
        hyperparameters=[
            Hyperparameters(name = "n_estimators",value = "3000"),
            Hyperparameters(name = "learning_rate",value = "1.0"),
            Hyperparameters(name = "max_depth",value = "25"),
        ],
        reasoning = "Gradient Boosting is generally a strong model",

    )
    for name,plan in [("GOOD PLAN",good_plan),("BAD PLAN",bad_plan)]:
        result = critique_experiment(profile,objective,plan,client)
        print(f"\n ---{name}----")
        print("verdict",result.verdict)
        print("concerns",result.concerns)