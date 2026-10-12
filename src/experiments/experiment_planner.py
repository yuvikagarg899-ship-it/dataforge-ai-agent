import os 
from google import genai
from typing import Optional
from src.profiling.schema import DatasetProfile
from src.planning.schema import ParsedObjective
from src.experiments.schema import ExperimentPlan
VALID_MODELS = {
    "classification":{
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    },
    "regression":{
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor"
    },
}
def plan_experiment(profile:DatasetProfile,objective:ParsedObjective,client:genai.Client,prior_experiments:Optional[dict] = None,critique_feedback:Optional[list[str]] = None)->ExperimentPlan:
    column_summary = "\n".join(
        f"- {col.name} (dtype={col.dtype}, missing_pct={col.missing_pct}, "
        f"unique_pct={col.unique_pct}, flags={col.flags})"
        for col in profile.columns
    )
    valid_model_names = VALID_MODELS[objective.task_type]
    prior_experiments_section = ""
    if prior_experiments:
        lines = []
        for exp in prior_experiments:
            status = "ACCEPTED" if exp["accepted"] else "REJECTED"
            lines.append(
                f"-[{status}] model = {exp['model_type']},features = {exp['features']},"
                f"{exp['metric_name']} = {exp['metric_value']:.4f}"
                f"(baseline was {exp['baseline_metric_value']:.4f})"
            )
        prior_experiments_section = (
            "\nRelevant past experiments on this exact dataset and objective:\n"
            + "\n".join(lines)
            + "\n\nDo not blindly repeat a REJECTED configuration. You may build on an "
              "ACCEPTED one if you believe you can do better.\n"
        )
    critique_section = ""
    if critique_feedback:
        critique_section = (
            "\nAn Independent reviewer rejected your previous proposal for this reason :\n"
            +"\n".join(f" - {c}" for c in critique_feedback)
            +"\n\n Address these specific concerns in your new proposal.\n"
        )

    prompt = f""" You are planning a machine learning Experiment .
Dataset Columns:{column_summary}
Task Type:{objective.task_type}
Target column:{objective.target_column}
Target metric:{objective.target_metric}
{prior_experiments_section}
Propose ONE experiment:a model to train ,which columns to use as features,and any hyperparameters worth setting explicitly.
Rules:
-model type must be one of :{sorted(valid_model_names)}
-features must be existing column names from the list above,and MUST not include the target column"{objective.target_column}"(including it would cause data leakage),
-hyperparameters must be a list of {{"name": ..., "value": ...}} pairs, where value is
  always given as a string (e.g. "100", "0.1", "lbfgs"). Only include hyperparameters you
  want to override — omit anything you want left at scikit-learn's default.
-Explain your reasoning for the model choice given the data characterstics and objective 
-The system automatically imputes missing values(median for numeric columns ,most frequent for others) and one hot encodes non - numeric columns ,so you may use categorical columns and columns that contain missing values,
-Do not use identifier columns (anything flagged likely_id or named Id) as features.
"""
    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = prompt,
        config = {
            "response_mime_type":"application/json",
            "response_schema":ExperimentPlan,
        },
    )
    plan = ExperimentPlan.model_validate_json(response.text)
    if plan.model_type not in valid_model_names:
        raise ValueError(
            f"Model '{plan.model_type}' is invalid or mismatched for task "
            f"'{objective.task_type}'. Valid options: {sorted(valid_model_names)}"
        )
    valid_columns = {col.name for col in profile.columns}
    for feat in plan.features:
        if feat not in valid_columns:
            raise ValueError(
                f"Features :'{feat}'does not exist in the dataset"
            )
        if feat == objective.target_column:
            raise ValueError(
                f"Feature list includes the target column '{feat}' — this is data leakage"
            )

    if len(plan.features) == 0:
        raise ValueError(
            "Experiment plan proposes zero features - nothing to train on"
        )
    return plan

