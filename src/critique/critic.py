from google import genai 
from src.critique.schema import CritiqueResult
def critique_experiment(profile,objective,experiment_plan,client:genai.Client,)->CritiqueResult:
    column_summary = "\n".join(
        f"- {col.name} (dtype={col.dtype}, missing_pct={col.missing_pct}, "
        f"unique_pct={col.unique_pct}, flags={col.flags})"
        for col in profile.columns
    )
    hyperparams_summary = ",".join(f"{hp.name} = {hp.value}" for hp in experiment_plan.hyperparameters) or "{none set,using scikit learn defaults}"
    prompt = f"""You are an independent , skeptical reviewer of a  proposed machine learning experiment , your job is to find genuine reasoning flaws before any compute is spent training this model - not to rubber stamp it .
    Dataset columns :{column_summary}
Task Type :{objective.task_type}
Target Column :{objective.target_column}
Target metric:{objective.target_metric}
Proposed experiment 
-Model:{experiment_plan.model_type}
-Features:{experiment_plan.features}
-Hyperparams:{hyperparams_summary}
-Planner's started reasoning :{experiment_plan.reasoning}
Critically evaluate this plan and consider things like:
-Does the model choice actually fit the data size and characterstics ?
-Do the hyperparameters(or lack thereof) create real risk (e.g., overfitting, ignoring
  known class imbalance, an inappropriate default)?
-Does the planners stated reasoning actually hold up or is it unjustified/generic?
Scope: review ONLY the model choice, features, and hyperparameters. The system itself
handles the following, so do not raise concerns about them: an 80/20 train/test split
with a fixed random seed, scikit-learn defaults for anything not listed, and the choice
of evaluation metric. If you approve the plan, return an empty concerns list.
IMPORTANT : Rejecting this plan has a real cost -it consumes another planning iteration and delays running any actual experiment , Only reject if the flaws is specific,consequential,and genuinely worth that delay to fix,Do not reject for stylistic preferences ,hypothetical improvements with no real downsides ,or differences from what you personally woould have chosen.If you are unsure whether a concern is serious enough to justify a rejection , approve the plan instead.
"""
    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = prompt,
        config = {
            "response_mime_type":"application/json",
            "response_schema":CritiqueResult,
        },
    )
    return CritiqueResult.model_validate_json(response.text)