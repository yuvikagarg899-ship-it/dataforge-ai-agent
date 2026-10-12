from google import genai

from src.profiling.schema import DatasetProfile
from src.planning.schema import ParsedObjective
from src.experiments.metrics import canonical_metric,METRICS


def parse_objective(
    user_objective: str,
    profile: DatasetProfile,
    client: genai.Client
) -> ParsedObjective:

    column_summary = "\n".join(
        f"- {col.name} ({col.dtype}, missing={col.missing_pct}, flags={col.flags})"
        for col in profile.columns
    )

    prompt = f"""
You are helping in planning a machine learning task.

Dataset columns:
{column_summary}

User's Objective:
"{user_objective}"

Determine the task type, target column, and appropriate evaluation metric.

The target column must be one of the exact column names listed above.
target_metric MUST be exactly one of: {sorted(METRICS)}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ParsedObjective,
        },
    )

    parsed = ParsedObjective.model_validate_json(response.text)
    if not parsed.is_satisfiable:
        raise ValueError(f"Objective cannot be satisfied with this dataset. LLM reasoning: {parsed.reasoning}")

    valid_columns = {col.name for col in profile.columns}

    if parsed.target_column not in valid_columns:
        raise ValueError(
            f"LLM proposed target column = '{parsed.target_column}' "
            f"which does not exist. "
            f"In the dataset, valid columns are: {sorted(valid_columns)}"
        )
    canonical = canonical_metric(parsed.target_metric)
    if canonical is None:
        raise ValueError(
            f"Unsupported metric '{parsed.target_metric}'."
            f"Supported:{sorted(METRICS)}"
        )
    if METRICS[canonical]["task"]!=parsed.task_type:
        raise ValueError(
            f"Metric '{canonical}' does not fit a {parsed.task_type} task"
        )
    parsed.target_metric = canonical

    return parsed