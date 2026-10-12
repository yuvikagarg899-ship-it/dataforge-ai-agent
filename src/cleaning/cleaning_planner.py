import os 
from google import genai 
from src.planning.schema import ParsedObjective
from src.profiling.schema import DatasetProfile
from src.cleaning.schema import CleaningPlan
def is_numeric_dtype(dtype:str)->bool:
     """Small helper — profile dtypes are strings like 'int64', 'float64', 'str'."""
     return "int" in dtype or "float" in dtype
def plan_cleaning(profile:DatasetProfile,objective:ParsedObjective,client:genai.Client)->CleaningPlan:
     column_summary = "\n".join(
        f"- {col.name} (dtype={col.dtype}, missing_pct={col.missing_pct}, "
        f"unique_pct={col.unique_pct}, flags={col.flags})"
        for col in profile.columns
     )
     prompt = f"""You are planning data cleaning steps for a machine learning task.

The dataset has the following columns. Use ONLY these exact names for the `column` field
in any cleaning action:
{column_summary}

Separately, here are dataset-wide observations. These are NOT column names and must never
be used as a `column` value:
Dataset-level flags: {profile.dataset_level_flags}

The task is: {objective.task_type}
The target column is: "{objective.target_column}"

Propose ONLY cleaning actions that are justified by actual evidence in the profile above
(e.g. only propose imputation on columns with missing_pct > 0, only propose dropping a
column if there is a real flag or stat justifying it).

Be conservative with the target column ("{objective.target_column}") — do not propose
actions on it unless there is strong, explicit justification (e.g. it has missing values).

If no cleaning is needed at all, return an empty actions list and explain why in
overall_reasoning.
"""
     response = client.models.generate_content(
          model = "gemini-3.5-flash-lite",
          contents = prompt,
          config = {
               "response_mime_type":"application/json",
               "response_schema":CleaningPlan,
          },
     )
     plan = CleaningPlan.model_validate_json(response.text)
     col_lookup = {col.name:col for col in profile.columns}
     for action in plan.actions:
          if action.column not in col_lookup:
               raise ValueError(f"Cleaning action reference unknown column '{action.column}'."
                                f"Valid columns are :{sorted(col_lookup.keys())}")
          col_profile = col_lookup[action.column]
          is_numeric = is_numeric_dtype(col_profile.dtype)
          if action.action == "impute_median" and not is_numeric:
               raise ValueError(f"Proposed impute_median on non numeric columns '{action.column}'"
                                f"Datatype = {col_profile.dtype}")
          if action.action in ("impute_median","impute_mode") and col_profile.missing_pct==0.0:
               raise ValueError(
                    f"Proposed {action.action} on '{action.column}' but it has 0% missing "
                    f"Values no basis for imputations,Reason given: '{action.reason}'"
               )
          if action.action == "drop_column" and not col_profile.flags:
               raise ValueError(
                    f"Proposed drop_column on '{action.column}' but it has no flags"
                    f"(no near_constant/likely_id/high_missingness signal) — no basis for dropping. "
                    f"Reason given : '{action.reason}'"
               )
     return plan

     