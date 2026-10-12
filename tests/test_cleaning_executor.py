import pandas as pd 
from src.cleaning.cleaning_executer import execute_cleaning_plan
from src.cleaning.schema import CleaningAction,CleaningPlan
def make_test_df():
    return pd.DataFrame({
        "age": [25, 30, None, 40, 35],
        "city": ["Delhi", "Mumbai", None, "Delhi", "Delhi"],
        "score": [10, 20, 30, 40, 50],
    })
def test_impute_median():
    df = make_test_df()
    plan = CleaningPlan(
        actions=[
            CleaningAction(column="age", action="impute_median", reason="test")
        ],
        overall_reasoning="test"
    )
    result = execute_cleaning_plan(df,plan)
    assert result["age"].isna().sum() == 0
    assert result.loc[2, "age"] == 32.5
def test_impute_mode():
    df = make_test_df()
    plan = CleaningPlan(
        actions=[CleaningAction(column="city",action="impute_mode",reason = "test")],
        overall_reasoning="test"
    )
    result = execute_cleaning_plan(df,plan)
    assert result["city"].isna().sum() == 0
    assert result.loc[2, "city"] == "Delhi" 
def test_drop_rows_with_missing():
    df = make_test_df()
    plan = CleaningPlan(
        actions=[
            CleaningAction(column="age", action="drop_rows", reason="test")
        ],
        overall_reasoning="test"
    )
    result = execute_cleaning_plan(df, plan)
    assert len(result) == 4
    assert result["age"].isna().sum() == 0

def test_drop_column():
    df = make_test_df()
    plan = CleaningPlan(
        actions=[
            CleaningAction(column="city", action="drop_column", reason="test")
        ],
        overall_reasoning="test"
    )
    result = execute_cleaning_plan(df, plan)
    assert "city" not in result.columns
    assert len(result.columns) == 2

def test_drop_column():
    df = make_test_df()
    plan = CleaningPlan(
        actions=[
            CleaningAction(column="city", action="drop_column", reason="test")
        ],
        overall_reasoning="test"
    )
    result = execute_cleaning_plan(df, plan)
    assert "city" not in result.columns
    assert len(result.columns) == 2
def test_unknown_action_raises():
    df = make_test_df()
    pass
