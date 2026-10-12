import pandas as pd 
def impute_median(df:pd.DataFrame,column:str)->pd.DataFrame:
    median_val = df[column].median()
    df[column] = df[column].fillna(median_val)
    return df
def impute_mode(df:pd.DataFrame,column:str)->pd.DataFrame:
    mode_val = df[column].mode(dropna = True)
    if len(mode_val) == 0:
        return df
    df[column] = df[column].fillna(mode_val.iloc[0])
    return df
def drop_rows_with_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.dropna(subset=[column])
def drop_column(df:pd.DataFrame,column:str)->pd.DataFrame:
    return df.drop(columns=[column])
def execute_cleaning_plan(df:pd.DataFrame,plan)->pd.DataFrame:
    for action in plan.actions:
        if action.action == "impute_median":
            df = impute_median(df,action.column)
        elif action.action == "impute_mode":
            df = impute_mode(df,action.column)
        elif action.action == "drop_rows":
            df = drop_rows_with_missing(df,action.column)
        elif action.action == "drop_column":
            df = drop_column(df,action.column)
        elif action.action == "none":
            pass
        else:
            raise ValueError(f"Unknown cleaning action :{action.action}")
    return df 
