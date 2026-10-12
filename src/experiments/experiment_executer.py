from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.metrics import (accuracy_score,r2_score,f1_score,confusion_matrix,precision_score,recall_score,mean_absolute_error,mean_squared_error)
from src.experiments.schema import ExperimentPlan
from src.planning.schema import ParsedObjective
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from src.experiments.metrics import METRICS,canonical_metric
import numpy as np 
import pandas as pd
MODEL_CLASSES = {
    "LogisticRegression":LogisticRegression,
    "RandomForestClassifier":RandomForestClassifier,
    "RandomForestRegressor":RandomForestRegressor,
    "GradientBoostingRegressor":GradientBoostingRegressor,
    "GradientBoostingClassifier":GradientBoostingClassifier,
    "LinearRegression":LinearRegression
}
def cast_hyperparametric_value(value:str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower()== "false":
        return False
    return value
def build_preprocessor(X:pd.DataFrame)->ColumnTransformer:
    numerical_cols = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_cols = [
        c for c in X.columns
        if c not in numerical_cols
    ]
    transformers = []
    if numerical_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numerical_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute",SimpleImputer(strategy="constant",fill_value="Missing")),
                    ("onehot",OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols
            )
        )
    return ColumnTransformer(transformers)

def run_experiment(df,plan:ExperimentPlan,objective:ParsedObjective)->dict:
    metric_key = canonical_metric(objective.target_metric)
    if metric_key is None:
        raise ValueError(f"Unsupported metric:{objective.target_metric}")
    spec = METRICS[metric_key]
    X = df[plan.features]
    y = df[objective.target_column]
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    model_class = MODEL_CLASSES[plan.model_type]

    kwargs = {
    hp.name: cast_hyperparametric_value(hp.value)
    for hp in plan.hyperparameters
    }
    if "random_state" in model_class().get_params() and "random_state" not in kwargs:
        kwargs["random_state"] = 42
    model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("model", model_class(**kwargs)),
])
    model.fit(X_train,y_train)
    score = float(spec["fn"](y_test,model.predict(X_test)))
    if objective.task_type == "classification":
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train),1)),y_train)
    baseline_score = float(
        spec["fn"](y_test,baseline.predict(np.zeros((len(y_test),1))))    
        )
    if spec["higher_is_better"]:
        beats = score>baseline_score
    else:
        beats = score<baseline_score

    return {
        "model_type": plan.model_type,
        "metric_name": metric_key,
        "metric_value": score,
        "baseline_metric_value": baseline_score,
        "beats_baseline": bool(beats),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }