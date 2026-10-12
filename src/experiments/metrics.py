from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score,mean_squared_error,mean_absolute_error,r2_score)
METRICS = {
    "accuracy":  {"fn": accuracy_score,  "higher_is_better": True,  "task": "classification"},
    "recall":    {"fn": recall_score,    "higher_is_better": True,  "task": "classification"},
    "precision": {"fn": precision_score, "higher_is_better": True,  "task": "classification"},
    "f1":        {"fn": f1_score,        "higher_is_better": True,  "task": "classification"},
    "rmse": {"fn": lambda y, p: mean_squared_error(y, p) ** 0.5,
             "higher_is_better": False, "task": "regression"},
    "mae":       {"fn": mean_absolute_error, "higher_is_better": False, "task": "regression"},
    "r2":        {"fn": r2_score,        "higher_is_better": True,  "task": "regression"},
}
ALIASES = {
    "root mean squared error": "rmse",
    "root mean square error": "rmse",
    "mean absolute error": "mae",
    "r2_score": "r2", "r-squared": "r2", "r squared": "r2",
    "f1_score": "f1", "f1-score": "f1", "f1 score": "f1",
}
def canonical_metric(name:str):
    key = name.strip().lower()
    key = ALIASES.get(key,key)
    return key if key in METRICS else None