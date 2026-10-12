import pandas as pd 
from src.experiments.schema import ExperimentPlan,Hyperparameters
from src.experiments.experiment_executer import run_experiment
from src.planning.schema import ParsedObjective
df = pd.read_csv("data/house_prices_train.csv")
df = df.drop(columns=["Id", "Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"])
features = [c for c in df.columns if c != "SalePrice"]
plan = ExperimentPlan(
    model_type="GradientBoostingRegressor",
    features=features,
    hyperparameters=[
        Hyperparameters(name = "n_estimators",value = "200"),
        Hyperparameters(name = "learning_rate",value = "0.05"),
        Hyperparameters(name = "max_depth",value = "4"),
    ],
    reasoning="noise_check",
)
objective = ParsedObjective(
    task_type="regression",target_column="SalePrice",target_metric="rmse",
    is_satisfiable=True,reasoning="noise_check",constraints=None,
)
for i in range(3):
    print(run_experiment(df,plan,objective)["metric_value"])