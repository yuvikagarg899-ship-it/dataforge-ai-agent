from pydantic import BaseModel 
class Hyperparameters(BaseModel):
    name:str
    value:str
class ExperimentPlan(BaseModel):
    model_type:str
    features:list[str]
    hyperparameters:list[Hyperparameters]
    reasoning:str