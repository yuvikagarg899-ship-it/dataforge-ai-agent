import os 
import pytest 
from google import genai 
from src.profiling.profiler import profile_dataset
from src.planning.objective_parser import parse_objective
@pytest.fixture
def client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
@pytest.fixture
def profile():
    return profile_dataset("data/Social_Network_Ads (1).csv")
def test_parses_Recall_preferences_correctly(client,profile):
    result = parse_objective("predict whether someone will purchase, I care most about not missing actual buyers",
        profile, client)
    assert result.is_satisfiable is True
    assert result.target_column == "Purchased"
    assert result.task_type == "classification"
    assert result.target_metric.lower() == "recall"
def test_rejects_unsatisfiable_objective(client, profile):
    with pytest.raises(ValueError):
        parse_objective("predict the user's favorite color", profile, client)