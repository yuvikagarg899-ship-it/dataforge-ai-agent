from src.profiling.profiler import profile_dataset
def test_profile_rpints_correctly():
    profile = profile_dataset("data/Social_Network_Ads (1).csv")
    assert profile.n_rows == 400
    assert profile.n_columns == 3
    assert "has_duplicate_rows" in profile.dataset_level_flags
    print(profile.model_dump_json(indent = 2))
def test_empty_columns_flagged_high_missing():
    profile = profile_dataset("data/empty_column.csv")
    col = get_columns(profile,"notes")
    assert col.missing_pct == 1.0
    assert "high_missingness" in col.flags
    assert col.mean is None
    assert col.std is None
def test_single_row_dataset_does_not_crash_and_has_no_false_flags():
    profile = profile_dataset("data/single_row.csv")
    assert profile.n_rows == 1
    for col in profile.columns:
        assert "near_constant" not in col.flags
        assert "likely_id" not in col.flags
def test_near_constant_flag_on_sufficient_rows():
    profile = profile_dataset("data/simple.csv")
    col = get_columns(profile, "country")
    assert "near_constant" in col.flags
    assert col.top_values == {"India": 6}
def test_likely_id_not_flagged_on_small_high_unique_dataset():
    profile = profile_dataset("data/small_high_unique_values.csv")
    col = get_columns(profile, "email")
    assert col.unique_pct == 1.0
    assert "likely_id" not in col.flags
def print_profile(path):
    print(f"\n{'='*50}\n{path}\n{'='*50}")
    profile = profile_dataset(path)
    print(profile.model_dump_json(indent=2))
def get_columns(profile,name):
    return next(c for c in profile.columns if c.name == name)
if __name__ == "__main__":
    print_profile("data/Social_Network_Ads (1).csv")
    print_profile("data/simple.csv")
    print_profile("data/single_row.csv")
    print_profile("data/small_high_unique_values.csv")
    print_profile("data/empty_column.csv")

