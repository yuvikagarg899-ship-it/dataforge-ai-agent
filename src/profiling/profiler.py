import numpy as np
import pandas as pd
from src.profiling.schema import ColumnProfile, DatasetProfile

HIGH_MISSING_THRESHOLD = 0.4
LIKELY_ID_UNIQUE_PCT = 0.95
NEAR_CONSTANT_UNIQUE_COUNT = 1
HIGH_CARDINALITY_THRESHOLD = 0.5
MIN_ROWS_FOR_ID_CHECK = 30
MIN_ROWS_FOR_CONSTANT_CHECK = 5


def profile_dataset(path: str) -> DatasetProfile:
    df = pd.read_csv(path)
    n_rows = len(df)
    column_profiles = []

    for col in df.columns:
        series = df[col]
        missing_pct = series.isna().mean()
        unique_count = series.nunique()
        unique_pct = unique_count / n_rows if n_rows > 0 else 0.0

        flags = []
        if missing_pct > HIGH_MISSING_THRESHOLD:
            flags.append("high_missingness")
        if  n_rows >= MIN_ROWS_FOR_CONSTANT_CHECK and unique_count == NEAR_CONSTANT_UNIQUE_COUNT:
            flags.append("near_constant")
        if n_rows >= MIN_ROWS_FOR_ID_CHECK and unique_pct > LIKELY_ID_UNIQUE_PCT:
            flags.append("likely_id")

        is_numeric = pd.api.types.is_numeric_dtype(series)

        mean_val = std_val = min_val = max_val = skew_val = None
        top_values = None

        if is_numeric:
            if series.notna().any():
                mean_val = float(series.mean())
                std_val = float(series.std())
                min_val = float(series.min())
                max_val = float(series.max())
            if series.notna().sum() > 2:
                skew_val = float(series.skew())
        else:
            value_counts = series.value_counts().head(5)
            top_values = {str(k): int(v) for k, v in value_counts.items()}

        col_profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            missing_pct=round(missing_pct, 4),
            unique_count=unique_count,
            unique_pct=round(unique_pct, 4),
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            skew=skew_val,
            top_values=top_values,
            flags=flags,
        )
        column_profiles.append(col_profile)

    dataset_flags = []
    if df.duplicated().sum() > 0:
        dataset_flags.append("has_duplicate_rows")

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=len(df.columns),
        columns=column_profiles,
        dataset_level_flags=dataset_flags,
    )