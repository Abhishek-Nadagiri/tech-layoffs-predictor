"""
utils/preprocessing.py
Shared data preprocessing utilities used by both train_models.py and app.py.
"""

import pandas as pd
import numpy as np

CAT_COLS = [
    "industry", "country", "company_size", "hiring_trend",
    "reason_for_layoffs", "market_condition", "month", "top_hiring_role",
]

NUMERIC_COLS = [
    "year", "layoffs_count", "layoff_percentage",
    "ai_automation_impact", "ai_replacement_risk", "open_roles",
    "remote_jobs_percentage", "stock_growth_percent",
    "revenue_growth_percent", "salary_budget_change",
    "ai_adoption_level", "employee_sentiment", "job_security_score",
]

CLUSTER_FEATURES = [
    "layoffs_count", "layoff_percentage", "ai_automation_impact",
    "ai_replacement_risk", "open_roles", "stock_growth_percent",
    "revenue_growth_percent", "employee_sentiment", "job_security_score",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and run minimal type checks."""
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    return df


def encode_dataframe(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Append encoded columns for all categorical fields."""
    df_enc = df.copy()
    for col in CAT_COLS:
        if col in df_enc.columns and col in encoders:
            le = encoders[col]
            df_enc[col + "_enc"] = le.transform(df_enc[col].astype(str))
    return df_enc


def encode_single_row(row_dict: dict, feature_list: list, encoders: dict):
    """
    Convert a raw row dict (with string categoricals) into a numpy array
    aligned to `feature_list` for model inference.
    """
    import numpy as np
    cat_map = {c: c + "_enc" for c in CAT_COLS}
    row = dict(row_dict)
    for raw, encoded in cat_map.items():
        le = encoders.get(raw)
        if le and raw in row:
            try:
                row[encoded] = le.transform([str(row[raw])])[0]
            except ValueError:
                row[encoded] = 0
    return np.array([row.get(f, 0) for f in feature_list]).reshape(1, -1)


def risk_label(score: float) -> str:
    """Map a numeric ai_replacement_risk to Low/Medium/High."""
    if score <= 4:
        return "Low"
    elif score <= 7:
        return "Medium"
    return "High"
