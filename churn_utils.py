"""Validation and inference utilities for the saved Telco churn model."""
from __future__ import annotations

import numpy as np
import pandas as pd

NONNEGATIVE_FIELDS = {"Tenure Months", "Monthly Charges", "Total Charges", "CLTV"}
INTEGER_FIELDS = {"Tenure Months"}


def validate_and_prepare(original: pd.DataFrame, artifact: dict):
    """Return (cleaned predictors, issue table, row flags, blocking errors).

    Invalid individual values become NaN for the fitted model imputer.
    Unrecognized categorical values are NOT silently one-hot ignored.
    Missing entire columns block scoring.
    """
    expected = list(artifact["feature_columns"])
    numeric = list(artifact["numeric_features"])
    categoricals = list(artifact["categorical_features"])
    absent = [c for c in expected if c not in original.columns]
    problems = []
    blocking = []
    if original.empty:
        blocking.append("The CSV contains no customer records.")
    if absent:
        blocking.append("Required predictor columns missing: " + ", ".join(absent))
    if original.columns.duplicated().any():
        blocking.append("Duplicate column names detected. Rename them before uploading.")
    if blocking:
        return None, pd.DataFrame(columns=["Column", "Problem", "Rows", "Handling"]), None, blocking

    x = original[expected].copy()
    flags = pd.DataFrame(index=original.index)
    flags["Missing Inputs"] = 0
    flags["Invalid Inputs"] = 0
    flags["Affected Fields"] = ""

    onehot = artifact["preprocessor"].named_transformers_["categorical"].named_steps["onehot"]
    known = dict(zip(categoricals, onehot.categories_))

    def register(column: str, mask: pd.Series, kind: str, action: str):
        mask = mask.fillna(False).astype(bool)
        count = int(mask.sum())
        if count == 0:
            return
        problems.append({"Column": column, "Problem": kind, "Rows": count, "Handling": action})
        flag_col = "Missing Inputs" if kind == "Missing" else "Invalid Inputs"
        flags.loc[mask, flag_col] += 1
        flags.loc[mask, "Affected Fields"] = flags.loc[mask, "Affected Fields"].apply(
            lambda old: f"{old}, {column}" if old else column
        )

    for col in numeric:
        raw = x[col]
        # Empty strings and whitespace are treated as missing, rather than invalid.
        blanks = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        converted = pd.to_numeric(raw.where(~blanks), errors="coerce")
        invalid = (~blanks) & converted.isna()
        if col in NONNEGATIVE_FIELDS:
            invalid |= converted.lt(0).fillna(False)
        if col in INTEGER_FIELDS:
            invalid |= (converted.notna() & (converted % 1 != 0))
        register(col, blanks, "Missing", "Median from training data")
        register(col, invalid, "Invalid", "Set to missing; training median")
        converted.loc[invalid | blanks] = np.nan
        x[col] = converted.astype(float)

    for col in categoricals:
        raw = x[col]
        blank = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        # Model trained on string categories; clean only surrounding whitespace.
        normalized = raw.astype("string").str.strip()
        unknown = (~blank) & (~normalized.isin(known[col]))
        register(col, blank, "Missing", "Training mode (most frequent)")
        register(col, unknown, "Invalid", "Set to missing; training mode")
        normalized.loc[blank | unknown] = pd.NA
        x[col] = normalized.astype(object).where(normalized.notna(), np.nan)

    flags["Data Quality Flag"] = np.select(
        [(flags["Missing Inputs"] > 0) & (flags["Invalid Inputs"] > 0),
         flags["Invalid Inputs"] > 0, flags["Missing Inputs"] > 0],
        ["Missing and invalid", "Invalid inputs imputed", "Missing inputs imputed"],
        default="Complete",
    )
    return x, pd.DataFrame(problems, columns=["Column", "Problem", "Rows", "Handling"]), flags, []


def predict_customers(original, x, flags, artifact):
    transformed = artifact["preprocessor"].transform(x)
    clusters = artifact["kmeans"].predict(transformed)
    matrix = np.column_stack([transformed, np.eye(artifact["cluster_count"])[clusters]])
    probs = artifact["classifier"].predict_proba(matrix)[:, 1]
    result = original.copy().reset_index(drop=True)
    result["Customer Segment"] = pd.Series(clusters + 1).map(lambda k: f"Cluster {k}")
    result["Churn Probability"] = probs
    for col in flags.columns:
        result[col] = flags[col].reset_index(drop=True)
    return result


def apply_threshold(results, threshold: float, low_cutoff: float = 0.30):
    out = results.copy()
    high = out["Churn Probability"] >= threshold
    out["Predicted Churn Value"] = high.astype(int)
    out["Predicted Churn"] = np.where(high, "Yes", "No")
    # Three tier categories: High >= threshold, Low < 0.3 (or adjusted if threshold lower).
    low = min(low_cutoff, threshold / 2)
    out["Risk Category"] = np.select(
        [high, out["Churn Probability"] < low], ["High", "Low"], default="Medium"
    )
    if "CLTV" in out.columns:
        cltv = pd.to_numeric(out["CLTV"], errors="coerce").clip(lower=0)
        out["Expected Value at Risk (proxy)"] = out["Churn Probability"] * cltv
    return out


def segment_summary(results: pd.DataFrame):
    return (results.groupby("Customer Segment", dropna=False)
            .agg(Customers=("Churn Probability", "size"),
                 **{"Average Churn Probability": ("Churn Probability", "mean"),
                    "Predicted High Risk": ("Predicted Churn Value", "sum")})
            .reset_index().sort_values("Customer Segment"))
