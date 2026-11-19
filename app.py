"""Run: streamlit run app.py (alongside trusted model.pkl)."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from churn_utils import validate_and_prepare, predict_customers, apply_threshold, segment_summary

st.set_page_config(page_title="Customer Churn & Retention Dashboard", page_icon="📊", layout="wide")
st.title("Customer Churn & Retention Dashboard")
st.caption("Predict churn, understand customer segments, and prioritize retention opportunities.")

MODEL_PATH = Path(__file__).with_name("model.pkl")

@st.cache_resource
def load_model():
    # SECURITY: Never load pickle files from users or other untrusted sources.
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file missing. Place the trusted `model.pkl` next to `app.py`.")
    st.stop()
except Exception:
    st.error("Unable to load the model. Check that the artifact and scikit-learn version are compatible.")
    st.stop()

with st.expander("Expected CSV columns / example file", expanded=False):
    st.write(", ".join(model["feature_columns"]))
    example = Path(__file__).with_name("test_data.csv")
    if example.exists():
        st.download_button("Download sample customer CSV", example.read_bytes(), "sample_customers.csv", "text/csv")
    st.caption("Extra columns (such as CustomerID) are preserved in downloads but not used to predict churn.")

upload = st.file_uploader("Upload customer data", type="csv")
if upload is None:
    st.info("Upload a CSV to run the data quality check and generate predictions.")
    st.stop()

try:
    original = pd.read_csv(upload)
except (pd.errors.ParserError, UnicodeError, ValueError, pd.errors.EmptyDataError):
    st.error("Unable to read this file. Please upload a valid UTF-8 CSV with column headers.")
    st.stop()

st.subheader("1. Data quality assessment")
st.write(f"Uploaded **{len(original):,} customers** with **{len(original.columns)} columns**.")
x, issues, flags, errors = validate_and_prepare(original, model)
if errors:
    for error in errors:
        st.error(error)
    st.stop()

rows_affected = int((flags["Data Quality Flag"] != "Complete").sum())
missing_cells = int(flags["Missing Inputs"].sum())
invalid_cells = int(flags["Invalid Inputs"].sum())
q1, q2, q3, q4 = st.columns(4)
q1.metric("Customer records", f"{len(original):,}")
q2.metric("Records affected", f"{rows_affected:,}")
q3.metric("Missing inputs", f"{missing_cells:,}")
q4.metric("Invalid inputs", f"{invalid_cells:,}")

if rows_affected:
    st.warning(f"{rows_affected:,} records contain missing or invalid predictor values. "
               "The model will impute them using statistics learned from its training data; "
               "predictions for affected records may be less reliable.")
    st.dataframe(issues, hide_index=True, use_container_width=True)
    with st.expander("Review records that will use imputation"):
        audit = original.copy()
        for col in flags.columns:
            audit[col] = flags[col].to_numpy()
        st.dataframe(audit.loc[flags["Data Quality Flag"] != "Complete"].head(200), use_container_width=True)
        st.download_button("Download full data-quality report", audit.to_csv(index=False).encode(),
                           "data_quality_report.csv", "text/csv")
else:
    st.success("No missing or invalid predictor values detected.")

# Predictions are computed once per upload; threshold/filter changes do not retrain or re-score the model.
# Validated features and model are deterministic. Cached data is not persisted between user sessions.
if st.button("Generate churn predictions", type="primary"):
    try:
        st.session_state["predictions"] = predict_customers(original, x, flags, model)
        st.session_state["file_id"] = (upload.name, upload.size, upload.getvalue())
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Prediction could not be completed. Review your CSV format. ({type(exc).__name__})")

file_id = (upload.name, upload.size, upload.getvalue())
if st.session_state.get("file_id") != file_id:
    st.info("Review the data quality report, then select **Generate churn predictions**.")
    st.stop()

st.subheader("2. Risk threshold and summary")
threshold = st.slider("Churn classification threshold", 0.10, 0.90,
                      float(model.get("threshold", 0.5)), step=0.025,
                      help="Customers at or above this estimated probability are labeled predicted churners. "
                           "Changing this cutoff does not change the model's predicted probabilities.")
results = apply_threshold(st.session_state["predictions"], threshold)
a, b, c, d = st.columns(4)
a.metric("Customers analyzed", f"{len(results):,}")
b.metric("Predicted to churn", f"{results['Predicted Churn Value'].sum():,}")
c.metric("Average predicted churn", f"{results['Churn Probability'].mean():.1%}")
d.metric("Affected by imputation", f"{rows_affected:,}")
st.caption("These are predictions, not observed churn rates. Threshold changes require labeled data for new accuracy/recall estimates.")

st.subheader("3. Segment-level analysis")
segment = segment_summary(results)
st.dataframe(segment.style.format({"Average Churn Probability": "{:.1%}"}),
             hide_index=True, use_container_width=True)
left, right = st.columns(2)
with left:
    st.write("Customers by segment")
    st.bar_chart(segment.set_index("Customer Segment")["Customers"])
with right:
    st.write("Average predicted churn probability by segment")
    st.bar_chart(segment.set_index("Customer Segment")["Average Churn Probability"])

with st.expander("Understand the segments"):
    st.write("Clusters are algorithm-generated groups, not automatically meaningful business segments. "
             "Use the table below to understand differences before assigning descriptive names.")
    profile_cols = [c for c in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
                    if c in x.columns]
    profile = x[profile_cols].copy()
    profile["Customer Segment"] = results["Customer Segment"].values
    st.dataframe(profile.groupby("Customer Segment")[profile_cols].mean().round(1), use_container_width=True)
    categorical_profile_cols = [c for c in ["Contract", "Internet Service", "Payment Method"] if c in x]
    for col in categorical_profile_cols:
        st.write(f"**{col} distribution within each segment**")
        dist = pd.crosstab(results["Customer Segment"], x[col], normalize="index")
        st.dataframe((dist * 100).round(1).astype(str) + "%", use_container_width=True)

st.subheader("4. Customer prioritization")
s1, s2, s3 = st.columns(3)
selected_risks = s1.multiselect("Risk category", ["High", "Medium", "Low"],
                               default=["High", "Medium", "Low"])
selected_segments = s2.multiselect("Segment", sorted(results["Customer Segment"].unique()),
                                   default=sorted(results["Customer Segment"].unique()))
sort_choice = s3.selectbox("Prioritize by", ["Churn probability", "Expected value at risk (proxy)"])
filtered = results[results["Risk Category"].isin(selected_risks) &
                   results["Customer Segment"].isin(selected_segments)].copy()
sort_col = "Churn Probability" if sort_choice == "Churn probability" else "Expected Value at Risk (proxy)"
if sort_col not in filtered.columns:
    st.info("CLTV not available; sorting by churn probability instead.")
    sort_col = "Churn Probability"
filtered = filtered.sort_values(sort_col, ascending=False)
col_order = ([c for c in ["CustomerID", "Customer Segment", "Churn Probability", "Risk Category",
                           "CLTV", "Expected Value at Risk (proxy)", "Data Quality Flag", "Affected Fields"]
              if c in filtered.columns])
other_cols = [c for c in filtered.columns if c not in col_order]
st.dataframe(filtered[col_order + other_cols], hide_index=True, use_container_width=True,
             column_config={"Churn Probability": st.column_config.ProgressColumn(
                 "Churn Probability", format="%.1f", min_value=0, max_value=1)})
st.caption("Expected value at risk = predicted churn probability × supplied CLTV. "
           "This is a prioritization proxy, not a causal estimate of preventable loss or retention ROI.")
st.download_button("Download filtered predictions (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                   "churn_predictions.csv", "text/csv", use_container_width=True)

st.subheader("5. Global model interpretation")
with st.expander("View logistic regression coefficients and caveats"):
    names = list(model["preprocessor"].get_feature_names_out()) + [
        f"Cluster {i+1} indicator" for i in range(model["cluster_count"])]
    coefs = model["classifier"].coef_.ravel()
    importance = pd.DataFrame({"Encoded feature": names, "Coefficient": coefs})
    importance["Absolute coefficient"] = importance["Coefficient"].abs()
    importance = importance.sort_values("Absolute coefficient", ascending=False)
    st.dataframe(importance.head(20), hide_index=True, use_container_width=True)
    st.caption("Positive coefficients are associated with higher predicted churn log-odds, "
               "holding other encoded inputs constant. These are associations, not causal effects. "
               "One-hot categories and correlated cluster indicators complicate interpretation.")
