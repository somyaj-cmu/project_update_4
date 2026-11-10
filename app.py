import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.write("Upload a customer CSV to assign four customer segments and estimate churn probability.")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifact = load_model()
except FileNotFoundError:
    st.error("model.pkl must be in the same folder as app.py.")
    st.stop()

uploaded = st.file_uploader("Upload customer data (CSV)", type=["csv"])
if uploaded is not None:
    try:
        original = pd.read_csv(uploaded)
        st.subheader("Uploaded data")
        st.write(f"{len(original):,} customers")
        st.dataframe(original.head(10), use_container_width=True)
        if st.button("Predict customer churn", type="primary"):
            model_input = original.drop(columns=[c for c in artifact["excluded_columns"] if c in original.columns], errors="ignore").copy()
            if "Total Charges" in model_input.columns:
                model_input["Total Charges"] = pd.to_numeric(model_input["Total Charges"], errors="coerce")
            model_input = model_input.reindex(columns=artifact["feature_columns"])
            transformed = artifact["preprocessor"].transform(model_input)
            clusters = artifact["kmeans"].predict(transformed)
            matrix = np.column_stack([transformed, np.eye(artifact["cluster_count"])[clusters]])
            probabilities = artifact["classifier"].predict_proba(matrix)[:, 1]
            predictions = (probabilities >= artifact.get("threshold", 0.5)).astype(int)
            results = original.copy()
            results["Customer Segment"] = [f"Cluster {i+1}" for i in clusters]
            results["Churn Probability"] = probabilities
            results["Predicted Churn Value"] = predictions
            results["Predicted Churn"] = np.where(predictions == 1, "Yes", "No")
            a,b,c = st.columns(3)
            a.metric("Customers analyzed", f"{len(results):,}")
            b.metric("Predicted to churn", f"{predictions.sum():,}")
            c.metric("Average churn probability", f"{probabilities.mean():.1%}")
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.subheader("Customer segment distribution")
            st.bar_chart(results["Customer Segment"].value_counts().sort_index())
            st.subheader("Predicted churn distribution")
            st.bar_chart(results["Predicted Churn"].value_counts())
            st.download_button("Download predictions as CSV", results.to_csv(index=False).encode("utf-8"), "customer_churn_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not process this CSV: {e}")
        st.info("Upload a CSV with the same predictor columns used to train the model.")
else:
    st.info("Upload a CSV file to get started.")
