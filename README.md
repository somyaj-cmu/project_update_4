# Improved Telco Churn & Retention App

## Quick start

Requires Python 3.11+ with a version compatible with scikit-learn 1.8.0.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `model.pkl`, `app.py`, `churn_utils.py`, and `test_data.csv` in the same directory. The bundled `model.pkl` is a copy of the classmate's saved artifact, not a newly trained model. Only load trusted pickle files.

## Implemented features

- Validate all 20 expected predictor columns; block incomplete schemas and empty datasets.
- Identify missing fields, nonnumeric entries, negative numeric values, noninteger tenure, and unseen categorical levels.
- Warn before scoring; show which rows and fields need training-statistic imputation; export a full quality audit.
- Apply the existing fitted median / most-frequent imputation pipeline; do not invent fill values.
- Keep predictions across Streamlit reruns; adjust the risk threshold without rescoring.
- Show segment-level size, average predicted churn probability, and observed feature profiles in the uploaded data.
- Filter and sort customers by predicted churn risk or predicted churn probability × CLTV proxy; download filtered results including data-quality flags.
- Inspect global logistic regression coefficients with interpretation caveats.

## Optional model comparison (requires original LABELED Telco source)

```bash
python compare_models.py --data Telco_customer_churn.xlsx
```

Writes validation model comparisons, K-Means cluster diagnostics, a threshold report, and held-out test metrics in `evaluation/`. It compares logistic regression, a random forest, and logistic regression with K=2..8 clusters. Train/validation/test split prevents the final test set from being used to choose a model or threshold. The benchmark is exploratory; use repeated stratified CV for stronger estimates. The benchmark does **not** automatically replace the app's saved artifact.

The provided `test_data.csv` has 1,409 predictor-only rows and cannot be used to calculate accuracy/recall without true `Churn Value` labels.

## Interpretations and limitations

- Churn probabilities are estimates, not certainties. Lowering the probability threshold labels more customers as at risk but can generate more false alarms.
- K-Means IDs are arbitrary. Descriptive segment labels must be supported by the actual segment profiles.
- Predicted churn × CLTV is a rough prioritization index; it is **not** expected savings, an estimate of intervention efficacy, or a causal retention effect.
- Unknown categorical values are flagged and treated as missing for training-mode imputation. This is a documented design choice; if new legitimate categories become common, retraining is preferable.
- Missing or invalid row flags indicate reduced input completeness but do not measure predictive uncertainty.
- The notebook's training source `Telco_customer_churn.xlsx` was not supplied in the available files, so the model comparison script is provided but cannot be rerun here.
