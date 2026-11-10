# Telco Customer Churn Streamlit App

Files: train_model.ipynb, model.pkl, app.py, requirements.txt, training_data.csv, test_data.csv.

Run locally:
1. pip install -r requirements.txt
2. Keep model.pkl beside app.py.
3. streamlit run app.py

Upload a CSV with the same predictor fields used for training. Geographic columns and Churn Value are not model inputs. Total Charges is converted to numeric. Missing expected columns are imputed, but supplying the complete predictor schema is recommended.

Model: stratified 80/20 split; median imputation and standard scaling for numeric fields; most-frequent imputation and one-hot encoding for categorical fields; four K-Means clusters; L2 logistic regression with C=1.0 and a 0.50 probability threshold.
