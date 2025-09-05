import os
import json
import pickle
from typing import Tuple

import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

#Config 
PROJECT = os.environ.get("GCP_PROJECT", "nyc-taxi-ml-rmadi-009")
DATASET = os.environ.get("BQ_DATASET", "taxi_ds")
FEATURE_VIEW = f"{PROJECT}.{DATASET}.vw_features_total_amount"
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", f"{PROJECT}-models")  #  nyc-taxi-ml-rmadi-009-models
MODEL_BLOB = os.environ.get("MODEL_BLOB", "xgb/xgb_total_amount.pkl")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "xgb_total_amount.pkl")

SAMPLE_FRACTION = float(os.environ.get("BQ_SAMPLE_FRACTION", "1.0"))
SAMPLE_CLAUSE = "" if SAMPLE_FRACTION >= 1.0 else f"WHERE RAND() < {SAMPLE_FRACTION}"

# Helpers
def query_to_dataframe(sql: str) -> pd.DataFrame:
    """Force REST (no BigQuery Storage / gRPC). Use BQ Storage only if explicitly allowed."""
    client = bigquery.Client(project=PROJECT)
    job = client.query(sql).result()

    use_bq_storage = os.environ.get("USE_BQ_STORAGE", "0") == "1"

    if use_bq_storage:
        # try fast path explicitly
        try:
            from google.cloud import bigquery_storage_v1
            bqs = bigquery_storage_v1.BigQueryReadClient()
            return job.to_dataframe(bqstorage_client=bqs)
        except Exception as e:
            print(f"BigQuery Storage failed ({e.__class__.__name__}). Falling back to REST...")

    # hard-disable BQ Storage: pass the flag so it won't auto-create a bqstorage client
    try:
        return job.to_dataframe(create_bqstorage_client=False)
    except TypeError:
        # very old bigquery versions may not accept the flag; as a last resort, use pandas-gbq
        import pandas_gbq
        return pandas_gbq.read_gbq(sql, project_id=PROJECT, dialect="standard")


# Data
def fetch_train_valid() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Pull features from BigQuery using same hash split as BQML:
    train = bucket < 80, valid = bucket >= 80
    """
    query = f"""
    WITH labeled AS (
      SELECT
        total_amount AS label,
        trip_distance, passenger_count, RatecodeID, payment_type,
        PULocationID, DOLocationID,
        pickup_hour, pickup_dow, trip_minutes,
        congestion_surcharge, airport_fee, cbd_congestion_fee,
        store_and_fwd_flag, is_rush_hour, is_weekend,
        MOD(ABS(FARM_FINGERPRINT(
          CAST(pickup_hour AS STRING) || CAST(pickup_dow AS STRING) ||
          CAST(PULocationID AS STRING) || CAST(DOLocationID AS STRING)
        )), 100) AS bucket
      FROM `{FEATURE_VIEW}`
    )
    SELECT * FROM labeled
    {SAMPLE_CLAUSE}
    """
    df = query_to_dataframe(query)

    train = df[df["bucket"] < 80].copy()
    valid = df[df["bucket"] >= 80].copy()

    y_train = train.pop("label").values
    y_valid = valid.pop("label").values
    train.drop(columns=["bucket"], inplace=True)
    valid.drop(columns=["bucket"], inplace=True)
    return train, valid, y_train, y_valid

# Model 
def build_pipeline(cat_cols, num_cols) -> Pipeline:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # dense output preferred for XGB
    )

    xgb_reg = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",   # set to "gpu_hist" if training on GPU
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("xgb", xgb_reg)])

def main():
    print("Loading data from BigQuery...")
    X_train, X_valid, y_train, y_valid = fetch_train_valid()
    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

    # Normalize NA markers so sklearn sees np.nan (not pandas.NA / <NA>)
    X_train = X_train.replace({pd.NA: np.nan, None: np.nan, "": np.nan})
    X_valid = X_valid.replace({pd.NA: np.nan, None: np.nan, "": np.nan})

    # Treat passenger_count as categorical (avoids nullable-int pitfalls)
    cat_cols = [
        "passenger_count",
        "RatecodeID",
        "payment_type",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_dow",
        "store_and_fwd_flag",
        "is_rush_hour",
        "is_weekend",
    ]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Ensure numeric columns are pure float64 (no pandas nullable dtypes)
    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float64")
        X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce").astype("float64")

    # Ensure categorical columns are plain 'object'
    X_train[cat_cols] = X_train[cat_cols].astype("object")
    X_valid[cat_cols] = X_valid[cat_cols].astype("object")
    X_train[cat_cols] = X_train[cat_cols].where(pd.notna(X_train[cat_cols]), np.nan)
    X_valid[cat_cols] = X_valid[cat_cols].where(pd.notna(X_valid[cat_cols]), np.nan)

    print("Building pipeline...")
    pipe = build_pipeline(cat_cols, num_cols)

    print("Training XGBoost...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    # Old sklearn compatibility: no 'squared' kw
    rmse = float(np.sqrt(mean_squared_error(y_valid, pipe.predict(X_valid))))
    r2 = float(r2_score(y_valid, pipe.predict(X_valid)))
    print(json.dumps({"rmse": rmse, "r2": r2}, indent=2))

    print(f"Saving model to {LOCAL_MODEL_PATH} ...")
    with open(LOCAL_MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    print(f"Uploading to gs://{MODEL_BUCKET}/{MODEL_BLOB} ...")
    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.bucket(MODEL_BUCKET)
    if not bucket.exists():
        print(f"Bucket {MODEL_BUCKET} not found. Creating in us-central1 ...")
        bucket = storage_client.create_bucket(MODEL_BUCKET, location="us-central1")
    blob = bucket.blob(MODEL_BLOB)
    blob.upload_from_filename(LOCAL_MODEL_PATH)
    print("Upload complete.")

if __name__ == "__main__":
    main()
