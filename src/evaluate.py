import os
import pickle
import json
import numpy as np
import pandas as pd

from google.cloud import bigquery, storage
from sklearn.metrics import mean_squared_error, r2_score

PROJECT = os.environ.get("GCP_PROJECT", "nyc-taxi-ml-rmadi-009")
DATASET = os.environ.get("BQ_DATASET", "taxi_ds")
FEATURE_VIEW = f"{PROJECT}.{DATASET}.vw_features_total_amount"

# Model locations
LOCAL_PICKLE = os.environ.get("LOCAL_MODEL_PATH", "xgb_total_amount.pkl")
MODEL_URI = os.environ.get(
    "MODEL_URI",
    f"gs://{os.environ.get('MODEL_BUCKET', 'nyc-taxi-ml-rmadi-009-data')}/xgb/xgb_total_amount.pkl"
)

# Sampling for faster eval (1.0 = use all)
EVAL_SAMPLE_FRACTION = float(os.environ.get("EVAL_SAMPLE_FRACTION", "0.25"))
_SAMPLE_CLAUSE = "" if EVAL_SAMPLE_FRACTION >= 1.0 else f" AND RAND() < {EVAL_SAMPLE_FRACTION}"

def query_to_dataframe(sql: str) -> pd.DataFrame:
    """Force REST (no BigQuery Storage / gRPC)."""
    client = bigquery.Client(project=PROJECT)
    job = client.query(sql).result()
    # Prevent auto-using BQ Storage
    try:
        return job.to_dataframe(create_bqstorage_client=False)
    except TypeError:
        import pandas_gbq
        return pandas_gbq.read_gbq(sql, project_id=PROJECT, dialect="standard")

def eval_bqml(model_name: str):
    client = bigquery.Client(project=PROJECT)
    q = f"""
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
    SELECT SQRT(mean_squared_error) AS rmse, r2_score
    FROM ML.EVALUATE(
      MODEL `{PROJECT}.{DATASET}.{model_name}`,
      (SELECT * FROM labeled WHERE bucket >= 80)
    )
    """
    df = client.query(q).result().to_dataframe()
    print(f"BQML {model_name}:", df.to_dict(orient="records")[0])

def _load_model() -> object:
    """Load local pickle; if missing, download from GCS MODEL_URI."""
    if os.path.exists(LOCAL_PICKLE):
        with open(LOCAL_PICKLE, "rb") as f:
            return pickle.load(f)

    if not MODEL_URI.startswith("gs://"):
        raise FileNotFoundError(f"Local model not found and MODEL_URI is not gs://: {MODEL_URI}")

    gcs = storage.Client(project=PROJECT)
    bucket_name, blob_path = MODEL_URI.replace("gs://", "").split("/", 1)
    blob = gcs.bucket(bucket_name).blob(blob_path)
    data = blob.download_as_bytes()
    return pickle.loads(data)

def eval_xgb():
    print(f"[eval] Sampling fraction: {EVAL_SAMPLE_FRACTION}")
    q = f"""
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
    SELECT * FROM labeled WHERE bucket >= 80{_SAMPLE_CLAUSE}
    """
    print("[eval] Pulling eval split from BigQuery ...")
    df = query_to_dataframe(q)
    print(f"[eval] Eval rows: {len(df)}")

    # Split label & drop split column
    y = df.pop("label").values
    if "bucket" in df.columns:
        df.drop(columns=["bucket"], inplace=True)

    # Match training-time dtype cleanup
    df = df.replace({pd.NA: np.nan, None: np.nan, "": np.nan})

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
    num_cols = [c for c in df.columns if c not in cat_cols]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # df[cat_cols] = df[cat_cols].astype("object").replace({pd.NA: np.nan})
        df[cat_cols] = df[cat_cols].astype("object")
    for c in cat_cols:
        s = df[c]
        df[c] = s.where(pd.notna(s), np.nan)

    # -------------------------------------------

    print("[eval] Loading model ...")
    model = _load_model()

    print("[eval] Scoring ...")
    preds = model.predict(df)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))  # compat with older sklearn
    r2 = float(r2_score(y, preds))
    print("XGB local/GCS model:", json.dumps({"rmse": rmse, "r2": r2}, indent=2))

if __name__ == "__main__":
    try:
        eval_bqml("bqml_linear_total_amount")
    except Exception as e:
        print("BQML linear eval skipped:", e)

    try:
        eval_bqml("bqml_dnn_total_amount")
    except Exception as e:
        print("BQML DNN eval skipped:", e)

    try:
        eval_xgb()
    except Exception as e:
        print("XGB eval skipped:", e)
