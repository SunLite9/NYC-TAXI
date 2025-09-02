
import os
from io import BytesIO
import pickle
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

from google.cloud import bigquery, storage

# ------------------- Config -------------------
PROJECT = os.environ.get("GCP_PROJECT", "nyc-taxi-ml-rmadi-009")
DATASET = os.environ.get("BQ_DATASET", "taxi_ds")
BASE_TABLE = f"{PROJECT}.{DATASET}.yellow_2025_01"
PUBLIC_ZONES = "bigquery-public-data.new_york_taxi_trips.taxi_zone_geom"

# --- Eval-on-startup config (prints to terminal only) ---
FEATURE_VIEW = os.environ.get(
    "FEATURE_VIEW",
    f"{PROJECT}.{DATASET}.vw_features_total_amount"  # must contain model features + total_amount
)
PRINT_STARTUP_METRICS = os.environ.get("PRINT_STARTUP_METRICS", "1") == "1"
EVAL_SAMPLE_FRACTION = float(os.environ.get("EVAL_SAMPLE_FRACTION", "0.05"))  # 5% sample


# Model pickle in GCS (you can override via env var)
MODEL_URI = os.environ.get(
    "MODEL_URI",
    f"gs://{PROJECT}-models/xgb/xgb_total_amount.pkl"  # override if your model is in the data bucket
)

# ------------------- Clients & helpers -------------------
@st.cache_resource(show_spinner=False)
def get_bq_client():
    return bigquery.Client(project=PROJECT)

@st.cache_resource(show_spinner=False)
def get_gcs_client():
    return storage.Client(project=PROJECT)

def bq_df(sql: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a parameterized query and return a DataFrame (no BQ Storage)."""
    client = get_bq_client()
    qparams: List[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = []

    def as_py_list(x):
        # normalize numpy arrays / tuples to plain Python list[str|float|int]
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return [x]

    if params:
        for k, v in params.items():
            if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
                vals = as_py_list(v)
                # Decide array elem type by peeking at real Python types (force strings to str)
                if len(vals) == 0:
                    qparams.append(bigquery.ArrayQueryParameter(k, "STRING", []))
                else:
                    first = vals[0]
                    if isinstance(first, (int, np.integer)):
                        vals = [int(z) for z in vals]
                        qparams.append(bigquery.ArrayQueryParameter(k, "INT64", vals))
                    elif isinstance(first, (float, np.floating)):
                        vals = [float(z) for z in vals]
                        qparams.append(bigquery.ArrayQueryParameter(k, "FLOAT64", vals))
                    else:
                        vals = [str(z) for z in vals]
                        qparams.append(bigquery.ArrayQueryParameter(k, "STRING", vals))
            else:
                if v is None:
                    qparams.append(bigquery.ScalarQueryParameter(k, "STRING", None))
                elif isinstance(v, (int, np.integer)):
                    qparams.append(bigquery.ScalarQueryParameter(k, "INT64", int(v)))
                elif isinstance(v, (float, np.floating)):
                    qparams.append(bigquery.ScalarQueryParameter(k, "FLOAT64", float(v)))
                else:
                    qparams.append(bigquery.ScalarQueryParameter(k, "STRING", str(v)))

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=qparams))
    return job.result().to_dataframe(create_bqstorage_client=False)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load sklearn pickle from GCS."""
    if not MODEL_URI.startswith("gs://"):
        raise ValueError("MODEL_URI must start with gs://")
    gcs = get_gcs_client()
    bucket_name, blob_path = MODEL_URI.replace("gs://", "").split("/", 1)
    blob = gcs.bucket(bucket_name).blob(blob_path)
    data = blob.download_as_bytes()
    return pickle.loads(data)


# --- Eval-on-startup config (already added earlier) ---
from sklearn.metrics import mean_squared_error, r2_score

FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "RatecodeID",
    "payment_type",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dow",
    "trip_minutes",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "store_and_fwd_flag",
    "is_rush_hour",
    "is_weekend",
]

def _print_baseline_linear_metrics():
    """BQML baseline: compute RMSE from mean_squared_error."""
    try:
        sql = f"""
        SELECT
          SQRT(mean_squared_error) AS rmse,
          r2_score
        FROM ML.EVALUATE(MODEL `{PROJECT}.{DATASET}.bqml_linear_total_amount`)
        """
        df = bq_df(sql)
        if not df.empty:
            rmse = float(df["rmse"].iloc[0])
            r2   = float(df["r2_score"].iloc[0])
            print(f"[Baseline Linear/BQML] RMSE={rmse:.4f}, R^2={r2:.4f}")
        else:
            print("[Baseline Linear/BQML] No rows returned.")
    except Exception as e:
        print(f"[Baseline Linear/BQML] Eval skipped: {e}")

def _print_xgb_metrics():
    """Local XGB pickle: sample FEATURE_VIEW, sanitize NA, score."""
    try:
        model = load_model()
    except Exception as e:
        print(f"[XGB] Could not load model ({MODEL_URI}). Eval skipped: {e}")
        return

    try:
        cols = ", ".join([*FEATURE_COLS, "total_amount"])
        sql = f"""
        SELECT {cols}
        FROM `{FEATURE_VIEW}`
        WHERE RAND() < @frac
        """
        df = bq_df(sql, {"frac": EVAL_SAMPLE_FRACTION})
        if df.empty:
            print("[XGB] Eval sample returned 0 rows.")
            return

        # Keep only rows with target present
        df = df[pd.notna(df["total_amount"])].copy()

        # Replace pandas NA with safe values for eval
        # 1) categorical flag
        if "store_and_fwd_flag" in df.columns:
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("string").fillna("N")
            # Ensure plain Python strings (not pandas NA scalar)
            df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype(str)

        # 2) numeric columns: coerce then fill
        num_cols = [c for c in FEATURE_COLS if c != "store_and_fwd_flag"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[num_cols] = df[num_cols].fillna(0.0)

        X = df[FEATURE_COLS]
        y = pd.to_numeric(df["total_amount"], errors="coerce")
        mask = pd.notna(y)
        X, y = X.loc[mask], y.loc[mask]

        if len(X) == 0:
            print("[XGB] No valid rows after NA cleanup.")
            return

        preds = model.predict(X)
        # rmse = mean_squared_error(y, preds, squared=False)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))

        r2   = r2_score(y, preds)
        print(f"[XGB] RMSE={rmse:.4f}, R^2={r2:.4f}  (n={len(X):,}, frac={EVAL_SAMPLE_FRACTION})")
    except Exception as e:
        print(f"[XGB] Eval failed: {e}")

@st.cache_resource(show_spinner=False)
def print_metrics_once_on_startup():
    if PRINT_STARTUP_METRICS:
        print("=== Startup metrics ===")
        _print_baseline_linear_metrics()
        _print_xgb_metrics()
        print("=======================")
    return True



# ------------------- Shared SQL snippets -------------------
RATE_CODE_MAPPING = """
CASE CAST(y.RatecodeID AS INT64)
  WHEN 1 THEN 'Standard rate'
  WHEN 2 THEN 'JFK'
  WHEN 3 THEN 'Newark'
  WHEN 4 THEN 'Nassau or Westchester'
  WHEN 5 THEN 'Negotiated fare'
  WHEN 6 THEN 'Group ride'
  ELSE 'Other'
END
"""

PAYMENT_MAPPING = """
CASE CAST(y.payment_type AS INT64)
  WHEN 1 THEN 'Credit card'
  WHEN 2 THEN 'Cash'
  WHEN 3 THEN 'No charge'
  WHEN 4 THEN 'Dispute'
  ELSE 'Other'
END
"""
# In BASE_CTE
BASE_CTE = f"""
WITH zones AS (
  SELECT
    SAFE_CAST(zone_id AS INT64) AS zone_id,
    zone_name,
    borough,
    zone_geom
  FROM `{PUBLIC_ZONES}`
),
base AS (
  SELECT
    y.*,
    {RATE_CODE_MAPPING} AS rate_code_name,
    tzpu.zone_name AS pickup_zone_name,
    tzpu.borough   AS pickup_borough,
    tzpu.zone_geom AS pickup_geom,
    tzdo.zone_name AS dropoff_zone_name,
    tzdo.borough   AS dropoff_borough
  FROM `{BASE_TABLE}` AS y
  LEFT JOIN zones tzpu ON tzpu.zone_id = y.PULocationID
  LEFT JOIN zones tzdo ON tzdo.zone_id = y.DOLocationID
  WHERE y.total_amount IS NOT NULL
)
"""



# WHERE builder using parameter arrays; if empty arrays are passed we skip filtering.
def build_filters_sql() -> str:
    return """
    WHERE
      (@min_dist IS NULL OR y.trip_distance >= CAST(@min_dist AS FLOAT64))
      AND (@max_dist IS NULL OR y.trip_distance <= CAST(@max_dist AS FLOAT64))
      AND (ARRAY_LENGTH(@boroughs) = 0 OR CAST(y.pickup_borough AS STRING)   IN UNNEST(@boroughs))
      AND (ARRAY_LENGTH(@zones)    = 0 OR CAST(y.pickup_zone_name AS STRING) IN UNNEST(@zones))
      AND (ARRAY_LENGTH(@rcodes)   = 0 OR CAST(rate_code_name AS STRING)     IN UNNEST(@rcodes))
    """

def rows_after_filter(params: Dict[str, Any]) -> int:
    sql = f"""
    {BASE_CTE}
    SELECT COUNT(*) AS n
    FROM base y
    {build_filters_sql()}
    """
    df = bq_df(sql, params)
    return int(df["n"].iloc[0]) if not df.empty else 0

# ------------------- Sidebar filters -------------------
@st.cache_data(show_spinner=False)
def get_filter_options():
    sql = f"""
    {BASE_CTE}
    SELECT
      ARRAY_AGG(DISTINCT pickup_borough IGNORE NULLS ORDER BY pickup_borough) AS boroughs,
      ARRAY_AGG(DISTINCT pickup_zone_name IGNORE NULLS ORDER BY pickup_zone_name) AS zones,
      ARRAY_AGG(DISTINCT rate_code_name IGNORE NULLS ORDER BY rate_code_name) AS rcodes,
      MIN(trip_distance) AS min_d,
      MAX(trip_distance) AS max_d
    FROM base
    """
    df = bq_df(sql)

    if df.empty:
        return [], [], [], 0.0, 150.0

    row = df.iloc[0]

    def _tolist(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        try:
            return list(v)
        except Exception:
            return []

    all_b = _tolist(row.get("boroughs"))
    all_z = _tolist(row.get("zones"))
    all_r = _tolist(row.get("rcodes"))

    min_d = float(row.get("min_d")) if pd.notna(row.get("min_d")) else 0.0
    max_d = float(row.get("max_d")) if pd.notna(row.get("max_d")) else 150.0

    return all_b, all_z, all_r, min_d, max_d

@st.cache_data(show_spinner=False)
def get_zone_map():
    sql = f"""
    {BASE_CTE}
    SELECT
      pickup_borough AS borough,
      ARRAY_AGG(DISTINCT pickup_zone_name IGNORE NULLS ORDER BY pickup_zone_name) AS zones
    FROM base
    WHERE pickup_borough IS NOT NULL AND pickup_zone_name IS NOT NULL
    GROUP BY borough
    ORDER BY borough
    """
    df = bq_df(sql)

    def _tolist(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        try:
            return list(v)
        except Exception:
            return []

    zone_map = {}
    for _, r in df.iterrows():
        zone_map[r["borough"]] = _tolist(r["zones"])

    all_boroughs = sorted(zone_map.keys())
    all_zones_present = sorted({z for zs in zone_map.values() for z in zs})
    return zone_map, all_boroughs, all_zones_present



def sidebar_filters():
    st.sidebar.header("Filters")

    # Base option lists
    all_b, all_z, all_r, min_d, max_d = get_filter_options()
    zone_map, _, all_zones_present = get_zone_map()

    # First-run defaults: select all
    if "boroughs" not in st.session_state:
        st.session_state["boroughs"] = list(all_b)
    if "zones" not in st.session_state:
        st.session_state["zones"] = list(all_zones_present)
    if "rcodes" not in st.session_state:
        st.session_state["rcodes"] = list(all_r)

    # Select-all toggles
    sel_all_b = st.sidebar.checkbox("Select all boroughs", value=(len(st.session_state["boroughs"]) == len(all_b)))
    sel_all_r = st.sidebar.checkbox("Select all rate codes", value=(len(st.session_state["rcodes"]) == len(all_r)))

    # Boroughs
    if sel_all_b:
        st.session_state["boroughs"] = list(all_b)
    boroughs = st.sidebar.multiselect(
        "Borough (pickup)",
        options=all_b,
        default=st.session_state["boroughs"],
        key="boroughs",
    )

    # Zones depend on boroughs
    if boroughs:
        allowed_zones = sorted({z for b in boroughs for z in zone_map.get(b, [])})
    else:
        allowed_zones = list(all_zones_present)

    sel_all_z = st.sidebar.checkbox(
        "Select all zones (in chosen boroughs)",
        value=(set(st.session_state["zones"]) == set(allowed_zones)),
    )
    if sel_all_z:
        st.session_state["zones"] = list(allowed_zones)

    zones = st.sidebar.multiselect(
        "Zone (pickup)",
        options=allowed_zones,
        default=[z for z in st.session_state["zones"] if z in allowed_zones],
        key="zones",
    )

    # Rate codes
    if sel_all_r:
        st.session_state["rcodes"] = list(all_r)
    rcodes = st.sidebar.multiselect(
        "Rate Code Name",
        options=all_r,
        default=st.session_state["rcodes"],
        key="rcodes",
    )

    # Trip distance: 0–150 miles
    slider_max = 150.0
    safe_min = float(min_d) if pd.notna(min_d) else 0.0
    observed_max = float(max_d) if pd.notna(max_d) else 10.0
    default_hi = min(slider_max, max(10.0, observed_max))
    default_range = (safe_min, default_hi) if safe_min <= default_hi else (safe_min, safe_min)

    dist = st.sidebar.slider(
        "Trip Distance (miles)",
        min_value=safe_min,
        max_value=slider_max,
        value=default_range,
        step=0.5,
    )

    if st.sidebar.button("Reset all filters"):
        st.session_state["boroughs"] = list(all_b)
        st.session_state["zones"] = list(all_zones_present)
        st.session_state["rcodes"] = list(all_r)
        st.experimental_rerun()

    return {
        "boroughs": boroughs,
        "zones": zones,
        "rcodes": rcodes,
        "min_dist": float(dist[0]),
        "max_dist": float(dist[1]),
    }



# ------------------- KPI / Analytics queries -------------------
def kpis(params: Dict[str, Any]) -> pd.DataFrame:
    sql = f"""
    {BASE_CTE}
    SELECT
      COALESCE(ROUND(SUM(y.total_amount), 2), 0) AS total_revenue,
      COALESCE(ROUND(AVG(y.total_amount), 2), 0) AS avg_trip_amount,
      COALESCE(ROUND(AVG(y.tip_amount), 2), 0)   AS avg_tip,
      COALESCE(ROUND(AVG(y.trip_distance), 2), 0) AS avg_distance,
      COUNT(*)                                    AS total_trips,
      COUNTIF(y.trip_distance = 0)                AS cancelled_trips,
      MIN(y.tpep_pickup_datetime)                 AS first_ts,
      MAX(y.tpep_pickup_datetime)                 AS last_ts
    FROM base y
    {build_filters_sql()}
    """
    return bq_df(sql, params)


def payment_pie(params: Dict[str, Any]) -> pd.DataFrame:
    sql = f"""
    {BASE_CTE}
    SELECT
      {PAYMENT_MAPPING} AS payment_method,
      COUNT(*) AS trips,
      ROUND(SUM(y.total_amount),2) AS revenue
    FROM base y
    {build_filters_sql()}
    GROUP BY payment_method
    ORDER BY trips DESC
    """
    return bq_df(sql, params)

def revenue_by_borough(params: Dict[str, Any]) -> pd.DataFrame:
    sql = f"""
    {BASE_CTE}
    SELECT
      pickup_borough AS borough,
      COUNT(*) AS trips,
      ROUND(SUM(y.total_amount),2) AS revenue,
      ROUND(AVG(y.total_amount),2) AS avg_amount
    FROM base y
    {build_filters_sql()}
    GROUP BY borough
    ORDER BY revenue DESC
    """
    return bq_df(sql, params)

def popular_routes(params: Dict[str, Any]) -> pd.DataFrame:
    sql = f"""
    {BASE_CTE}
    SELECT
      CONCAT(y.pickup_borough, '–', y.dropoff_borough) AS route_borough,
      COUNT(*) AS trips,
      ROUND(AVG(y.trip_distance), 2) AS avg_miles,
      ROUND(AVG(y.total_amount), 2) AS avg_amount
    FROM base y
    {build_filters_sql()}
      AND y.pickup_borough IS NOT NULL
      AND y.dropoff_borough IS NOT NULL
    GROUP BY route_borough
    ORDER BY trips DESC
    LIMIT 5
    """
    return bq_df(sql, params)

def prime_pickup_spots(params: Dict[str, Any]) -> pd.DataFrame:
    # One row per pickup zone: dominant (most frequent) rate_code_name + centroid
    sql = f"""
    {BASE_CTE}
    , z AS (
      SELECT SAFE_CAST(zone_id AS INT64) AS zone_id, zone_name, borough, zone_geom
      FROM `{PUBLIC_ZONES}`
    ),
    joined AS (
      SELECT
        y.pickup_zone_name AS zone_name,
        y.pickup_borough   AS borough,
        y.rate_code_name   AS rate_code_name,
        z.zone_geom        AS geom
      FROM base y
      JOIN z ON z.zone_id = y.PULocationID
      {build_filters_sql()}
        AND z.zone_geom IS NOT NULL
    ),
    agg AS (
      SELECT
        zone_name,
        borough,
        ANY_VALUE(ST_Y(ST_CENTROID(geom))) AS lat,
        ANY_VALUE(ST_X(ST_CENTROID(geom))) AS lon,
        rate_code_name,
        COUNT(*) AS trips
      FROM joined
      GROUP BY zone_name, borough, rate_code_name
    ),
    dom AS (
      SELECT
        zone_name, borough, lat, lon,
        ARRAY_AGG(STRUCT(rate_code_name, trips) ORDER BY trips DESC)[OFFSET(0)] AS top
      FROM agg
      GROUP BY zone_name, borough, lat, lon
    )
    SELECT
      zone_name,
      borough,
      lat,
      lon,
      top.rate_code_name AS rate_code_name,
      top.trips          AS trips
    FROM dom
    ORDER BY trips DESC
    LIMIT 400
    """
    return bq_df(sql, params)


def smoke_total_rows() -> int:
    sql = f"""
    SELECT COUNT(*) AS n
    FROM `{BASE_TABLE}`
    WHERE total_amount IS NOT NULL
    """
    return int(bq_df(sql).iloc[0]["n"])

def selection_count(params: Dict[str, Any]) -> int:
    sql = f"""
    {BASE_CTE}
    SELECT COUNT(*) AS n
    FROM base y
    {build_filters_sql()}
    """
    return int(bq_df(sql, params).iloc[0]["n"])


# ------------------- UI blocks -------------------
def show_kpi_tiles(df: pd.DataFrame):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if df.empty:
        c1.metric("Total Revenue", "$0.00")
        c2.metric("Avg Trip Amount", "$0.00")
        c3.metric("Avg Tip", "$0.00")
        c4.metric("Avg Distance", "0.00 mi")
        c5.metric("Total Trip Count", "0")
        c6.metric("Cancelled Taxi Trips", "0")
        st.caption("Data Last Updated: (no rows)")
        return

    row = df.iloc[0].fillna(0)
    c1.metric("Total Revenue", f"${row.get('total_revenue', 0):,.2f}")
    c2.metric("Avg Trip Amount", f"${row.get('avg_trip_amount', 0):,.2f}")
    c3.metric("Avg Tip", f"${row.get('avg_tip', 0):,.2f}")
    c4.metric("Avg Distance", f"{row.get('avg_distance', 0):,.2f} mi")
    c5.metric("Total Trip Count", f"{int(row.get('total_trips', 0)):,}")
    c6.metric("Cancelled Taxi Trips", f"{int(row.get('cancelled_trips', 0)):,}")

    last_ts = row.get("last_ts")
    try:
        ts_label = pd.to_datetime(last_ts, utc=True).tz_convert("America/New_York")
    except Exception:
        ts_label = "(unknown)"
    # st.caption(f"Data Last Updated: {ts_label}")

def show_payment_pie(df: pd.DataFrame):
    if df.empty:
        st.info("No data for selection.")
        return
    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta="trips:Q",
            color=alt.Color("payment_method:N", legend=alt.Legend(title="Payment methods")),
            tooltip=["payment_method", "trips", "revenue"]
        )
        .properties(width=300, height=150)   # 👈 smaller pie
    )
    st.altair_chart(chart, use_container_width=False)


def show_revenue_table(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df.empty:
        st.info("No data for selection.")
    else:
        st.dataframe(df)

def show_routes_table(df: pd.DataFrame):
    st.subheader("Popular routes (top 5)")
    if df.empty:
        st.info("No data for selection.")
    else:
        st.dataframe(df)

def show_pickup_map(df: pd.DataFrame):
    st.subheader("Prime NYC Taxi Pickup Spots (top by trips)")
    if df.empty:
        st.info("No data for selection.")
        return

    # Color map (R,G,B) and matching hex for legend
    rate_colors_rgb = {
        "Standard rate":          [0, 122, 255],
        "JFK":                    [0, 200, 83],
        "Newark":                 [255, 82, 82],
        "Nassau or Westchester":  [255, 159, 28],
        "Negotiated fare":        [171, 71, 188],
        "Group ride":             [100, 181, 246],
        "Other":                  [128, 128, 128],
    }
    rate_colors_hex = {
        "Standard rate": "#007AFF",
        "JFK": "#00C853",
        "Newark": "#FF5252",
        "Nassau or Westchester": "#FF9F1C",
        "Negotiated fare": "#AB47BC",
        "Group ride": "#64B5F6",
        "Other": "#808080",
    }

    df = df.copy()
    df["color"] = df["rate_code_name"].apply(
        lambda x: rate_colors_rgb.get(x, rate_colors_rgb["Other"])
    )

    # Map + legend side-by-side
    map_col, legend_col = st.columns([4, 1])

    with map_col:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[lon, lat]',
            get_radius=120,
            get_fill_color="color",
            pickable=True,
            opacity=0.75,
        )
        view_state = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10.3, pitch=0)
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{zone_name}\n{borough}\nDominant: {rate_code_name}\nTrips: {trips}"}
        )
        st.pydeck_chart(r, use_container_width=True)

    with legend_col:
        st.markdown("**Legend**")
        legend_html = "<div style='line-height:1.6'>"
        for label, color in rate_colors_hex.items():
            legend_html += f"<div><span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;border-radius:2px;'></span>{label}</div>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)




# ------------------- Prediction tab -------------------

import datetime as dt

def tab_predict(model):
    st.subheader("Live Prediction (XGBoost)")

    mode = st.radio("Mode", ["Simple", "Advanced"], horizontal=True)

    if mode == "Simple":
        with st.form("pred_simple"):
            PULocationID = st.number_input("Pickup zone (TLC LocationID)", min_value=1, max_value=300, value=161, step=1)
            DOLocationID = st.number_input("Dropoff zone (TLC LocationID)", min_value=1, max_value=300, value=236, step=1)
            trip_distance = st.number_input("Trip distance (miles)", min_value=0.0, max_value=150.0, value=2.5, step=0.1)
            rate_code = st.selectbox("Rate code", options=[1,2,3,4,5,6], index=0)
            col_d, col_t = st.columns(2)
            with col_d:
                d = st.date_input("Pickup date", value=dt.date.today())
            with col_t:
                t = st.time_input("Pickup time", value=dt.time(8, 0))

            submit = st.form_submit_button("Predict total_amount")

        if submit:
            # Derive hour & day-of-week (1=Sun ... 7=Sat as used in training)
            pick_dt = dt.datetime.combine(d, t)
            py_wd = pick_dt.weekday()             # Mon=0..Sun=6
            pickup_dow = 1 if py_wd == 6 else py_wd + 2
            pickup_hour = t.hour
            is_rush_hour = 1 if (7 <= pickup_hour <= 10) or (16 <= pickup_hour <= 19) else 0
            is_weekend = 1 if pickup_dow in (1, 7) else 0

            # Heuristic duration (≈11.3 mph) with a small floor
            trip_minutes = max(3.0, (trip_distance / 11.3) * 60.0)

            # Defaults for the rest
            row = pd.DataFrame([{
                "trip_distance": float(trip_distance),
                "passenger_count": 1,
                "RatecodeID": int(rate_code),
                "payment_type": 1,                  # Credit card
                "PULocationID": int(PULocationID),
                "DOLocationID": int(DOLocationID),
                "pickup_hour": int(pickup_hour),
                "pickup_dow": int(pickup_dow),
                "trip_minutes": float(trip_minutes),
                "congestion_surcharge": 2.5,
                "airport_fee": 0.0,
                "cbd_congestion_fee": 0.0,
                "store_and_fwd_flag": "N",
                "is_rush_hour": int(is_rush_hour),
                "is_weekend": int(is_weekend),
            }])

            try:
                pred = model.predict(row)[0]
                st.success(f"Predicted total_amount: ${pred:0.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        # Advanced mode: your full original form (unchanged)
        with st.form("pred_adv"):
            trip_distance = st.number_input("Trip distance (miles)", min_value=0.0, value=1.6, step=0.1)
            passenger_count = st.number_input("Passenger count", min_value=0, max_value=8, value=1, step=1)
            RatecodeID = st.selectbox("Rate Code", options=[1,2,3,4,5,6], index=0)
            payment_type = st.selectbox("Payment Type", options=[1,2,3,4], index=0)
            PULocationID = st.number_input("PU LocationID", min_value=1, max_value=300, value=229, step=1)
            DOLocationID = st.number_input("DO LocationID", min_value=1, max_value=300, value=237, step=1)
            pickup_hour = st.slider("Pickup hour (0-23)", 0, 23, 0)
            pickup_dow = st.slider("Pickup day-of-week (1=Sun ... 7=Sat)", 1, 7, 4)
            trip_minutes = st.number_input("Trip duration (minutes)", min_value=0.0, value=8.5, step=0.5)
            congestion_surcharge = st.number_input("Congestion surcharge ($)", min_value=0.0, value=2.5, step=0.5)
            airport_fee = st.number_input("Airport fee ($)", min_value=0.0, value=0.0, step=0.5)
            cbd_congestion_fee = st.number_input("CBD congestion fee ($)", min_value=0.0, value=0.0, step=0.5)
            store_and_fwd_flag = st.selectbox("Store-and-forward flag", ["N", "Y"], index=0)

            is_rush_hour = 1 if (7 <= pickup_hour <= 10) or (16 <= pickup_hour <= 19) else 0
            is_weekend = 1 if pickup_dow in (1, 7) else 0

            submit = st.form_submit_button("Predict total_amount")

        if submit:
            row = pd.DataFrame([{
                "trip_distance": float(trip_distance),
                "passenger_count": int(passenger_count),
                "RatecodeID": int(RatecodeID),
                "payment_type": int(payment_type),
                "PULocationID": int(PULocationID),
                "DOLocationID": int(DOLocationID),
                "pickup_hour": int(pickup_hour),
                "pickup_dow": int(pickup_dow),
                "trip_minutes": float(trip_minutes),
                "congestion_surcharge": float(congestion_surcharge),
                "airport_fee": float(airport_fee),
                "cbd_congestion_fee": float(cbd_congestion_fee),
                "store_and_fwd_flag": store_and_fwd_flag,
                "is_rush_hour": int(is_rush_hour),
                "is_weekend": int(is_weekend),
            }])
            try:
                pred = model.predict(row)[0]
                st.success(f"Predicted total_amount: ${pred:0.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")





# ------------------- Main -------------------
def main():
    st.set_page_config(page_title="NYC Taxi ML", layout="wide")
    st.title("NYC Taxi Insights & Fare Prediction — Hybrid (BigQuery + XGBoost)")

    _ = print_metrics_once_on_startup()

    # Sidebar filters drive everything
    params = sidebar_filters()

    # Sanity + auto-widen once if current filters yield 0 rows
    # total_any = smoke_total_rows()
    # st.caption(f"Total rows in table (total_amount IS NOT NULL): {total_any:,}")

    n = selection_count(params)
    # st.caption(f"Rows with current filters: {n:,}")
    if n == 0:
        st.warning("Your current filter selection returns 0 rows. Widening filters to show data…")
        params = {"boroughs": [], "zones": [], "rcodes": [], "min_dist": None, "max_dist": None}
        n = selection_count(params)
        # st.caption(f"Rows after widening: {n:,}")

    # Tabs: Insights (KPIs + charts + map) | Predict
    tab_insights, tab_predict_tab = st.tabs(["Insights", "Predict"])

    with tab_insights:
        # KPI row
        kpi_df = kpis(params)
        show_kpi_tiles(kpi_df)
        # st.caption(f"Rows in selection: {n:,}")

        # Two columns: LEFT (pie small + tables), RIGHT (map)
        left, right = st.columns([1, 1])

        with left:
            # Pie ABOVE revenue table
            st.subheader("Payment methods (pie)")
            pm = payment_pie(params)
            show_payment_pie(pm)

            rb = revenue_by_borough(params)
            show_revenue_table(rb, "Revenue by borough (pickup)")

            routes = popular_routes(params)
            show_routes_table(routes)

        with right:
            mp = prime_pickup_spots(params)
            show_pickup_map(mp)

    with tab_predict_tab:
        st.subheader("Live Predictions")
        try:
            mdl = load_model()
            tab_predict(mdl)
        except Exception as e:
            st.warning(f"Could not load model from {MODEL_URI}. Train & upload first. Error: {e}")




if __name__ == "__main__":
    main()
