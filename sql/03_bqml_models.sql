-- Example BQML linear model, adjust as needed
CREATE OR REPLACE MODEL `${PROJECT}.${DATASET}.bqml_linear_total_amount`
OPTIONS(model_type='linear_reg') AS
SELECT
  total_amount AS label,
  trip_distance, passenger_count, RatecodeID, payment_type,
  PULocationID, DOLocationID, pickup_hour, pickup_dow, trip_minutes,
  congestion_surcharge, airport_fee, cbd_congestion_fee, store_and_fwd_flag,
  is_rush_hour, is_weekend
FROM `${PROJECT}.${DATASET}.vw_features_total_amount`;
