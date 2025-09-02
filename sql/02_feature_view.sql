CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.vw_features_total_amount` AS
WITH src AS (
  SELECT * FROM `${PROJECT}.${DATASET}.yellow_2025_01`
  WHERE total_amount IS NOT NULL
)
SELECT
  total_amount,
  trip_distance, passenger_count, RatecodeID, payment_type,
  PULocationID, DOLocationID,
  EXTRACT(HOUR FROM tpep_pickup_datetime) AS pickup_hour,
  EXTRACT(DAYOFWEEK FROM tpep_pickup_datetime) AS pickup_dow,
  TIMESTAMP_DIFF(tpep_dropoff_datetime, tpep_pickup_datetime, MINUTE) AS trip_minutes,
  congestion_surcharge, airport_fee, cbd_congestion_fee,
  store_and_fwd_flag,
  IF(EXTRACT(HOUR FROM tpep_pickup_datetime) BETWEEN 7 AND 10 OR
     EXTRACT(HOUR FROM tpep_pickup_datetime) BETWEEN 16 AND 19, 1, 0) AS is_rush_hour,
  IF(EXTRACT(DAYOFWEEK FROM tpep_pickup_datetime) IN (1,7), 1, 0) AS is_weekend
FROM src;
