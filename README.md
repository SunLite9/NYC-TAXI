# NYC Taxi Fare Prediction and Analytics Pipeline 🚖

End-to-end machine learning pipeline for NYC Yellow Taxi data:  
- Data ingestion and feature engineering in **BigQuery**  
- Baseline modeling with **BigQuery ML**  
- Advanced modeling with **XGBoost (Python)**  
- Model storage in **Google Cloud Storage (GCS)**  
- **Streamlit dashboard** for interactive insights and fare prediction  

---

## Quick Start
1. **Setup GCP**  
   Create the dataset, load taxi CSVs, and apply feature views/models in BigQuery.  
2. **Train Model**  
   Run the Python training script to train and upload an XGBoost model.  
3. **Evaluate Models**  
   Run the evaluation script to compare BQML and XGBoost.  
4. **Run App**  
Launch the Streamlit dashboard to explore insights and predict fares.  

---

## Models & Code (Detailed Explanation)

### 1. BigQuery SQL Layer
The system begins in **BigQuery**, where raw NYC taxi trip records are stored in a structured table. To prepare the data for machine learning, a **feature view** is created. This view transforms raw trip records into a clean, enriched dataset suitable for modeling.  

The feature engineering logic includes:  
- **Time-based features**: extracting pickup hour, day of week, and computing trip duration in minutes.  
- **Location-based features**: using pickup and dropoff location IDs, which are linked to official NYC Taxi & Limousine Commission zone geometries.  
- **Fare components**: including congestion surcharges, airport fees, and central business district congestion fees.  
- **Behavioral flags**: such as whether a trip occurred during rush hour or on a weekend.  

This results in a **single consistent dataset** that both BigQuery ML and Python-based models can use, ensuring feature parity across modeling approaches.

---

### 2. GCP Infrastructure
Once the schema and features are defined, **Google Cloud Platform services** orchestrate the workflow:  
- **BigQuery** holds the raw data, the feature view, and also supports in-database machine learning (BQML).  
- **BigQuery ML** provides the first baseline model — a linear regression predicting `total_amount` using the engineered features. This baseline is simple, interpretable, and runs entirely within BigQuery, making it a useful starting benchmark.  
- **Google Cloud Storage (GCS)** is used to persist advanced models trained outside BigQuery. When the Python training pipeline completes, the fitted XGBoost model is serialized as a pickle and uploaded to a dedicated GCS bucket.  

This design means all critical components — data, features, models — live inside GCP, enabling scalability and reproducibility across environments.

---

### 3. Python Training (XGBoost)
The advanced model is trained using **Python with XGBoost**. Instead of working directly with local CSVs, the pipeline queries the feature view from BigQuery, ensuring it uses the exact same features as BQML.  

A **deterministic data split** is implemented via hashing functions:  
- 80% of rows are assigned to training,  
- 20% are assigned to validation.  

This guarantees consistency between BQML evaluation and Python evaluation.  

The training pipeline includes:  
- **Preprocessing**:  
- Categorical variables are imputed (most frequent value) and one-hot encoded.  
- Numeric variables are imputed (median) and converted to floats.  
- **Model**:  
- An XGBoost regressor is trained with 600 trees, maximum depth of 8, and learning rate of 0.05.  
- Regularization and subsampling parameters (`subsample=0.8`, `colsample_bytree=0.8`) prevent overfitting.  
- **Persistence**:  
- After training, the model is saved locally as a pickle.  
- It is then uploaded to a GCS bucket, where downstream services (like the Streamlit app) can load it directly.  

This approach provides a **powerful, non-linear model** that captures complex interactions in the data and outperforms the linear baseline.

---

### 4. Model Evaluation
To ensure fairness and reproducibility, both BQML and XGBoost are evaluated on the **same holdout split** (20% of the dataset).  

- **BQML Evaluation**:  
Evaluation is done directly in BigQuery using the `ML.EVALUATE` function. Metrics like RMSE and R² are computed within the database.  

- **XGBoost Evaluation**:  
The evaluation script pulls the validation split from BigQuery, performs the same preprocessing steps as training, loads the saved XGBoost model (from local disk or GCS), and computes metrics using scikit-learn.  

The comparison clearly shows the improvement of the advanced model:  
```
- Linear regression (BQML): RMSE ≈ 17.7, R² ≈ 0.43  
- XGBoost (Python): RMSE ≈ 5.8, R² ≈ 0.89  
```
This demonstrates how moving from a linear baseline to a boosted tree model dramatically increases predictive accuracy.

---

### 5. Streamlit Application
Finally, the pipeline comes to life in a **Streamlit dashboard**. The app has two main tabs:  

- **Insights**:  
- Displays revenue, average fare, tip size, and trip counts as KPI tiles.  
- Shows a payment method breakdown (pie chart).  
- Provides revenue by borough and top routes in tabular form.  
- Visualizes prime pickup hotspots on an interactive NYC map.  

- **Predict**:  
- Simple mode: user inputs pickup/dropoff zones, trip distance, and time; derived features (rush hour, weekend, trip duration) are auto-calculated.  
- Advanced mode: user manually inputs all model features.  
- The app loads the XGBoost model from GCS, generates a prediction for `total_amount`, and displays the result instantly.  

This interface allows non-technical users to explore the data and interact directly with the ML model.  

---

## Tech Stack
- **BigQuery**: data storage, feature engineering, baseline ML  
- **BigQuery ML**: linear regression baseline  
- **Google Cloud Storage**: model storage  
- **Python (pandas, scikit-learn, XGBoost)**: advanced model training and evaluation  
- **Streamlit**: interactive dashboard for insights and predictions  

---

The pipeline runs end-to-end: raw trip data → BigQuery feature engineering → baseline ML → advanced XGBoost training and evaluation → live analytics and predictions in Streamlit.  
