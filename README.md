# Modeling Pipeline Guide

This comprehensive guide provides step-by-step instructions for setting up and running a complete modeling pipeline, including training, logging, tuning, evaluation, and serving a machine learning model using MLflow and MinIO. It incorporates lessons learned to help you avoid common pitfalls and ensure a smooth experience, even if you're a beginner.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Setting Up and Running the Services](#setting-up-and-running-the-services)
4. [Training and Logging the Model](#training-and-logging-the-model)
5. [Tuning and Evaluation](#tuning-and-evaluation)
6. [Serving the Model](#serving-the-model)
7. [Testing the Served Model](#testing-the-served-model)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Conclusion](#conclusion)

---

## Prerequisites

Ensure you have the following installed:

- **Docker Desktop**
- **Python 3.8 - 3.11**
- **pip**

---

## Environment Setup

### 1. Create Project Directory

Open your terminal and run:

```bash
mkdir modeling-pipeline-guide
cd modeling-pipeline-guide
```
### 2. Create `config.env.example` File

Create a file named `config.env.example` in your project directory with the following content:

    # Postgres configuration
    PG_USER=your_pg_user
    PG_PASSWORD=your_pg_password
    PG_DATABASE=your_pg_database
    PG_PORT=5432

    # MLflow configuration
    MLFLOW_PORT=5000
    MLFLOW_BUCKET_NAME=your_mlflow_bucket_name
    MLFLOW_TRACKING_USERNAME=your_tracking_username
    MLFLOW_TRACKING_PASSWORD=your_tracking_password

    # MinIO access keys - these are needed by MLflow
    MINIO_ACCESS_KEY=your_minio_access_key
    MINIO_SECRET_ACCESS_KEY=your_minio_secret_key

    # MinIO configuration
    MINIO_ROOT_USER=your_minio_root_user
    MINIO_ROOT_PASSWORD=your_minio_root_password
    MINIO_STORAGE_USE_HTTPS=False
    MINIO_CONSOLE_ADDRESS=:9001
    MINIO_CONSOLE_PORT=9001

**Instructions:**

- **Rename** `config.env.example` to `config.env`.
- **Replace** all placeholder values (e.g., `your_pg_user`) with your actual credentials.
- **Important:** Do not commit `config.env` to version control. **Add it to `.gitignore`** to prevent accidentally exposing sensitive information.

### 3. Create `.gitignore` File

Create a `.gitignore` file to exclude sensitive files:

    # Exclude environment configuration
    config.env

---

## Setting Up and Running the Services

### 1. Create `docker-compose.yml` File

Create a `docker-compose.yml` file in your project directory with the following content:

```yaml
version: '3.7'

services:
  db:
    image: postgres:13
    container_name: mlflow_db
    environment:
      POSTGRES_USER: ${PG_USER}
      POSTGRES_PASSWORD: ${PG_PASSWORD}
      POSTGRES_DB: ${PG_DATABASE}
    ports:
      - "${PG_PORT}:5432"
    networks:
      - backend

  s3:
    image: minio/minio
    container_name: mlflow_minio
    command: server /data --console-address ":${MINIO_CONSOLE_PORT}"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"
      - "${MINIO_CONSOLE_PORT}:9001"
    networks:
      - backend

  create_buckets:
    image: minio/mc
    container_name: mlflow_create_buckets
    depends_on:
      - s3
    entrypoint: >
      /bin/sh -c "
      sleep 5;
      /usr/bin/mc config host add s3 http://s3:9000 ${MINIO_ACCESS_KEY} ${MINIO_SECRET_ACCESS_KEY} --api S3v4;
      if /usr/bin/mc ls s3/${MLFLOW_BUCKET_NAME};
      then
        echo 'Bucket ${MLFLOW_BUCKET_NAME} already exists.';
      else
        /usr/bin/mc mb s3/${MLFLOW_BUCKET_NAME};
        /usr/bin/mc policy set download s3/${MLFLOW_BUCKET_NAME};
        echo 'Bucket ${MLFLOW_BUCKET_NAME} created.';
      fi;
      exit 0;
      "
    networks:
      - backend

  tracking_server:
    image: python:3.10-slim
    container_name: mlflow_server
    depends_on:
      - db
      - s3
    environment:
      AWS_ACCESS_KEY_ID: ${MINIO_ACCESS_KEY}
      AWS_SECRET_ACCESS_KEY: ${MINIO_SECRET_ACCESS_KEY}
      MLFLOW_S3_ENDPOINT_URL: http://s3:9000
      AWS_S3_VERIFY: "0"
    ports:
      - "${MLFLOW_PORT}:5000"
    command: >
      /bin/sh -c "mlflow server
      --backend-store-uri postgresql://${PG_USER}:${PG_PASSWORD}@db:${PG_PORT}/${PG_DATABASE}
      --default-artifact-root s3://${MLFLOW_BUCKET_NAME}
      --host 0.0.0.0
      --port 5000"
    networks:
      - backend

networks:
  backend:
    driver: bridge
```
Notes:

The variables in ${...} are sourced from your config.env file.
Ensure that docker-compose.yml and config.env are in the same directory.
### 2. Start the Services
Run the following command to build and start the services:

bash
```bash
docker-compose --env-file config.env up -d --build
```
### 3. Verify the Services are Running
Check running containers:

Terminal
```bash
docker ps
```
You should see containers for mlflow_server, mlflow_create_buckets, mlflow_minio, and mlflow_db.

Access MLflow UI:

Open your browser and navigate to:

```bash
http://localhost:5000
```
Access MinIO Console:

Open your browser and navigate to:


```bash
http://localhost:9001
```
Login Credentials:

Access Key (Username): Your MINIO_ROOT_USER from config.env.
Secret Key (Password): Your MINIO_ROOT_PASSWORD from config.env.


## Training and Logging the Model
### 1. Prepare the Dataset
Wine Quality Dataset:

Download the [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
Save it as wine_quality_df.csv in your project directory.

### 2. Create the Training Script
Create a file named train_logistic_regression.py with the following content:

```python

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment name
mlflow.set_experiment("Wine_Quality_Experiment")

# Load data
data = pd.read_csv('wine_quality_df.csv')

# Prepare data
X = data.drop(columns=['quality'])
y = data['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log params, metrics, and model
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model accuracy: {accuracy}")
```
Notes:

Ensure that mlflow.set_tracking_uri("http://localhost:5000") points to your tracking server.
Adjust the dataset path if necessary.
### 3. Install Python Dependencies
In your virtual environment or system Python, install the required packages:

```bash
pip install mlflow pandas scikit-learn
```
### 4. Run the Training Script
Execute the script:

```bash
python3 train_logistic_regression.py
```
Expected Output:

The script should output the model accuracy.
The run will be logged in MLflow.
### 5. Verify the Run in MLflow UI
Navigate to the MLflow UI at http://localhost:5000.
Find your experiment "Wine_Quality_Experiment" and verify that the run appears with logged parameters, metrics, and artifacts.

## Tuning and Evaluation
### 1. Modify Hyperparameters
Edit train_logistic_regression.py and change hyperparameters, such as max_iter:

```python

# Example: Increase max_iter
model = LogisticRegression(max_iter=200)
```
### 2. Rerun the Training Script
Execute the script again:

```bash
python3 train_logistic_regression.py
```
### 3. Compare Runs in MLflow
In the MLflow UI, compare the runs to see how changes in hyperparameters affect the model performance.
## Serving the Model
### 1. Identify the Best Model's Run ID
In the MLflow UI, select the run corresponding to the best model.
Copy the Run ID from the run details page.
### 2. Set Environment Variables for Model Serving
In your terminal, set the following environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=your_minio_access_key
export AWS_SECRET_ACCESS_KEY=your_minio_secret_key
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export MLFLOW_S3_IGNORE_TLS=true
export AWS_S3_PATH_STYLE=true
```
Note: Replace your_minio_access_key and your_minio_secret_key with your actual MinIO credentials from config.env.

### 3. Install Any Additional Dependencies
Ensure all model dependencies are installed in your environment:

```bash
pip install mlflow scikit-learn pandas
```
### 4. Serve the Model
Run the following command to serve the model:

```bash
mlflow models serve -m "runs:/RUN_ID/model" -p 5001 --env-manager=local
```
Replace RUN_ID with your actual Run ID.

Wait for the server to start. You should see logs indicating the server is running.

## Testing the Served Model
### 1. Prepare the Input Data
Create a file named input.json with the following content:

```json

{
  "inputs": [
    {
      "fixed acidity": 7.4,
      "volatile acidity": 0.70,
      "citric acid": 0.00,
      "residual sugar": 1.9,
      "chlorides": 0.076,
      "free sulfur dioxide": 11,
      "total sulfur dioxide": 34,
      "density": 0.9978,
      "pH": 3.51,
      "sulphates": 0.56,
      "alcohol": 9.4
    },
    {
      "fixed acidity": 7.8,
      "volatile acidity": 0.88,
      "citric acid": 0.00,
      "residual sugar": 2.6,
      "chlorides": 0.098,
      "free sulfur dioxide": 25,
      "total sulfur dioxide": 67,
      "density": 0.9968,
      "pH": 3.20,
      "sulphates": 0.68,
      "alcohol": 9.8
    }
  ]
}
```
Ensure the feature names match your model's expectations.

### 2. Send a Prediction Request Using curl
Run the following command:

```bash
curl -X POST -H "Content-Type: application/json" --data @input.json http://127.0.0.1:5001/invocations
```
Expected Output:

```json
{"predictions": [5, 5]}
```
### 3. Send a Prediction Request Using Python (Optional)
Create a script named test_prediction.py with the following content:

```python
import requests

data = {
    "inputs": [
        {
            "fixed acidity": 7.4,
            "volatile acidity": 0.70,
            "citric acid": 0.00,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11,
            "total sulfur dioxide": 34,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        },
        {
            "fixed acidity": 7.8,
            "volatile acidity": 0.88,
            "citric acid": 0.00,
            "residual sugar": 2.6,
            "chlorides": 0.098,
            "free sulfur dioxide": 25,
            "total sulfur dioxide": 67,
            "density": 0.9968,
            "pH": 3.20,
            "sulphates": 0.68,
            "alcohol": 9.8
        }
    ]
}

response = requests.post(
    url='http://127.0.0.1:5001/invocations',
    headers={'Content-Type': 'application/json'},
    json=data
)

print("Predictions:", response.json())
```
Install the requests library if needed:

```bash
pip install requests
```
Run the script:

```bash
python test_prediction.py
```
Expected Output:

```css
Predictions: {'predictions': [5, 5]}
```
