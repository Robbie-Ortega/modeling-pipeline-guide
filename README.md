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
Notes:

The variables in ${...} are sourced from your config.env file.
Ensure that docker-compose.yml and config.env are in the same directory.
### 2. Start the Services
Run the following command to build and start the services:

bash
Copy code
docker-compose --env-file config.env up -d --build
### 3. Verify the Services are Running
Check running containers:

bash
Copy code
docker ps
You should see containers for mlflow_server, mlflow_create_buckets, mlflow_minio, and mlflow_db.

Access MLflow UI:

Open your browser and navigate to:

arduino
Copy code
http://localhost:5000
Access MinIO Console:

Open your browser and navigate to:

arduino
Copy code
http://localhost:9001
Login Credentials:

Access Key (Username): Your MINIO_ROOT_USER from config.env.
Secret Key (Password): Your MINIO_ROOT_PASSWORD from config.env.
