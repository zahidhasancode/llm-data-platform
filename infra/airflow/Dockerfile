FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends awscli && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir "mlflow==2.22.0" boto3
