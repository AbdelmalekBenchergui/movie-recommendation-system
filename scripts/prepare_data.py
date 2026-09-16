#!/usr/bin/env python3
import datetime
import json
import os
import time

import boto3

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("ML_BUCKET", "recsys-dev-bucket")
DATABASE = os.getenv("GLUE_DATABASE", "recsys_dev_db")
WORKGROUP = os.getenv("ATHENA_WORKGROUP", "recsys-dev-athena")
INPUT_S3_PREFIX = os.getenv("INPUT_S3_PREFIX", "ml/input")

UNLOAD_SQL = {
    "ratings": (
        "SELECT CAST(user_id AS VARCHAR) AS user_id, "
        "CAST(movie_id AS VARCHAR) AS movie_id, "
        "CAST(rating AS VARCHAR) AS rating "
        "FROM ratings_fact"
    ),
    "movies": (
        "SELECT CAST(m.movie_id AS VARCHAR) AS movie_id, "
        "CAST(m.title AS VARCHAR) AS title, "
        "CAST(m.genres AS VARCHAR) AS genres "
        "FROM movies m "
        "WHERE m.movie_id IN (SELECT DISTINCT movie_id FROM ratings_fact)"
    ),
    "users": (
        "SELECT CAST(user_id AS VARCHAR) AS user_id, "
        "CAST(gender AS VARCHAR) AS gender, "
        "CAST(age AS VARCHAR) AS age, "
        "CAST(occupation AS VARCHAR) AS occupation, "
        "CAST(zip_code AS VARCHAR) AS zip_code "
        "FROM users"
    ),
}


def run_query(client, sql):
    resp = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": DATABASE},
        WorkGroup=WORKGROUP,
    )
    qid = resp["QueryExecutionId"]
    while True:
        state = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return qid, state
        time.sleep(2)


def handler(event, context):
    epochs = event.get("epochs", "3")
    instance_type = event.get("instance_type", "ml.m5.large")
    run = event.get("run") or "recsys-" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    prefix = f"{BUCKET}/{INPUT_S3_PREFIX}/{run}"

    client = boto3.client("athena", region_name=REGION)
    for name in ("ratings", "movies", "users"):
        sql = f"UNLOAD ( {UNLOAD_SQL[name]} ) TO 's3://{prefix}/{name}/' WITH (format='TEXTFILE', field_delimiter=',')"
        qid, state = run_query(client, sql)
        if state != "SUCCEEDED":
            raise RuntimeError(f"unload {name} {state}: {qid}")

    print(json.dumps({"level": "info", "event": "prepare_done", "run": run,
                      "input_s3": f"s3://{prefix}", "epochs": epochs, "instance_type": instance_type}))
    return {"run": run, "input_s3": f"s3://{prefix}", "instance_type": instance_type, "epochs": epochs}


if __name__ == "__main__":
    print(handler({}, None))