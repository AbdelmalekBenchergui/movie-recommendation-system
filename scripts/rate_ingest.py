#!/usr/bin/env python3
import json
import os
import time
import uuid

import boto3

ML_BUCKET = os.getenv("ML_BUCKET", "recsys-dev-bucket")
PREFIX = os.getenv("STREAMING_PREFIX", "raw/streaming/ratings")
REGION = os.getenv("AWS_REGION", "us-east-1")


def _respond(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def handler(event, context):
    try:
        data = json.loads(event.get("body") or "{}")
    except Exception:
        return _respond(400, {"error": "invalid_json"})

    user_id = data.get("user_id")
    movie_id = data.get("movie_id")
    try:
        rating = float(data.get("rating"))
    except (TypeError, ValueError):
        rating = None

    if user_id is None or movie_id is None or rating is None:
        return _respond(400, {"error": "user_id, movie_id and rating required"})
    if not (1.0 <= rating <= 5.0):
        return _respond(400, {"error": "rating must be between 1 and 5"})

    timestamp = data.get("timestamp") or int(time.time())
    row = f"{user_id};{movie_id};{rating};{timestamp}\n"
    key = f"{PREFIX}/{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}.csv"

    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(Bucket=ML_BUCKET, Key=key, Body=row.encode())
    return _respond(200, {"ok": True, "key": key, "user_id": user_id, "movie_id": movie_id, "rating": rating})


if __name__ == "__main__":
    print(handler({"body": json.dumps({"user_id": 99999, "movie_id": 1, "rating": 4})}, None))