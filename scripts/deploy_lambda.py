#!/usr/bin/env python3
import json
import os

import boto3

BUCKET = os.getenv("ML_BUCKET", "recsys-dev-bucket")
RECOMMEND_FN = os.getenv("RECOMMEND_FUNCTION_NAME", "recsys-dev-recommend")
SSM_PARAM = os.getenv("MODEL_PREFIX_PARAM", "/recsys/dev/model-prefix")
REGION = os.getenv("AWS_REGION", "us-east-1")


def handler(event, context):
    run = (event.get("run") or "").strip().strip("/")
    if not run:
        return {"statusCode": 400, "body": {"error": "missing run"}}
    prefix = f"ml/models/{run}"

    try:
        lam = boto3.client("lambda", region_name=REGION)
        lam.update_function_configuration(
            FunctionName=RECOMMEND_FN,
            Environment={"Variables": {"ML_BUCKET": BUCKET, "MODEL_PREFIX": prefix}},
        )

        ssm = boto3.client("ssm", region_name=REGION)
        ssm.put_parameter(
            Name=SSM_PARAM,
            Value=prefix,
            Type="String",
            Overwrite=True,
        )
    except Exception as e:
        return {"statusCode": 500, "body": {"error": str(e)}}

    print(json.dumps({"level": "info", "event": "deployed", "function": RECOMMEND_FN, "model_prefix": prefix}))
    return {"statusCode": 200, "body": {"function": RECOMMEND_FN, "model_prefix": prefix}}


if __name__ == "__main__":
    import sys
    print(handler({"run": sys.argv[1] if len(sys.argv) > 1 else ""}, None))