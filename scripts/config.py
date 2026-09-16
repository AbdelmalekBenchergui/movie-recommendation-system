#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "")
ML_BUCKET = os.getenv("ML_BUCKET", "recsys-dev-bucket")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "ml/models")
INPUT_S3_PREFIX = os.getenv("INPUT_S3_PREFIX", "ml/input")
CODE_S3_PREFIX = os.getenv("CODE_S3_PREFIX", "ml/retrain-code")
GLUE_DATABASE = os.getenv("GLUE_DATABASE", "recsys_dev_db")
ATHENA_WORKGROUP = os.getenv("ATHENA_WORKGROUP", "recsys-dev-athena")
STATE_MACHINE_NAME = os.getenv("STATE_MACHINE_NAME", "recsys-dev-retrain")
SAGEMAKER_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")