#!/usr/bin/env python3
import json
import os
import urllib.request

API_URL = os.getenv("API_URL", "")
REGION = os.getenv("AWS_REGION", "us-east-1")


def handler(event, context):
    if not API_URL:
        return {"statusCode": 500, "body": {"error": "API_URL not set"}}
    base = API_URL.rstrip("/")
    checks = {}
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            health = json.loads(r.read().decode())
        checks["health"] = {"status": r.status, "index": health.get("index"), "items": health.get("items")}
    except Exception as e:
        checks["health"] = {"error": str(e)}

    try:
        with urllib.request.urlopen(f"{base}/recommendations?user_id=1&k=3", timeout=60) as r:
            recs = json.loads(r.read().decode())
        checks["recommendations"] = {
            "status": r.status,
            "user_id": recs.get("user_id"),
            "n": len(recs.get("recommendations", [])),
        }
    except Exception as e:
        checks["recommendations"] = {"error": str(e)}

    ok = all(c.get("error") is None for c in checks.values())
    print(json.dumps({"level": "info" if ok else "error", "event": "verify", "checks": checks}))
    return {"statusCode": 200 if ok else 500, "body": checks}


if __name__ == "__main__":
    print(handler(None, None))