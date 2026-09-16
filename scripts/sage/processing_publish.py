#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ML_BUCKET  

CODE_ROOT = "/opt/ml/processing/input/code"


def ensure_deps():
    try:
        import faiss  
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])


def find_flat_dir(root):
    root = Path(root)
    if (root / "model_best.pt").exists():
        return root
    subs = [d for d in root.iterdir() if d.is_dir()]
    if len(subs) == 1 and (subs[0] / "model_best.pt").exists():
        return subs[0]
    raise SystemExit(f"could not locate model_best.pt under {root}")


def main():
    parser = argparse.ArgumentParser(description="SageMaker Processing publish wrapper")
    parser.add_argument("--model-tar", default="/opt/ml/processing/input/model/model.tar.gz")
    parser.add_argument("--data-root", default="/opt/ml/processing/input/data")
    parser.add_argument("--bucket", default=ML_BUCKET)
    parser.add_argument("--run", default=os.environ.get("RUN"))
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=100)
    args = parser.parse_args()
    args.run = (args.run or "pub-unknown").strip().strip("/")

    os.chdir("/opt/ml/processing")
    ensure_deps()

    if not os.path.exists(args.model_tar):
        raise SystemExit(f"model.tar.gz not found at {args.model_tar}")
    extract_dir = Path("/opt/ml/processing/output/model")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.model_tar, "r:gz") as t:
        t.extractall(extract_dir)
    flat = find_flat_dir(extract_dir)

    pub = f"{CODE_ROOT}/scripts/publish_ml.py"
    if not os.path.exists(pub):
        raise SystemExit(f"publish_ml.py not found at {pub}")
    cmd = [
        sys.executable, pub,
        "--model-dir", str(flat),
        "--data-root", args.data_root,
        "--bucket", args.bucket,
        "--run", args.run,
        "--ef-construction", str(args.ef_construction),
        "--ef-search", str(args.ef_search),
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"publish complete for run {args.run}")


if __name__ == "__main__":
    main()