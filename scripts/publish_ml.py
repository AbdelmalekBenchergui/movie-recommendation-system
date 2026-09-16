#!/usr/bin/env python3
import argparse
import datetime
import glob
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import boto3
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.sage import two_tower_train as T  # noqa: E402
from config import AWS_REGION, MODEL_S3_PREFIX, ML_BUCKET  # noqa: E402


def load_state(model_dir):
    pt = os.path.join(model_dir, "model_best.pt")
    sd = torch.load(pt, map_location="cpu")
    weights = {k: v.numpy() for k, v in sd.items()}
    return weights, sd


def relu(x):
    return np.maximum(x, 0.0)


def as_np(v):
    return np.asarray(v, dtype=np.float32)


def numpy_item_vecs(weights, genre_mat, n_movies):
    id_part = as_np(weights["i_id.weight"])
    geom = genre_mat @ as_np(weights["genres.weight"])
    geom = geom / genre_mat.sum(axis=1, keepdims=True).clip(min=1)
    x = np.concatenate([id_part, geom], axis=1)
    x = x @ as_np(weights["i_mlp.0.weight"].T) + as_np(weights["i_mlp.0.bias"])
    x = relu(x)
    x = x @ as_np(weights["i_mlp.3.weight"].T) + as_np(weights["i_mlp.3.bias"])
    assert x.shape == (n_movies, 64)
    return x


def numpy_user_vecs(weights, uidx, feat_idx):
    id_part = as_np(weights["u_id.weight"])[uidx]
    c = np.concatenate([
        as_np(weights["gender.weight"])[feat_idx[:, 0]],
        as_np(weights["age.weight"])[feat_idx[:, 1]],
        as_np(weights["occ.weight"])[feat_idx[:, 2]],
    ], axis=1)
    x = np.concatenate([id_part, c], axis=1)
    x = x @ as_np(weights["u_mlp.0.weight"].T) + as_np(weights["u_mlp.0.bias"])
    x = relu(x)
    x = x @ as_np(weights["u_mlp.3.weight"].T) + as_np(weights["u_mlp.3.bias"])
    return x


def numpy_rating_head(weights, u_raw, i_raw):
    x = np.concatenate([u_raw, i_raw], axis=1)
    x = x @ as_np(weights["rating_head.0.weight"].T) + as_np(weights["rating_head.0.bias"])
    x = relu(x)
    x = x @ as_np(weights["rating_head.3.weight"].T) + as_np(weights["rating_head.3.bias"])
    p = 1.0 + 4.0 * (1.0 / (1.0 + np.exp(-x)))
    return p


def l2norm(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def torch_ref(sd, meta, genre_mat, user_feat, n_check=128):
    dev = "cpu"
    model = T.TwoTower(
        meta["n_users"], meta["n_movies"], meta["n_genres"], meta["n_ages"], meta["n_occs"],
        64, 24, 64, 0.2, 0.5, "cosine", 0.5,
    ).to(dev)
    model.load_state_dict(sd)
    model.eval()
    gm_t = torch.from_numpy(genre_mat)
    with torch.no_grad():
        iv = model.item_vec(torch.arange(meta["n_movies"]), gm_t.to(dev)).numpy()
    items = np.random.RandomState(0).choice(meta["n_movies"], size=n_check, replace=False)
    users = np.random.RandomState(1).choice(meta["n_users"], size=n_check, replace=False)
    with torch.no_grad():
        u_raw_ref = model.user_vec(
            torch.from_numpy(users),
            user_feat["gender"][users], user_feat["age"][users], user_feat["occ"][users],
        ).numpy()
    return iv, u_raw_ref, items, users


def build_hnsw(vectors, dim, M=32, ef_construction=100, ef_search=100):
    import faiss
    idx = faiss.index_factory(dim, f"HNSW{M}", faiss.METRIC_INNER_PRODUCT)
    if hasattr(idx, "hnsw"):
        idx.hnsw.efConstruction = ef_construction
        idx.hnsw.efSearch = ef_search
    idx.add(vectors)
    return idx


def recall_at_100(hnsw, flat, vectors, k=100, sample=1000):
    import faiss
    q = vectors[:sample] if len(vectors) > sample else vectors
    _, ids_hnsw = hnsw.search(q, k)
    _, ids_flat = flat.search(q, k)
    ids_flat_set = {tuple(sorted(row)) for row in ids_flat}
    inter = sum(len(set(a) & set(b)) for a, b in zip(ids_hnsw, ids_flat))
    return inter / (len(q) * k)


def main():
    parser = argparse.ArgumentParser(description="Publish two-tower checkpoint to S3 (SageMaker + Lambda bundle)")
    parser.add_argument("--model-dir", default=".build/train/twotower-r5-cos-tau05-long")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--bucket", default=ML_BUCKET)
    parser.add_argument("--run", default=None)
    parser.add_argument("--upload", action="store_true", default=True)
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    parser.add_argument("--region", default=AWS_REGION)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=100)
    args = parser.parse_args()

    run = args.run or datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = Path(".build/serve") / run
    out.mkdir(parents=True, exist_ok=True)
    total = time.time()

    if args.upload:
        s3 = boto3.client("s3", region_name=args.region)
        base = f"{MODEL_S3_PREFIX}/{run}"
        existing = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"{base}/")
        if existing.get("KeyCount", 0):
            raise SystemExit(f"refusing to overwrite existing prefix s3://{args.bucket}/{base}/")

    print("loading data + model ...")
    df, user_feat, genre_mat, meta = T.load_data(args.data_root)
    weights, sd = load_state(args.model_dir)
    metrics = json.load(open(os.path.join(args.model_dir, "metrics.json")))

    n_users, n_movies = meta["n_users"], meta["n_movies"]
    tau = 0.5
    dim = weights["u_id.weight"].shape[1]
    content_dim = weights["gender.weight"].shape[1]

    print(f"computing item embeddings ({n_movies}x{dim}) ...")
    item_vec_raw = numpy_item_vecs(weights, genre_mat, n_movies)
    item_emb = l2norm(item_vec_raw)

    print("computing user features + lookup ...")
    feat_idx = np.stack([
        user_feat["gender"].numpy(), user_feat["age"].numpy(), user_feat["occ"].numpy(),
    ], axis=1).astype(np.int64)
    user_lookup = {str(raw): int(idx) for raw, idx in meta["user_idx_of_raw"].items()}

    print("verifying numpy forward vs torch ...")
    iv_ref, u_raw_ref, items, users = torch_ref(sd, meta, genre_mat, user_feat)
    dv = np.abs(item_vec_raw[items] - iv_ref[items]).max()
    du = np.abs(numpy_user_vecs(weights, users, feat_idx[users]) - u_raw_ref).max()
    print(f"numpy vs torch max|diff|: item_vec={dv:.2e} user_vec={du:.2e}")
    assert dv < 1e-4 and du < 1e-4, "numpy forward diverges from torch"

    item_meta = []
    for i in range(n_movies):
        mid = str(meta["raw_of_movie_idx"][i])
        item_meta.append({
            "movie_id": int(mid),
            "title": meta["movie_titles"].get(mid, ""),
            "genres": [meta["genre_vocab"][j] for j in np.where(genre_mat[i])[0]],
        })

    print("computing popularity ...")
    popularity_top = [
        {"movie_id": int(k), "rating_count": int(v)} for k, v in df["movie_id"].value_counts().items()
    ]

    config = {
        "run": run,
        "model_source": os.path.basename(os.path.normpath(args.model_dir)),
        "score_mode": "cosine",
        "tau": tau,
        "dim": dim,
        "content_dim": content_dim,
        "out_dim": item_vec_raw.shape[1],
        "n_users": n_users,
        "n_movies": n_movies,
        "n_genres": meta["n_genres"],
        "index_type": "hnsw-v1",
        "rating_scale": [1.0, 5.0],
    }
    lineage = {
        "run": run,
        "model_source": args.model_dir,
        "hyperparams": metrics.get("hyperparams", {}),
        "metrics": {k: metrics.get(k) for k in ("best_val_ndcg10", "val_rmse", "val_mae", "train_rmse") if k in metrics},
        "n_train": metrics.get("n_train"), "n_val": metrics.get("n_val"),
        "published_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    print("building FAISS index ...")
    import faiss
    D = item_emb.shape[1]
    flat = faiss.IndexFlatIP(D)
    flat.add(np.ascontiguousarray(item_emb))
    faiss.write_index(flat, str(out / "index_exact.faiss"))
    hnsw = build_hnsw(np.ascontiguousarray(item_emb), D, ef_construction=args.ef_construction, ef_search=args.ef_search)
    rec = recall_at_100(hnsw, flat, np.ascontiguousarray(item_emb))
    print(f"HNSW recall@100 vs exact: {rec:.4f}")
    chosen = hnsw if rec >= 0.99 else flat
    config["index_type"] = "hnsw-v1" if chosen is hnsw else "flat-exact"
    config["recall@100_vs_exact"] = float(rec)
    faiss.write_index(chosen, str(out / "model.faiss"))

    print("writing package files ...")
    np.savez(out / "weights.npz", **weights)
    np.save(out / "item_embeddings.npy", item_emb)
    np.save(out / "item_vectors_raw.npy", item_vec_raw)
    np.savez(out / "user_features.npz", gender=feat_idx[:, 0], age=feat_idx[:, 1], occ=feat_idx[:, 2])
    (out / "user_lookup.json").write_text(json.dumps(user_lookup))
    (out / "item_meta.json").write_text(json.dumps(item_meta))
    (out / "popularity_top.json").write_text(json.dumps(popularity_top))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
    shutil.copy(os.path.join(args.model_dir, "mappings.json"), out / "mappings.json")
    shutil.copy(os.path.join(args.model_dir, "model_best.pt"), out / "model_best.pt")

    print("building SageMaker model.tar.gz ...")
    tar_path = out / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in ("model_best.pt", "mappings.json", "config.json", "lineage.json"):
            tar.add(out / f, arcname=f)

    if not args.upload:
        print(f"local only -> {out}")
        return

    print("uploading to S3 ...")
    for local, key in [
        (out / "model.tar.gz", f"{base}/model.tar.gz"),
        (out / "weights.npz", f"{base}/serving/weights.npz"),
        (out / "item_embeddings.npy", f"{base}/serving/item_embeddings.npy"),
        (out / "item_vectors_raw.npy", f"{base}/serving/item_vectors_raw.npy"),
        (out / "model.faiss", f"{base}/serving/model.faiss"),
        (out / "index_exact.faiss", f"{base}/serving/index_exact.faiss"),
        (out / "item_meta.json", f"{base}/serving/item_meta.json"),
        (out / "popularity_top.json", f"{base}/serving/popularity_top.json"),
        (out / "user_lookup.json", f"{base}/serving/user_lookup.json"),
        (out / "user_features.npz", f"{base}/serving/user_features.npz"),
        (out / "config.json", f"{base}/serving/config.json"),
        (out / "lineage.json", f"{base}/lineage.json"),
    ]:
        s3.upload_file(str(local), args.bucket, key)

    print(f"done in {time.time()-total:.1f}s -> s3://{args.bucket}/{base}/ (index={config['index_type']})")


if __name__ == "__main__":
    main()