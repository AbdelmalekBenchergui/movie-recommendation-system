#!/usr/bin/env python3
import argparse
import datetime
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


def numpy_user_vecs(weights, uidx, feat_idx, hist=None):
    id_part = as_np(weights["u_id.weight"])[uidx]
    c = np.concatenate([
        as_np(weights["gender.weight"])[feat_idx[:, 0]],
        as_np(weights["age.weight"])[feat_idx[:, 1]],
        as_np(weights["occ.weight"])[feat_idx[:, 2]],
    ], axis=1)
    x = np.concatenate([id_part, c], axis=1)
    if hist is not None:
        x = np.concatenate([x, hist], axis=1)
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


def build_user_history(item_vec_raw, u_tr, i_tr, r_tr, n_users, history_dim):
    """Replicate training-time history: signed rating weights, mean-normalized."""
    hw = np.clip((r_tr - T.SIGN_HALF) / 2.0, -1.0, 1.0).astype(np.float32)
    S = np.zeros((n_users, history_dim), dtype=np.float32)
    W = np.zeros(n_users, dtype=np.float32)
    np.add.at(S, u_tr, hw[:, None] * item_vec_raw[i_tr])
    np.add.at(W, u_tr, np.abs(hw))
    hist = S / np.maximum(W, 1.0)[:, None]
    hist[W <= 0.0] = 0.0
    return hist


def rebuild_split(df, model_dir):
    """Reproduce the training 80/20 split from the checkpoint's saved hyperparams."""
    metrics = json.load(open(os.path.join(model_dir, "metrics.json")))
    hp = metrics.get("hyperparams", {})
    val_fraction = float(hp.get("val_fraction", 0.2))
    seed = int(hp.get("seed", 42))
    df = df.sort_values("user_id").reset_index(drop=True)
    train_idx, val_idx = T.split_by_user(df, val_fraction, seed)
    assert len(train_idx) == int(metrics.get("n_train", -1)), \
        f"rebuilt train split {len(train_idx)} != metrics n_train {metrics.get('n_train')}"
    assert len(val_idx) == int(metrics.get("n_val", -1)), \
        f"rebuilt val split {len(val_idx)} != metrics n_val {metrics.get('n_val')}"
    return train_idx, val_idx, metrics


def torch_ref(sd, meta, genre_mat, user_feat, hist, weights_shape):
    dev = "cpu"
    dim, content_dim, out_dim, history_dim = weights_shape
    model = T.TwoTower(
        meta["n_users"], meta["n_movies"], meta["n_genres"], meta["n_ages"], meta["n_occs"],
        dim, content_dim, out_dim, 0.2, 0.5, "cosine", 0.5, history_dim, 0.0, True,
    ).to(dev)
    model.load_state_dict(sd)
    model.eval()
    gm_t = torch.from_numpy(genre_mat)
    with torch.no_grad():
        iv = model.item_vec(torch.arange(meta["n_movies"]), gm_t.to(dev)).numpy()
    items = np.random.RandomState(0).choice(meta["n_movies"], size=128, replace=False)
    users = np.random.RandomState(1).choice(meta["n_users"], size=128, replace=False)
    feat_t = {k: torch.as_tensor(v) for k, v in user_feat.items()}
    with torch.no_grad():
        u_raw_ref = model.user_vec(
            torch.from_numpy(users),
            feat_t["gender"][users], feat_t["age"][users], feat_t["occ"][users],
            torch.from_numpy(hist[users]),
        ).numpy()
    return iv, u_raw_ref, items, users


def fusion_grid(weights, item_vec_raw, item_emb, meta, user_feat, hist,
                u_all, i_all, r_all, train_idx, val_idx, k=10, pool=64,
                alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Ranking under hybrid reranking, emulating serving (top-pool by cosine, min-max norm)."""
    feat_idx = np.stack([
        user_feat["gender"].numpy(), user_feat["age"].numpy(), user_feat["occ"].numpy(),
    ], axis=1).astype(np.int64)
    n_items = meta["n_movies"]
    item_norm = l2norm(item_vec_raw)
    item_raw = as_np(item_vec_raw)

    train_pos = {}
    for idx in train_idx:
        train_pos.setdefault(u_all[idx], set()).add(int(i_all[idx]))
    val_pos = {}
    for idx in val_idx:
        val_pos.setdefault(u_all[idx], set()).add(int(i_all[idx]))
    val_users = sorted(u for u in val_pos if len(val_pos[u]) > 0)

    results = {}
    for user in val_users:
        mask = np.ones(n_items, dtype=bool)
        mask[list(train_pos.get(user, ()))] = False
        uvec = numpy_user_vecs(weights, np.array([user]), feat_idx[[user]], hist[user][None, :])
        u_norm = uvec / (np.linalg.norm(uvec) + 1e-9)
        sim = (u_norm @ item_norm.T)[0]                       # [n_items]
        sim_masked = np.where(mask, sim, -1e9)
        order = np.argsort(-sim_masked)[:pool]                # serving-style candidate pool
        cand = order[mask[order]]
        if len(cand) == 0:
            continue
        pool_sim = sim[cand]
        lo, hi = float(pool_sim.min()), float(pool_sim.max())
        sim_norm = (pool_sim - lo) / (hi - lo) if hi > lo else np.full(len(cand), 0.5)
        r_raw = np.repeat(uvec, len(cand), axis=0)
        pred = numpy_rating_head(weights, r_raw, item_raw[cand])[:, 0]
        rating_norm = (pred - 1.0) / 4.0
        pos_set = val_pos[user]
        for alpha in alphas:
            fused = alpha * sim_norm + (1.0 - alpha) * rating_norm
            order_f = np.argsort(-fused)[:k]
            ranks = {int(cand[t]): t + 1 for t in order_f if int(cand[t]) in pos_set}
            dcg = sum(1.0 / np.log2(rpos + 1) for rpos in ranks.values())
            idcg = sum(1.0 / np.log2(t + 1) for t in range(1, min(k, len(pos_set)) + 1))
            ndcg_user = dcg / idcg if idcg > 0 else 0.0
            hit_user = 1.0 if ranks else 0.0
            results.setdefault(alpha, {"ndcg@10": 0.0, "hit@10": 0.0, "n": 0})
            results[alpha]["ndcg@10"] += ndcg_user
            results[alpha]["hit@10"] += hit_user
            results[alpha]["n"] += 1
    out = {}
    for alpha, acc in results.items():
        n = max(acc["n"], 1)
        out[str(alpha)] = {"ndcg@10": round(acc["ndcg@10"] / n, 4),
                           "hit@10": round(acc["hit@10"] / n, 4), "n_users": acc["n"]}
    return out


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
    parser.add_argument("--fusion-pool", type=int, default=64)
    parser.add_argument("--fusion-alphas", default="0.0,0.25,0.5,0.75,1.0")
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
    df = df.sort_values("user_id").reset_index(drop=True)
    weights, sd = load_state(args.model_dir)
    metrics = json.load(open(os.path.join(args.model_dir, "metrics.json")))
    train_idx, val_idx, _mb = rebuild_split(df, args.model_dir)

    n_users, n_movies = meta["n_users"], meta["n_movies"]
    dim = weights["u_id.weight"].shape[1]
    content_dim = weights["gender.weight"].shape[1]
    out_dim = weights["u_mlp.3.weight"].shape[0]
    history_dim = weights["u_mlp.0.weight"].shape[1] - (dim + 3 * content_dim)

    print(f"rebuilt split: train={len(train_idx)} val={len(val_idx)} history_dim={history_dim}")

    dfn = df.astype({"user_id": str, "movie_id": str})
    u_all = dfn["user_id"].map(meta["user_idx_of_raw"]).values.astype(np.int64)
    i_all = dfn["movie_id"].map(meta["movie_idx_of_raw"]).values.astype(np.int64)
    r_all = dfn["rating"].values.astype(np.float32)

    print(f"computing item embeddings ({n_movies}x{dim}) ...")
    item_vec_raw = numpy_item_vecs(weights, genre_mat, n_movies)
    item_emb = l2norm(item_vec_raw)

    print("building user history (train-only, signed rating weights) ...")
    hist = build_user_history(item_vec_raw, u_all[train_idx], i_all[train_idx],
                              r_all[train_idx], n_users, max(history_dim, 0))
    feat_idx = np.stack([
        user_feat["gender"].numpy(), user_feat["age"].numpy(), user_feat["occ"].numpy(),
    ], axis=1).astype(np.int64)
    user_lookup = {str(raw): int(idx) for raw, idx in meta["user_idx_of_raw"].items()}

    print("verifying numpy forward vs torch ...")
    iv_ref, u_raw_ref, items, users = torch_ref(sd, meta, genre_mat, user_feat, hist, (dim, content_dim, out_dim, history_dim))
    dv = np.abs(item_vec_raw[items] - iv_ref[items]).max()
    du = np.abs(numpy_user_vecs(weights, users, feat_idx[users], hist[users]) - u_raw_ref).max()
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

    print("computing hybrid fusion weights on validation ...")
    alphas = tuple(float(a) for a in args.fusion_alphas.split(","))
    fusion = fusion_grid(weights, item_vec_raw, item_emb, meta, user_feat, hist,
                         u_all, i_all, r_all, train_idx, val_idx,
                         k=10, pool=args.fusion_pool, alphas=alphas)
    print(json.dumps(fusion, indent=2))
    best_alpha = float(max(fusion, key=lambda a: fusion[a]["ndcg@10"]))

    print("building user dislike sets (train ratings <= %d) ..." % T.NEG_MAX)
    disliked = {}
    for idx in train_idx:
        if r_all[idx] <= T.NEG_MAX:
            disliked.setdefault(int(u_all[idx]), []).append(int(i_all[idx]))
    disliked = {str(u): sorted(set(v)) for u, v in disliked.items()}

    config = {
        "run": run,
        "model_source": os.path.basename(os.path.normpath(args.model_dir)),
        "score_mode": "cosine",
        "tau": 0.5,
        "dim": dim,
        "content_dim": content_dim,
        "out_dim": item_vec_raw.shape[1],
        "history_dim": int(history_dim),
        "n_users": n_users,
        "n_movies": n_movies,
        "n_genres": meta["n_genres"],
        "index_type": "hnsw-v1",
        "rating_scale": [1.0, 5.0],
        "fusion_pool": args.fusion_pool,
        "fusion_alpha": float(best_alpha),
        "fusion_grid": fusion,
        "rerank": {"sim_norm": "minmax-over-pool", "rating_norm": "(pred-1)/4",
                   "score": "alpha*sim_norm + (1-alpha)*rating_norm"},
        "neg_max": int(T.NEG_MAX),
        "sign_half": float(T.SIGN_HALF),
        "history_dim_src": "train-only signed rating-weighted mean of item vectors",
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
    if history_dim > 0:
        np.save(out / "user_history.npy", np.ascontiguousarray(hist))
    (out / "user_lookup.json").write_text(json.dumps(user_lookup))
    (out / "user_disliked.json").write_text(json.dumps(disliked))
    (out / "item_meta.json").write_text(json.dumps(item_meta))
    (out / "popularity_top.json").write_text(json.dumps(popularity_top))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
    (out / "fusion_results.json").write_text(json.dumps(fusion, indent=2))
    shutil.copy(os.path.join(args.model_dir, "mappings.json"), out / "mappings.json")
    shutil.copy(os.path.join(args.model_dir, "model_best.pt"), out / "model_best.pt")

    print("building SageMaker model.tar.gz ...")
    tar_path = out / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in ("model_best.pt", "mappings.json", "config.json", "lineage.json"):
            tar.add(out / f, arcname=f)

    if not args.upload:
        print(f"local only -> {out}")
        print(json.dumps({"fusion_alpha": config["fusion_alpha"], "fusion_grid": fusion}))

    else:
        print("uploading to S3 ...")
        files = [
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
        ]
        if history_dim > 0:
            files.append((out / "user_history.npy", f"{base}/serving/user_history.npy"))
            files.append((out / "user_disliked.json", f"{base}/serving/user_disliked.json"))
        for local, key in files:
            s3.upload_file(str(local), args.bucket, key)

    print(f"done in {time.time()-total:.1f}s -> local .build/serve/{run} (index={config['index_type']}, fusion_alpha={config['fusion_alpha']})")


if __name__ == "__main__":
    main()