#!/usr/bin/env python3
import json
import math
import os
import time

import faiss
import numpy as np

STATE = {}


class Bundle:
    def __init__(self):
        cfg = STATE["config"]
        w = STATE["weights"]
        self.dim = w["u_mlp.3.weight"].shape[1]
        self.tau = cfg.get("tau", 0.5)
        self.n_movies = cfg["n_movies"]
        emb = STATE["embeddings"]
        self.item_norm = emb
        self.item_raw = STATE["item_raw"]
        self.index = STATE["index"]
        feat = STATE["features"]
        self.feat = {"gender": feat["gender"], "age": feat["age"], "occ": feat["occ"]}
        self.user_lookup = STATE["user_lookup"]
        self.item_meta = STATE["item_meta"]
        self.popularity = STATE.get("popularity", [])
        self.movie_idx = {m["movie_id"]: i for i, m in enumerate(self.item_meta)}
        self.history = STATE.get("history")
        self.user_disliked = STATE.get("user_disliked") or {}
        a = cfg.get("fusion_alpha")
        self.fusion_alpha = float(a) if a is not None else None

    def popular_items(self, k):
        out = []
        for p in self.popularity:
            mid = p["movie_id"]
            if mid not in self.movie_idx:
                continue
            m = self.item_meta[self.movie_idx[mid]]
            out.append({
                "movie_id": m["movie_id"],
                "title": m["title"],
                "genres": m["genres"],
                "rating_count": p["rating_count"],
            })
            if len(out) >= k:
                break
        return out

    def user_vec_raw(self, uidx):
        w = STATE["weights"]
        nu = np.array([uidx], dtype=np.int64)
        id_part = w["u_id.weight"][nu]
        c = np.concatenate([
            w["gender.weight"][[int(self.feat["gender"][uidx])]],
            w["age.weight"][[int(self.feat["age"][uidx])]],
            w["occ.weight"][[int(self.feat["occ"][uidx])]],
        ], axis=1)
        x = np.concatenate([id_part, c], axis=1)
        if self.history is not None:
            x = np.concatenate([x, self.history[uidx][None, :]], axis=1)
        x = x @ w["u_mlp.0.weight"].T + w["u_mlp.0.bias"]
        x = np.maximum(x, 0.0)
        x = x @ w["u_mlp.3.weight"].T + w["u_mlp.3.bias"]
        return x[0]

    def predict_rating(self, uvec_raw, item_idx):
        w = STATE["weights"]
        i_raw = self.item_raw[item_idx]
        x = np.concatenate([uvec_raw, i_raw])
        x = x @ w["rating_head.0.weight"].T + w["rating_head.0.bias"]
        x = np.maximum(x, 0.0)
        x = x @ w["rating_head.3.weight"][0] + w["rating_head.3.bias"][0]
        return float(1.0 + 4.0 / (1.0 + math.exp(-float(x))))


def _load_file(path):
    with open(path, "rb") as f:
        return f.read()


def _load_bundle_s3(bucket, prefix):
    import boto3
    client = boto3.client("s3")
    keys = ["config.json", "weights.npz", "item_embeddings.npy", "item_vectors_raw.npy",
            "user_features.npz", "user_lookup.json", "item_meta.json", "model.faiss", "popularity_top.json"]
    for k in keys:
        obj = client.get_object(Bucket=bucket, Key=f"{prefix}/serving/{k}")
        path = f"/tmp/srv_{k}"
        with open(path, "wb") as f:
            f.write(obj["Body"].read())
    STATE["config"] = json.loads(_load_file("/tmp/srv_config.json"))
    if STATE["config"].get("history_dim", 0) > 0 and STATE["config"].get("fusion_alpha") is not None:
        for k in ["user_history.npy", "user_disliked.json"]:
            obj = client.get_object(Bucket=bucket, Key=f"{prefix}/serving/{k}")
            with open(f"/tmp/srv_{k}", "wb") as f:
                f.write(obj["Body"].read())
    STATE["weights"] = np.load("/tmp/srv_weights.npz")
    STATE["embeddings"] = np.load("/tmp/srv_item_embeddings.npy")
    STATE["item_raw"] = np.load("/tmp/srv_item_vectors_raw.npy")
    STATE["features"] = np.load("/tmp/srv_user_features.npz")
    STATE["user_lookup"] = json.loads(_load_file("/tmp/srv_user_lookup.json"))
    STATE["item_meta"] = json.loads(_load_file("/tmp/srv_item_meta.json"))
    STATE["index"] = faiss.read_index("/tmp/srv_model.faiss")
    try:
        STATE["popularity"] = json.loads(_load_file("/tmp/srv_popularity_top.json"))
    except Exception:
        STATE["popularity"] = []
    try:
        STATE["history"] = np.load("/tmp/srv_user_history.npy") if os.path.exists("/tmp/srv_user_history.npy") else None
    except Exception:
        STATE["history"] = None
    try:
        STATE["user_disliked"] = json.loads(_load_file("/tmp/srv_user_disliked.json"))
    except Exception:
        STATE["user_disliked"] = {}


def recommend_for_user(bundle, user_id, k=10, exclude=None):
    raw = str(user_id)
    if raw not in bundle.user_lookup:
        return None
    uidx = int(bundle.user_lookup[raw])
    uvec_raw = bundle.user_vec_raw(uidx)
    nrm = np.linalg.norm(uvec_raw)
    u_norm = uvec_raw / (nrm + 1e-9)

    excl = set()
    for x in (exclude or "").split(","):
        x = x.strip()
        if x and x.isdigit() and int(x) in bundle.movie_idx:
            excl.add(bundle.movie_idx[int(x)])
    dislike_set = set(bundle.user_disliked.get(str(uidx), []))

    n = bundle.n_movies
    pool = min(n, k + 2 * len(excl) + 2 * len(dislike_set) + 32)
    scores, cand = bundle.index.search(np.ascontiguousarray(u_norm[None, :]), pool)
    scores = scores[0].astype(np.float64)
    cand = cand[0]
    valid = [(i, int(idx)) for i, idx in enumerate(cand) if idx >= 0 and idx not in excl and idx not in dislike_set]

    fused = bundle.fusion_alpha is not None
    sim = {i: float(scores[i]) for i, _ in valid}
    sim_norm = None
    if fused and valid:
        vals = [sim[i] for i, _ in valid]
        lo, hi = min(vals), max(vals)
        sim_norm = {i: (v - lo) / (hi - lo) if hi > lo else 0.5 for i, v in sim.items()}
    elif fused:
        sim_norm = {}

    rows = []
    for i, idx in valid:
        meta = bundle.item_meta[idx]
        pred = round(bundle.predict_rating(uvec_raw, idx), 2)
        if fused:
            rating_norm = (pred - 1.0) / 4.0
            score = bundle.fusion_alpha * sim_norm[i] + (1.0 - bundle.fusion_alpha) * rating_norm
        else:
            score = sim[i]
        rows.append((score, meta, sim[i], pred))
    if fused:
        rows.sort(key=lambda r: r[0], reverse=True)

    out = []
    for score, meta, cos, pred in rows[:k]:
        out.append({
            "movie_id": meta["movie_id"],
            "title": meta["title"],
            "genres": meta["genres"],
            "score": round(float(score), 6),
            "cosine_score": round(float(cos), 6),
            "predicted_rating": pred,
        })
    return out


def similar_for_item(bundle, movie_id, k=10):
    if int(movie_id) not in bundle.movie_idx:
        return None
    qi = bundle.movie_idx[int(movie_id)]
    q = np.ascontiguousarray(bundle.item_norm[qi][None, :])
    pool = min(bundle.n_movies, k + 8)
    scores, cand = bundle.index.search(q, pool)
    scores = scores[0].astype(np.float64)
    out = []
    for i, idx in enumerate(cand[0]):
        idx = int(idx)
        if idx < 0 or idx == qi:
            continue
        meta = bundle.item_meta[idx]
        out.append({
            "movie_id": meta["movie_id"],
            "title": meta["title"],
            "genres": meta["genres"],
            "score": round(float(scores[i]), 6),
        })
        if len(out) >= k:
            break
    return out


def _respond(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event, context):
    t0 = time.perf_counter()
    method = (event.get("requestContext", {}).get("http", {}) or {}).get("method", "GET")
    path = event.get("rawPath") or event.get("path") or "/"
    qp = event.get("queryStringParameters") or {}

    if not STATE.get("ready"):
        bucket = os.getenv("ML_BUCKET")
        prefix = os.getenv("MODEL_PREFIX", "")
        if not bucket:
            return _respond(500, {"error": "ML_BUCKET not set"})
        try:
            _load_bundle_s3(bucket, prefix)
        except Exception as e:
            print(json.dumps({"level": "error", "event": "load_bundle_failed", "err": str(e)}))
            return _respond(500, {"error": "bundle_load_failed"})
        STATE["ready"] = True
        STATE["bundle"] = Bundle()

    bundle = STATE["bundle"]
    status = 200
    payload = {"error": "not_found"}
    try:
        if path.rstrip("/").endswith("/recommendations"):
            uid = (qp.get("user_id") or "").strip()
            k = min(int(qp.get("k", 10)), 50)
            if not uid:
                status, payload = 400, {"error": "user_id required"}
            else:
                recs = recommend_for_user(bundle, uid, k=k, exclude=qp.get("exclude", ""))
                if recs is None:
                    status, payload = 200, {"user_id": uid, "k": k, "cold_start": True,
                                            "recommendations": bundle.popular_items(k)}
                else:
                    payload = {"user_id": uid, "k": len(recs), "recommendations": recs}
        elif path.rstrip("/").endswith("/similar"):
            mid = (qp.get("movie_id") or "").strip()
            k = min(int(qp.get("k", 10)), 50)
            if not mid or not mid.isdigit():
                status, payload = 400, {"error": "movie_id required"}
            else:
                sim = similar_for_item(bundle, int(mid), k=k)
                if sim is None:
                    status, payload = 200, {"movie_id": int(mid), "k": k, "cold_start": True,
                                            "similar": bundle.popular_items(k)}
                else:
                    payload = {"movie_id": int(mid), "k": len(sim), "similar": sim}
        elif path.rstrip("/").endswith("/health"):
            payload = {"status": "ok", "index": STATE["config"].get("index_type"), "items": bundle.n_movies}
        else:
            status, payload = 404, {"error": "unknown_path", "path": path}
    except Exception as e:
        status = 500
        payload = {"error": "internal_error", "err": str(e)}

    dt = (time.perf_counter() - t0) * 1000.0
    log = {
        "level": "info" if status < 500 else "error",
        "event": "request",
        "path": path,
        "method": method,
        "status": status,
        "latency_ms": round(dt, 1),
        "cold": not bool(STATE.get("warm")),
    }
    STATE["warm"] = True
    print(json.dumps(log))
    return _respond(status, payload)