#!/usr/bin/env python3
import argparse
import glob
import gzip
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SM_INPUT = "/opt/ml/input/data"
SM_MODEL = "/opt/ml/model"

# ── rating-aware configuration ─────────────────────────────────────────────
# Ratings are graded preferences: 5 -> strong positive, 3 -> neutral, 1 -> strong negative.
# The signed weight drives both the ranking loss and the history aggregate.
NEG_MAX = 2          # ratings <= this get an explicit "not liked" penalty
SIGN_HALF = 3.0      # inflection rating
DEFAULT_HISTORY_DIM = 64


def rating_sign_weight(r):
    # r: float tensor/array of ratings in [1,5]
    # 5 -> +1.0, 4 -> +0.5, 3 -> 0.0, 2 -> -0.5, 1 -> -1.0
    return ((r - SIGN_HALF) / 2.0).clamp(-1.0, 1.0)


def _default_data_root():
    if os.path.isdir(f"{SM_INPUT}/ratings"):
        return SM_INPUT
    if os.path.isdir("data/processed/ratings_fact"):
        return "data/processed"
    return "data/ml_input"


def _default_out_root():
    if os.path.isdir(SM_MODEL):
        return SM_MODEL
    return ".build/train"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RatingDataset(Dataset):
    def __init__(self, u, i, r):
        self.u = u
        self.i = i
        self.r = r

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]


def _load_export(data_root):
    ratings_files = sorted(glob.glob(f"{data_root}/ratings/*.gz"))
    movies_files = sorted(glob.glob(f"{data_root}/movies/*.gz"))
    users_files = sorted(glob.glob(f"{data_root}/users/*.gz"))
    if not (ratings_files and movies_files and users_files):
        raise FileNotFoundError(f"no training inputs under {data_root}/{{ratings,movies,users}}")

    df = pd.concat(
        [pd.read_csv(f, compression="gzip", header=None, quotechar='"') for f in ratings_files],
        ignore_index=True,
    )
    df.columns = ["user_id", "movie_id", "rating"]

    movies_rows = []
    for f in movies_files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.rsplit(",", 1)
                head, genres = parts[0], parts[-1]
                mid, _, title = head.partition(",")
                movies_rows.append([mid, title, genres])
    movies = pd.DataFrame(movies_rows, columns=["movie_id", "title", "genres"])

    users = pd.concat(
        [pd.read_csv(f, compression="gzip", header=None, quotechar='"', dtype=str) for f in users_files],
        ignore_index=True,
    )
    users.columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    return df, movies, users


def _load_processed(data_root):
    fact_path = f"{data_root}/ratings_fact"
    if not os.path.isdir(fact_path):
        raise FileNotFoundError(f"no processed parquet under {fact_path}")
    table = ds.dataset(fact_path, format="parquet", partitioning="hive").to_table(
        columns=["user_id", "movie_id", "rating"]
    )
    df = table.to_pandas()
    movies = pq.read_table(f"{data_root}/movies").to_pandas()
    users = pq.read_table(f"{data_root}/users").to_pandas()
    return df, movies, users


def load_data(data_root):
    if os.path.isdir(f"{data_root}/ratings_fact"):
        df, movies, users = _load_processed(data_root)
    else:
        df, movies, users = _load_export(data_root)

    user_ids = sorted(df["user_id"].unique().astype(str))
    movie_ids = sorted(df["movie_id"].unique().astype(str))
    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    n_users, n_movies = len(user2idx), len(movie2idx)

    users = users[users["user_id"].astype(str).isin(user2idx)].copy()
    users["uidx"] = users["user_id"].astype(str).map(user2idx)
    movies = movies[movies["movie_id"].astype(str).isin(movie2idx)].copy()
    movies["midx"] = movies["movie_id"].astype(str).map(movie2idx)
    movies = movies.drop_duplicates("midx")

    gender_vocab = {g: i for i, g in enumerate(sorted(users["gender"].unique()))}
    age_vocab = {a: i for i, a in enumerate(sorted(set(users["age"].astype(str))))}
    occ_vocab = {o: i for i, o in enumerate(sorted(set(users["occupation"].astype(str))))}

    genres = sorted(set("|".join(movies["genres"]).split("|")))
    genre_vocab = {g: i for i, g in enumerate(genres)}

    genre_mat = np.zeros((n_movies, len(genres)), dtype=np.float32)
    for _, row in movies.iterrows():
        for g in row["genres"].split("|"):
            genre_mat[row["midx"], genre_vocab[g]] = 1.0

    ordered = users.sort_values("uidx")
    feat = {
        "gender": np.zeros(n_users, dtype=np.int64),
        "age": np.zeros(n_users, dtype=np.int64),
        "occ": np.zeros(n_users, dtype=np.int64),
    }
    uidx_vals = ordered["uidx"].values.astype(np.int64)
    feat["gender"][uidx_vals] = ordered["gender"].map(gender_vocab).values.astype(np.int64)
    feat["age"][uidx_vals] = ordered["age"].astype(str).map(age_vocab).values.astype(np.int64)
    feat["occ"][uidx_vals] = ordered["occupation"].astype(str).map(occ_vocab).values.astype(np.int64)
    user_feat = {
        "gender": torch.from_numpy(feat["gender"]),
        "age": torch.from_numpy(feat["age"]),
        "occ": torch.from_numpy(feat["occ"]),
    }

    meta = {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_genres": len(genres),
        "n_ages": len(age_vocab),
        "n_occs": len(occ_vocab),
        "gender_vocab": gender_vocab,
        "age_vocab": age_vocab,
        "occ_vocab": occ_vocab,
        "genre_vocab": genres,
        "movie_titles": {str(m): t for m, t in zip(movies["movie_id"].astype(str), movies["title"])},
        "user_idx_of_raw": user2idx,
        "movie_idx_of_raw": movie2idx,
        "raw_of_user_idx": {v: int(k) for k, v in user2idx.items()},
        "raw_of_movie_idx": {v: int(k) for k, v in movie2idx.items()},
    }

    return df, user_feat, genre_mat, meta


def split_by_user(df, val_fraction, seed):
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for idxs in df.groupby("user_id", sort=False).indices.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        if len(idxs) - n_val < 1:
            n_val = len(idxs) - 1
        if n_val < 1:
            train_idx.extend(idxs)
        else:
            train_idx.extend(idxs[n_val:])
            val_idx.extend(idxs[:n_val])
    return np.array(train_idx), np.array(val_idx)


def tower_logits(u_vec, i_vec, tau, score_mode):
    if score_mode == "cosine":
        u_vec = nn.functional.normalize(u_vec, dim=1)
        i_vec = nn.functional.normalize(i_vec, dim=1)
    return u_vec @ i_vec.t() / tau


class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, n_genres, n_ages, n_occs, dim, content_dim,
                 out_dim, dropout, tau_init, score_mode="cosine", tau_fixed=None,
                 history_dim=0, neg_weight=0.0, rating_aware=True):
        super().__init__()
        self.score_mode = score_mode
        self.history_dim = history_dim
        self.neg_weight = neg_weight
        self.rating_aware = rating_aware
        self.u_id = nn.Embedding(n_users, dim)
        self.i_id = nn.Embedding(n_movies, dim)
        self.gender = nn.Embedding(2, content_dim)
        self.age = nn.Embedding(n_ages, content_dim)
        self.occ = nn.Embedding(n_occs, content_dim)
        self.genres = nn.Embedding(n_genres, content_dim)

        nn.init.normal_(self.u_id.weight, std=0.01)
        nn.init.normal_(self.i_id.weight, std=0.01)

        self.u_mlp = nn.Sequential(
            nn.Linear(dim + 3 * content_dim + history_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.i_mlp = nn.Sequential(
            nn.Linear(dim + content_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.rating_head = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        if tau_fixed is not None:
            with torch.no_grad():
                self.log_tau.fill_(math.log(tau_fixed))
            self.log_tau.requires_grad_(False)

    def tau(self):
        return self.log_tau.exp().clamp(min=0.05)

    def user_vec(self, u, gender, age, occ, history=None):
        content = torch.cat([self.gender(gender), self.age(age), self.occ(occ)], dim=-1)
        x = torch.cat([self.u_id(u), content], dim=-1)
        if self.history_dim > 0:
            if history is None:
                history = torch.zeros(len(u), self.history_dim, device=u.device)
            x = torch.cat([x, history], dim=-1)
        return self.u_mlp(x)

    def item_vec(self, i, genre_mat):
        id_part = self.i_id(i)
        content_part = (genre_mat @ self.genres.weight) / genre_mat.sum(dim=1, keepdim=True).clamp(min=1)
        return self.i_mlp(torch.cat([id_part, content_part], dim=-1))

    def forward(self, u, gender, age, occ, i, genre_mat, rating, history=None):
        u_vec = self.user_vec(u, gender, age, occ, history)
        i_vec = self.item_vec(i, genre_mat)
        logits = tower_logits(u_vec, i_vec, self.tau(), self.score_mode)
        b = torch.arange(len(u), device=u.device)
        if not self.rating_aware:
            loss_rank = nn.functional.cross_entropy(logits, b)
            pred = 1.0 + 4.0 * torch.sigmoid(self.rating_head(torch.cat([u_vec, i_vec], dim=-1)).squeeze(-1))
            loss_mse = nn.functional.mse_loss(pred, rating)
            return loss_rank, loss_mse
        # rating-aware ranking: weight each positive row by its rating magnitude
        w_pos = rating_sign_weight(rating).clamp(min=0.0)
        loss_rank = torch.mean(w_pos * nn.functional.cross_entropy(logits, b, reduction="none"))
        # explicit negative term: low-rated items must score low for their own user
        if self.neg_weight > 0:
            neg_mask = (rating <= NEG_MAX).float()
            loss_neg = torch.mean(neg_mask * nn.functional.softplus(logits.diag()))
            loss_rank = loss_rank + self.neg_weight * loss_neg
        pred = 1.0 + 4.0 * torch.sigmoid(self.rating_head(torch.cat([u_vec, i_vec], dim=-1)).squeeze(-1))
        loss_mse = nn.functional.mse_loss(pred, rating)
        return loss_rank, loss_mse


@torch.no_grad()
def rating_metrics(model, u, i, rating, user_feat, genre_mat, device, history=None, chunk=65536):
    model.eval()
    preds = []
    for s in range(0, len(u), chunk):
        e = min(s + chunk, len(u))
        bs = slice(s, e)
        u_ids = torch.from_numpy(u[bs]).to(device)
        i_ids = torch.from_numpy(i[bs]).to(device)
        g = user_feat["gender"][u[bs]].to(device)
        a = user_feat["age"][u[bs]].to(device)
        o = user_feat["occ"][u[bs]].to(device)
        gm = torch.from_numpy(genre_mat[i[bs]]).to(device)
        history_b = history[u_ids] if history is not None else None
        u_vec = model.user_vec(u_ids, g, a, o, history_b)
        i_vec = model.item_vec(i_ids, gm)
        rp = 1.0 + 4.0 * torch.sigmoid(model.rating_head(torch.cat([u_vec, i_vec], dim=-1)).squeeze(-1))
        preds.append(rp.to("cpu"))
    pred = torch.cat(preds).numpy()
    truth = rating.astype(np.float64)
    model.train()
    return float(np.sqrt(np.mean((pred - truth) ** 2))), float(np.mean(np.abs(pred - truth)))


@torch.no_grad()
def ranking_metrics_full(model, user_feat, train_pos, val_pos, genre_mat, meta, device,
                         ks=(5, 10, 20), chunk_users=256, score_mode="cosine", history=None):
    model.eval()
    n_items = meta["n_movies"]
    genre_mat_t = torch.from_numpy(genre_mat)
    item_ids = torch.arange(n_items, device=device)
    i_vec_all = model.item_vec(item_ids, genre_mat_t.to(device)).to("cpu")

    val_users = sorted(u for u in val_pos if len(val_pos[u]) > 0)
    hits = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    recalls = {k: 0.0 for k in ks}
    precisions = {k: 0.0 for k in ks}
    n_users = 0

    for s in range(0, len(val_users), chunk_users):
        chunk = val_users[s : s + chunk_users]
        u_np = np.asarray(chunk)
        u_ids = torch.from_numpy(u_np).to(device)
        g = user_feat["gender"][u_np].to(device)
        a = user_feat["age"][u_np].to(device)
        o = user_feat["occ"][u_np].to(device)
        history_b = history[u_ids] if history is not None else None
        u_vec = model.user_vec(u_ids, g, a, o, history_b).to("cpu")
        scores = tower_logits(u_vec, i_vec_all, model.tau().item(), score_mode)

        for j, user in enumerate(chunk):
            mask = np.ones(n_items, dtype=bool)
            mask[list(train_pos.get(user, ()))] = False
            pos = np.array(list(val_pos[user]), dtype=np.int64)
            row = scores[j].numpy().copy()
            row[~mask] = -1e9
            order = np.argsort(-row)
            ranks = {p: int(np.where(order == p)[0][0]) + 1 for p in pos}
            n_users += 1
            for k in ks:
                dcg = 0.0
                rel = 0
                for r in ranks.values():
                    if r <= k:
                        dcg += 1.0 / math.log2(r + 1)
                        rel += 1
                idcg = sum(1.0 / math.log2(t + 1) for t in range(1, min(k, len(pos)) + 1))
                ndcg_contrib = dcg / idcg if idcg > 0 else 0.0
                hits[k] += 1.0 if rel > 0 else 0.0
                ndcg[k] += ndcg_contrib
                recalls[k] += rel / len(pos)
                precisions[k] += rel / k

    model.train()
    out = {}
    for k in ks:
        out[f"hit@{k}"] = hits[k] / max(n_users, 1)
        out[f"recall@{k}"] = recalls[k] / max(n_users, 1)
        out[f"precision@{k}"] = precisions[k] / max(n_users, 1)
        out[f"ndcg@{k}"] = ndcg[k] / max(n_users, 1)
    out["n_eval_users"] = n_users
    return out


def popularity_baseline(train_pos, val_pos, train_counts_by_item, n_items, ks=(5, 10, 20)):
    val_users = [u for u in val_pos if len(val_pos[u]) > 0]
    counts = np.zeros(n_items)
    for it, c in train_counts_by_item.items():
        counts[it] = c
    hits = {k: 0 for k in ks}
    for user in val_users:
        mask = np.ones(n_items, dtype=bool)
        mask[list(train_pos.get(user, ()))] = False
        pos = set(val_pos[user])
        for it in np.argsort(-counts):
            if not mask[it]:
                continue
            if it in pos:
                for k in ks:
                    hits[k] += 1.0
            break
    return {f"pop_hit@{k}": hits[k] / len(val_users) for k in ks}


def mean_baseline(u_train, r_train, u_val, r_val):
    g, user_mean = defaultdict(list), {}
    for uid, rat in zip(u_train, r_train):
        g[uid].append(float(rat))
    for uid, vals in g.items():
        user_mean[uid] = np.mean(vals)
    pred = np.asarray([user_mean.get(uid, np.mean(r_train)) for uid in u_val], dtype=np.float64)
    truth = r_val.astype(np.float64)
    return {
        "user_mean_val_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "user_mean_val_mae": float(np.mean(np.abs(pred - truth))),
        "global_mean_val_rmse": float(np.sqrt(np.mean((np.mean(r_train) - truth) ** 2))),
        "global_mean_val_mae": float(np.mean(np.abs(np.mean(r_train) - truth))),
    }


def main():
    parser = argparse.ArgumentParser(description="Two-tower recommender with in-batch sampled softmax + rating head")
    parser.add_argument("--data-root", default=_default_data_root())
    parser.add_argument("--out-root", default=_default_out_root())
    parser.add_argument("--run", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--content-dim", type=int, default=24)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tau-init", type=float, default=0.25)
    parser.add_argument("--tau-fixed", type=float, default=None)
    parser.add_argument("--tau-schedule", choices=["none", "linear"], default="none")
    parser.add_argument("--tau-end", type=float, default=None)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--score-mode", choices=["cosine", "dot"], default="cosine")
    parser.add_argument("--history-dim", type=int, default=0,
                        help="dim of the rating-weighted history aggregate fed to the user tower (0 = no history)")
    parser.add_argument("--neg-weight", type=float, default=1.0,
                        help="weight of the negative (rating<=2) BCE term in the ranking loss")
    parser.add_argument("--history-grad", action="store_true",
                        help="allow gradients to flow through the history aggregate into the item tower (default: detached)")
    parser.add_argument("--no-rating-aware", action="store_true",
                        help="replicate the original loss: plain in-batch CE, no rating weighting, no negative term")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    t_start = time.time()
    df, user_feat, genre_mat, meta = load_data(args.data_root)
    print("data loaded:", {k: meta[k] for k in ("n_users", "n_movies", "n_genres")})

    df = df.sort_values("user_id").reset_index(drop=True)
    train_idx, val_idx = split_by_user(df, args.val_fraction, args.seed)
    print(f"train={len(train_idx)} val={len(val_idx)}")

    dfn = df.astype({"user_id": str, "movie_id": str})
    u_all = dfn["user_id"].map(meta["user_idx_of_raw"]).values.astype(np.int64)
    i_all = dfn["movie_id"].map(meta["movie_idx_of_raw"]).values.astype(np.int64)
    r_all = dfn["rating"].values.astype(np.float32)

    train_pos, val_pos, train_counts = {}, {}, {}
    for idx in train_idx:
        train_pos.setdefault(u_all[idx], set()).add(i_all[idx])
        train_counts[i_all[idx]] = train_counts.get(i_all[idx], 0) + 1
    for idx in val_idx:
        val_pos.setdefault(u_all[idx], set()).add(i_all[idx])

    # train-only history rows used to build the user preference aggregates (no leakage)
    hist_rows = {
        "u": torch.from_numpy(u_all[train_idx]),
        "i": torch.from_numpy(i_all[train_idx]),
        "r": torch.from_numpy(r_all[train_idx].copy()),
    }

    dataset = RatingDataset(u_all[train_idx], i_all[train_idx], r_all[train_idx])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device == "cuda"))

    u_val, i_val, r_val = u_all[val_idx], i_all[val_idx], r_all[val_idx]

    model = TwoTower(
        meta["n_users"], meta["n_movies"], meta["n_genres"], meta["n_ages"], meta["n_occs"],
        args.dim, args.content_dim, args.out_dim, args.dropout, args.tau_init, args.score_mode,
        args.tau_fixed, args.history_dim, args.neg_weight, not args.no_rating_aware,
    ).to(device)
    if args.tau_schedule != "none":
        model.log_tau.requires_grad_(False)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=args.lr_patience)

    best_ndcg10, best_state, hist, no_improve = -1.0, None, [], 0

    # history state refreshed once per epoch from a frozen snapshot of the item tower
    hist_S = hist_W = history_full = i_vec_all = None

    def build_history_tables():
        nonlocal hist_S, hist_W, history_full, i_vec_all
        if args.history_dim <= 0:
            history_full = None
            return
        genre_mat_t = torch.from_numpy(genre_mat)
        with torch.no_grad():
            i_vec_all = model.item_vec(torch.arange(meta["n_movies"], device=device),
                                       genre_mat_t.to(device)).detach()
        r_tr = hist_rows["r"].to(device)
        hw_tr = rating_sign_weight(r_tr)                       # [ntr]
        contrib = hw_tr[:, None] * i_vec_all[hist_rows["i"].to(device)]
        hist_S = torch.zeros(meta["n_users"], args.history_dim, device=device)
        hist_W = torch.zeros(meta["n_users"], device=device)
        hist_S.index_add_(0, hist_rows["u"].to(device), contrib)
        hist_W.index_add_(0, hist_rows["u"].to(device), hw_tr.abs())
        hist_W.clamp_(min=1.0)
        history_full = hist_S / hist_W[:, None]
        history_full[hist_W <= 0.0] = 0.0
        return

    for epoch in range(1, args.epochs + 1):
        if args.tau_schedule == "linear" and args.tau_end is not None:
            frac = (epoch - 1) / max(args.epochs - 1, 1)
            tau_t = args.tau_init + (args.tau_end - args.tau_init) * frac
            with torch.no_grad():
                model.log_tau.fill_(math.log(max(tau_t, 0.05)))
        model.train()
        build_history_tables()
        t0 = time.time()
        tot_rank, tot_mse, n_batch = 0.0, 0.0, 0
        for u_b, i_b, r_b in loader:
            u_np = u_b.numpy()
            u_b = u_b.long().to(device)
            i_b = i_b.long().to(device)
            r_b = r_b.to(device)
            g = user_feat["gender"][u_np].to(device)
            a = user_feat["age"][u_np].to(device)
            o = user_feat["occ"][u_np].to(device)
            gm = torch.from_numpy(genre_mat[i_b.cpu()]).to(device)
            if history_full is not None:
                # leave-one-out: exclude the current row's own item from the user's aggregate
                hw_b = rating_sign_weight(r_b)
                hist_b = (hist_S[u_b] - hw_b[:, None] * i_vec_all[i_b]) / (hist_W[u_b] - hw_b.abs()).clamp(min=1.0)[:, None]
                if not args.history_grad:
                    hist_b = hist_b.detach()
            else:
                hist_b = None
            opt.zero_grad()
            loss_rank, loss_mse = model(u_b, g, a, o, i_b, gm, r_b, history=hist_b)
            loss = loss_rank + args.reg_weight * loss_mse
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot_rank += loss_rank.item()
            tot_mse += loss_mse.item()
            n_batch += 1
        rmse_v, mae_v = rating_metrics(model, u_val, i_val, r_val, user_feat, genre_mat, device,
                                       history=history_full)
        rk = ranking_metrics_full(model, user_feat, train_pos, val_pos, genre_mat, meta, device,
                                  ks=(5, 10, 20), score_mode=args.score_mode, history=history_full)
        scheduler.step(rk["ndcg@10"])
        cur_tau = float(model.tau().item())
        hist.append({
            "epoch": epoch,
            "rank_loss": tot_rank / max(n_batch, 1),
            "mse_loss": tot_mse / max(n_batch, 1),
            "tau": cur_tau,
            "lr": opt.param_groups[0]["lr"],
            "val_rmse": rmse_v,
            "val_mae": mae_v,
            "val_ndcg@10": rk["ndcg@10"],
            "val_hit@10": rk["hit@10"],
        })
        print(
            f"[{epoch:02d}/{args.epochs}] rank={tot_rank/max(n_batch,1):.4f} mse={tot_mse/max(n_batch,1):.4f} "
            f"tau={cur_tau:.3f} val_rmse={rmse_v:.4f} val_mae={mae_v:.4f} "
            f"ndcg@10={rk['ndcg@10']:.4f} hit@10={rk['hit@10']:.4f} ({time.time()-t0:.1f}s)", flush=True
        )
        if rk["ndcg@10"] > best_ndcg10:
            best_ndcg10 = rk["ndcg@10"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    build_history_tables()   # rebuild aggregates with the best checkpoint's item tower

    rmse_tr, mae_tr = rating_metrics(model, u_all[train_idx], i_all[train_idx], r_all[train_idx], user_feat, genre_mat, device, history=history_full)
    rmse_v, mae_v = rating_metrics(model, u_val, i_val, r_val, user_feat, genre_mat, device, history=history_full)
    rk = ranking_metrics_full(model, user_feat, train_pos, val_pos, genre_mat, meta, device, ks=(5, 10, 20), score_mode=args.score_mode, history=history_full)
    pop_b = popularity_baseline(train_pos, val_pos, train_counts, meta["n_movies"])
    mean_b = mean_baseline(u_all[train_idx], r_all[train_idx], u_val, r_val)

    results = {
        "hyperparams": vars(args),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "num_params": sum(p.numel() for p in model.parameters()),
        "best_val_ndcg10": best_ndcg10,
        "train_rmse": rmse_tr,
        "train_mae": mae_tr,
        "val_rmse": rmse_v,
        "val_mae": mae_v,
        "baselines": {**pop_b, **mean_b,
                      "random_hit@5": 5 / meta["n_movies"],
                      "random_hit@10": 10 / meta["n_movies"],
                      "random_hit@20": 20 / meta["n_movies"]},
        "ranking": rk,
        "history": hist,
        "history_recipe": {"neg_max": NEG_MAX, "sign_half": SIGN_HALF,
                           "history_dim": args.history_dim, "neg_weight": args.neg_weight,
                           "history_grad": args.history_grad,
                           "rating_aware": not args.no_rating_aware},
        "train_seconds": round(time.time() - t_start, 1),
    }

    run_dir = Path(args.out_root) / (args.run or time.strftime("twotower-%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (run_dir / "history_recipe.json").write_text(json.dumps(results["history_recipe"], indent=2))
    if best_state is not None:
        torch.save(best_state, run_dir / "model_best.pt")
    (run_dir / "mappings.json").write_text(json.dumps({
        "n_users": meta["n_users"], "n_movies": meta["n_movies"], "n_genres": meta["n_genres"],
        "genre_vocab": meta["genre_vocab"], "movie_titles": meta["movie_titles"],
        "raw_of_user_idx": {str(k): v for k, v in meta["raw_of_user_idx"].items()},
        "raw_of_movie_idx": {str(k): v for k, v in meta["raw_of_movie_idx"].items()},
    }))
    print("wrote artifacts to", run_dir)
    print(json.dumps({k: results[k] for k in ("train_rmse", "train_mae", "val_rmse", "val_mae", "baselines", "ranking")}, indent=2, default=str))


if __name__ == "__main__":
    main()