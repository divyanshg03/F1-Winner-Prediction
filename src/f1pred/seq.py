"""Sequence models over each driver's recent race history -> per-driver score -> race softmax.

Fixes the v1 notebook's bugs: finite masking (no NaN), no label smoothing, a CLS token so the
Transformer never sees a fully-masked sequence, and early stopping only on held-out *training* races.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

K = 10
HIST_COLS = ["h_grid", "h_pos", "h_perf", "h_dnf", "h_qgap", "h_conperf"]
NEG = -1e4  # finite mask value


def build_races(df: pd.DataFrame, static_cols: list[str], dmax: int = 24):
    """Return dict race_key -> arrays, ordered by date. History uses strictly earlier races only."""
    d = df[df.won.notna()].sort_values(["date", "round", "driver_id"]).copy()
    n = d.groupby("race_key")["driver_id"].transform("size")
    fin = d["pos"].fillna(n)
    perf = (n - fin) / (n - 1).clip(lower=1)
    perf = np.where(d["dnf"] == 1, np.minimum(perf, 0.15), perf)
    d["h_grid"] = d["grid"] / n
    d["h_pos"] = fin / n
    d["h_perf"] = perf
    d["h_dnf"] = d["dnf"].astype(float)
    d["h_qgap"] = d["qual_gap"].clip(0, 0.2)
    d["h_conperf"] = d.groupby(["race_key", "constructor_id"])["h_perf"].transform("mean")
    order = d.drop_duplicates("race_key")[["race_key", "date"]].sort_values("date")
    hist: dict[str, list[np.ndarray]] = {}
    out = {}
    for key in order.race_key:
        g = d[d.race_key == key]
        D = len(g)
        H = np.zeros((dmax, K, len(HIST_COLS)), np.float32)
        M = np.zeros((dmax, K), np.float32)
        for i, drv in enumerate(g.driver_id):
            past = hist.get(drv, [])[-K:]
            if past:
                H[i, -len(past):] = np.stack(past)
                M[i, -len(past):] = 1.0
        S = np.zeros((dmax, len(static_cols)), np.float32)
        S[:D] = g[static_cols].to_numpy(np.float32)
        dm = np.zeros(dmax, bool)
        dm[:D] = True
        w = int(np.argmax(g.won.to_numpy()))
        out[key] = dict(H=H, M=M, S=S, dm=dm, y=w, drivers=g.driver_id.tolist(), date=g.date.iloc[0])
        for i, drv in enumerate(g.driver_id):  # update AFTER building the sample
            hist.setdefault(drv, []).append(g[HIST_COLS].to_numpy(np.float32)[i])
    return out


class Net(nn.Module):
    def __init__(self, arch: str, n_static: int, hid: int = 48, dropout: float = 0.2):
        super().__init__()
        self.arch = arch
        nf = len(HIST_COLS)
        if arch == "gru":
            self.enc = nn.GRU(nf + 1, hid, batch_first=True)
        else:
            self.proj = nn.Linear(nf, hid)
            self.cls = nn.Parameter(torch.zeros(1, 1, hid))
            self.pos = nn.Parameter(torch.randn(1, K + 1, hid) * 0.02)
            layer = nn.TransformerEncoderLayer(hid, 4, hid * 2, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hid + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, H, M, S, dm):
        B, D = H.shape[:2]
        h = H.reshape(B * D, K, -1)
        m = M.reshape(B * D, K)
        if self.arch == "gru":
            x = torch.cat([h * m[..., None], m[..., None]], -1)
            z = self.enc(x)[0][:, -1]
        else:
            x = torch.cat([self.cls.expand(B * D, -1, -1), self.proj(h)], 1) + self.pos
            keep = torch.cat([torch.ones(B * D, 1, device=m.device), m], 1) > 0  # CLS always kept
            z = self.enc(x, src_key_padding_mask=~keep)[:, 0]
        s = self.head(torch.cat([z, S.reshape(B * D, -1)], -1)).reshape(B, D)
        return s.masked_fill(~dm, NEG)


def _stack(races, keys, dev):
    H = torch.tensor(np.stack([races[k]["H"] for k in keys]), device=dev)
    M = torch.tensor(np.stack([races[k]["M"] for k in keys]), device=dev)
    S = torch.tensor(np.stack([races[k]["S"] for k in keys]), device=dev)
    dm = torch.tensor(np.stack([races[k]["dm"] for k in keys]), device=dev)
    y = torch.tensor([races[k]["y"] for k in keys], device=dev)
    return H, M, S, dm, y


def fit_predict(races, train_keys, test_keys, arch, use_static, seed, dev, epochs=60, patience=8):
    """Train on train_keys (last 15% held out for early stopping), return scores for test_keys: [T, dmax]."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_val = max(10, int(0.15 * len(train_keys)))
    tr_k, va_k = train_keys[:-n_val], train_keys[-n_val:]
    H, M, S, dm, y = _stack(races, tr_k, dev)
    # scalers from training races only
    hm = M.bool()
    hmu = H[hm].mean(0); hsd = H[hm].std(0).clamp_min(1e-6)
    sd_mask = dm.reshape(-1)
    S2 = S.reshape(-1, S.shape[-1])[sd_mask]
    smu = S2.mean(0); ssd = S2.std(0).clamp_min(1e-6)

    def prep(keys):
        H_, M_, S_, dm_, y_ = _stack(races, keys, dev)
        H_ = ((H_ - hmu) / hsd) * M_[..., None]
        S_ = ((S_ - smu) / ssd) * dm_[..., None]
        if not use_static:
            S_ = S_[..., :0]
        return H_, M_, S_, dm_, y_

    trd, vad, ted = prep(tr_k), prep(va_k), prep(test_keys)
    net = Net(arch, trd[2].shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()
    best, best_state, bad = 1e9, None, 0
    n = len(tr_k)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 32):
            ix = perm[i:i + 32]
            loss = ce(net(trd[0][ix], trd[1][ix], trd[2][ix], trd[3][ix]), trd[4][ix])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = ce(net(*vad[:4]), vad[4]).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(*ted[:4]).cpu().numpy()
