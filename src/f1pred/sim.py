"""Lap-by-lap Monte Carlo race simulator.

Each of S simulated races draws, per driver, a race-pace shock around the pace-model
prediction, a pit strategy resampled from history at that circuit, a retirement lap, and
shared safety-car events. Cars then race lap by lap: tyre wear, pit-stop loss, SC bunching
(and cheap SC stops), and a probabilistic overtaking rule that makes a faster car sit
behind a slower one until it is quick enough to pass.

Winner probabilities are simply the share of simulated races each driver wins.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

SC_LAP_FACTOR = 1.40  # lap time multiplier behind the safety car
SC_GAP = 1.0  # seconds between cars once bunched
FOLLOW_GAP = 0.6  # seconds a held-up car sits behind the car in front


@dataclass
class Params:
    """Physics knobs. pass_thr/pass_scale/start_sigma are calibrated on 2019-21 then frozen."""

    pass_thr: float = 0.006  # per-lap pace advantage (fraction of a lap) for a 50% pass
    pass_scale: float = 0.003
    start_sigma: float = 0.9  # seconds of random start/first-lap time shuffle
    start_slot_gap: float = 0.18  # seconds per grid slot
    lap_sigma: float = 0.0035  # lap-to-lap noise (fraction)
    pace_sigma_mult: float = 1.0  # scales the pace-model residual spread
    sc_pit_prob: float = 0.55  # chance an eligible car dives into the pits under SC
    pit_loss_sd: float = 1.2
    strategy_share: float = 0.0  # prob. a car copies the race's main strategy (0 = fully independent, v1 behaviour)
    s: int = 2000


@dataclass
class RaceSetup:
    drivers: list[str]
    grid: np.ndarray  # starting slot, 1..D
    pace: np.ndarray  # predicted pace deficit (fraction), >= 0
    pace_sigma: float  # residual sd of the pace model
    p_retire: np.ndarray
    total_laps: int
    base_lap_s: float
    deg: float  # fractional slow-down per lap of tyre age
    pit_loss: float  # seconds
    strategies: list[list[float]] = field(default_factory=list)  # pit lap fractions, sampled from history
    sc_counts: list[int] = field(default_factory=lambda: [0])
    sc_starts: list[float] = field(default_factory=lambda: [0.3])
    sc_lens: list[int] = field(default_factory=lambda: [4])
    force_sc: bool | None = None  # what-if: True = guaranteed SC, False = none


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(setup: RaceSetup, prm: Params, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    S, D, L = prm.s, len(setup.drivers), setup.total_laps
    base = setup.base_lap_s

    # --- per-sim draws ------------------------------------------------------------------
    pace = setup.pace[None, :] + rng.normal(0, setup.pace_sigma * prm.pace_sigma_mult, (S, D))
    pace -= pace.min(axis=1, keepdims=True)  # fastest car this race defines zero
    t = (setup.grid[None, :] - 1) * prm.start_slot_gap + rng.normal(0, prm.start_sigma, (S, D)) * (
        0.6 + 0.04 * setup.grid[None, :]
    )
    age = np.zeros((S, D))

    # retirements: a quarter happen at the start (collisions), the rest uniformly
    retire = rng.random((S, D)) < setup.p_retire[None, :]
    first = rng.random((S, D)) < 0.25
    rlap = np.where(first, 1, rng.integers(2, L + 1, (S, D)))
    dnf_lap = np.where(retire, rlap, 10**9)

    # pit plans sampled from history (jittered); padded with a sentinel
    K = 4
    plan = np.full((S, D, K), 10**9)
    strat = setup.strategies or [[0.4]]
    pick = rng.integers(0, len(strat), (S, D))
    if prm.strategy_share > 0:
        main = rng.integers(0, len(strat), (S, 1))
        pick = np.where(rng.random((S, D)) < prm.strategy_share, main, pick)
    for si, fr in enumerate(strat):
        m = pick == si
        if not fr or not m.any():
            continue
        laps = np.round(np.array(fr[:K]) * L).astype(int)
        jitter = rng.integers(-2, 3, (S, D, len(laps)))
        plan[..., : len(laps)] = np.where(m[..., None], np.clip(laps[None, None, :] + jitter, 2, L - 2), plan[..., : len(laps)])
    plan.sort(axis=2)
    stops_done = np.zeros((S, D), dtype=int)

    # safety-car events (shared by all cars in a sim)
    n_ev = rng.choice(setup.sc_counts, S)
    if setup.force_sc is True:
        n_ev = np.maximum(n_ev, 1)
    elif setup.force_sc is False:
        n_ev = np.zeros(S, dtype=int)
    n_ev = np.minimum(n_ev, 3)
    sc_start = np.zeros((S, 3), dtype=int)
    sc_len = np.zeros((S, 3), dtype=int)
    for e in range(3):
        has = n_ev > e
        st = np.maximum(2, np.round(rng.choice(setup.sc_starts, S) * L).astype(int))
        if setup.force_sc is True and e == 0:
            st = np.maximum(2, np.round(rng.uniform(0.15, 0.75, S) * L).astype(int))
        ln = rng.choice(setup.sc_lens, S)
        sc_start[:, e] = np.where(has, np.minimum(st, L - 3), 0)
        sc_len[:, e] = np.where(has, ln, 0)

    alive = np.ones((S, D), dtype=bool)
    rows = np.arange(S)[:, None]

    for lap in range(1, L + 1):
        alive &= dnf_lap > lap
        in_sc = np.zeros(S, dtype=bool)
        sc_first = np.zeros(S, dtype=bool)
        for e in range(3):
            active = (sc_len[:, e] > 0) & (lap >= sc_start[:, e]) & (lap < sc_start[:, e] + sc_len[:, e])
            in_sc |= active
            sc_first |= (sc_len[:, e] > 0) & (lap == sc_start[:, e])

        lt = base * (1.0 + pace + setup.deg * age + rng.normal(0, prm.lap_sigma, (S, D)))
        lt = np.where(in_sc[:, None], base * SC_LAP_FACTOR, lt)

        # pit stops: planned, or opportunistic under the safety car
        planned = (plan[np.arange(S)[:, None], np.arange(D)[None, :], np.minimum(stops_done, K - 1)] == lap) & (stops_done < K)
        opportunistic = sc_first[:, None] & (age >= 6) & (stops_done < K) & (rng.random((S, D)) < prm.sc_pit_prob)
        pit = (planned | opportunistic) & alive
        loss = setup.pit_loss * np.where(in_sc[:, None], 0.45, 1.0) + rng.normal(0, prm.pit_loss_sd, (S, D))
        lt = lt + np.where(pit, loss, 0.0)
        stops_done = stops_done + pit
        age = np.where(pit, 0.0, age + 1.0)

        t_new = np.where(alive, t + lt, np.inf)

        # --- overtaking resolution, front to back -----------------------------------
        order = np.argsort(t, axis=1)  # positions at the start of the lap
        ts = np.take_along_axis(t_new, order, 1)
        lts = np.take_along_axis(lt, order, 1)
        pits = np.take_along_axis(pit, order, 1)
        res = ts.copy()
        for k in range(1, D):
            prev = res[:, k - 1]
            clash = ts[:, k] < prev + FOLLOW_GAP
            free = pits[:, k] | pits[:, k - 1] | in_sc | ~np.isfinite(prev)
            adv = (lts[:, k - 1] - lts[:, k]) / base
            passed = rng.random(S) < _sigmoid((adv - prm.pass_thr) / prm.pass_scale)
            blocked = clash & ~free & ~passed
            res[:, k] = np.where(blocked, prev + FOLLOW_GAP, ts[:, k])
        # safety car bunches the field
        if in_sc.any():
            srt = np.sort(res, axis=1)
            for k in range(1, D):
                cap = srt[:, k - 1] + SC_GAP
                srt[:, k] = np.where(in_sc & np.isfinite(srt[:, k]), np.minimum(srt[:, k], cap), srt[:, k])
            # map bunched times back by rank
            rank = np.argsort(np.argsort(res, axis=1), axis=1)
            res = np.take_along_axis(srt, rank, 1)
        t_new_sorted = res
        t = np.empty_like(t_new)
        np.put_along_axis(t, order, t_new_sorted, 1)

    finish = np.argsort(t, axis=1)
    winner = finish[:, 0]
    wins = np.bincount(winner, minlength=D) / S
    podium = np.stack([np.bincount(finish[:, j], minlength=D) / S for j in range(3)]).sum(0)
    pos = np.argsort(finish, axis=1) + 1
    return dict(win=wins, podium=podium, exp_pos=pos.mean(0), had_sc=float((n_ev > 0).mean()), winner_idx=winner)


def laplace(p: np.ndarray, s: int, floor_mass: float = 0.5) -> np.ndarray:
    """Smooth Monte-Carlo shares so a driver who never won in simulation still gets mass."""
    q = (p * s + floor_mass) / (s + floor_mass * len(p))
    return q / q.sum()
