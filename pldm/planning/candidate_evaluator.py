"""Candidate-ranking and counterfactual-oracle diagnostics (ECCHP stage 2).

Stage 2 of the research plan asks one question before any new mechanism is
built: *does the original HWM miss high-level plans that would have been
better under real execution?*  Answering it needs three quantities per decision
state, each computed from a small set of candidates ``P_0 ... P_{K-1}``:

* ``J_H(P_i)`` / ``C_L(P_i)`` -- what the model thinks (lower is better);
* ``R_H(P_i)``               -- what real execution delivered (higher is better,
  e.g. the reduction of the true graph distance over a fixed horizon).

This module contains the pure statistics, deliberately free of any planner or
environment import so they can be unit-tested and reused by both the collection
and the analysis script:

* :func:`oracle_candidate_gain` -- ``max_i R_H(P_i) - R_H(P_baseline)``;
* :func:`kendall_tau` -- rank correlation between model score and real return;
* :func:`pairwise_accuracy` -- fraction of candidate pairs ordered correctly;
* :func:`best_candidate_hit` -- whether the model's top pick is the real best;
* :func:`bootstrap_mean_ci` / :func:`paired_bootstrap_diff_ci` -- confidence
  intervals for the aggregate numbers.

Conventions
-----------
* tensors are ``(K, B)``: ``K`` candidates, ``B`` environments;
* ``model_scores`` are *costs* (lower is better), ``real_returns`` are
  *rewards* (higher is better);
* undefined statistics (``K < 2``, all ties) are returned as ``nan`` rather
  than silently becoming zero, so the analysis can count them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor


def _as_candidate_matrix(value: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.detach().to(dtype=torch.float64)
    if value.dim() == 1:
        value = value.unsqueeze(1)
    if value.dim() != 2:
        raise ValueError(
            f"{name} must have shape (K, B), got {tuple(value.shape)}"
        )
    return value


def _pair_indices(k: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


# --------------------------------------------------------------------------- #
# Oracle gain
# --------------------------------------------------------------------------- #
def oracle_candidate_gain(
    real_returns: Tensor, baseline_index: int = 0
) -> Tensor:
    """``max_i R_H(P_i) - R_H(P_baseline)`` per environment.

    A positive value means the candidate set contained a plan that executed
    better than the plan the original planner chose.  That is the direct
    evidence stage 2 is looking for; a value of ~0 means re-ranking could not
    have helped *on this decision*, no matter how good the ranker is.
    """
    returns = _as_candidate_matrix(real_returns, "real_returns")
    baseline = returns[int(baseline_index)]
    best = torch.nan_to_num(returns, nan=float("-inf")).max(dim=0).values
    gain = best - baseline
    return torch.where(torch.isfinite(gain), gain, torch.full_like(gain, float("nan")))


def best_candidate_index(real_returns: Tensor) -> Tensor:
    """Index of the real best candidate per environment (``-1`` if all NaN)."""
    returns = _as_candidate_matrix(real_returns, "real_returns")
    masked = returns.clone()
    masked[~torch.isfinite(masked)] = float("-inf")
    best = masked.max(dim=0).values
    index = masked.argmax(dim=0)
    return torch.where(torch.isfinite(best), index, torch.full_like(index, -1))


# --------------------------------------------------------------------------- #
# Ranking agreement
# --------------------------------------------------------------------------- #
def kendall_tau(model_scores: Tensor, real_returns: Tensor) -> Tensor:
    """Kendall ``tau-b`` between model ranking and real returns, per env.

    ``model_scores`` is a cost, so its ranking is reversed before comparison.
    Ties are handled with the ``tau-b`` correction; a candidate pair tied in
    both quantities carries no information and is skipped.
    """
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    if scores.shape != returns.shape:
        raise ValueError(
            "model_scores and real_returns must have the same shape, got "
            f"{tuple(scores.shape)} and {tuple(returns.shape)}"
        )
    k, batch = scores.shape
    if k < 2:
        return torch.full((batch,), float("nan"), dtype=torch.float64)

    # Higher score value = worse plan, so flip the sign to get "higher is
    # better" on both sides.
    model_rank = -scores
    concordant = torch.zeros(batch, dtype=torch.float64)
    discordant = torch.zeros(batch, dtype=torch.float64)
    ties_model = torch.zeros(batch, dtype=torch.float64)
    ties_real = torch.zeros(batch, dtype=torch.float64)

    for i, j in _pair_indices(k):
        d_model = model_rank[i] - model_rank[j]
        d_real = returns[i] - returns[j]
        valid = torch.isfinite(d_model) & torch.isfinite(d_real)
        sign_model = torch.sign(d_model)
        sign_real = torch.sign(d_real)
        agree = valid & (sign_model * sign_real > 0)
        disagree = valid & (sign_model * sign_real < 0)
        model_tie = valid & (sign_model == 0) & (sign_real != 0)
        real_tie = valid & (sign_real == 0) & (sign_model != 0)
        concordant += agree.to(torch.float64)
        discordant += disagree.to(torch.float64)
        ties_model += model_tie.to(torch.float64)
        ties_real += real_tie.to(torch.float64)

    numerator = concordant - discordant
    denom = torch.sqrt(
        (concordant + discordant + ties_model).clamp_min(0.0)
        * (concordant + discordant + ties_real).clamp_min(0.0)
    )
    tau = torch.where(
        denom > 0,
        numerator / denom.clamp_min(1e-12),
        torch.full_like(numerator, float("nan")),
    )
    return tau


def pairwise_accuracy(model_scores: Tensor, real_returns: Tensor) -> Tensor:
    """Fraction of candidate pairs the model orders correctly, per env.

    Only pairs with a strict real-order are counted; ties are ignored rather
    than counted as half-correct, because a tie carries no preference.
    """
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    if scores.shape != returns.shape:
        raise ValueError("model_scores and real_returns must have equal shapes")
    k, batch = scores.shape
    if k < 2:
        return torch.full((batch,), float("nan"), dtype=torch.float64)

    model_rank = -scores
    correct = torch.zeros(batch, dtype=torch.float64)
    counted = torch.zeros(batch, dtype=torch.float64)
    for i, j in _pair_indices(k):
        d_model = model_rank[i] - model_rank[j]
        d_real = returns[i] - returns[j]
        valid = torch.isfinite(d_model) & torch.isfinite(d_real) & (d_real != 0)
        agree = valid & (torch.sign(d_model) == torch.sign(d_real))
        correct += agree.to(torch.float64)
        counted += valid.to(torch.float64)
    result = torch.where(
        counted > 0, correct / counted.clamp_min(1e-12), torch.full_like(correct, float("nan"))
    )
    return result


def best_candidate_hit(model_scores: Tensor, real_returns: Tensor) -> Tensor:
    """Whether the model's cheapest candidate is the real best one.

    Returns ``nan`` for environments whose real returns are all undefined.
    """
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    if scores.shape != returns.shape:
        raise ValueError("model_scores and real_returns must have equal shapes")
    masked_scores = scores.clone()
    masked_scores[~torch.isfinite(masked_scores)] = float("inf")
    chosen = masked_scores.argmin(dim=0)
    best = best_candidate_index(returns)
    hit = (chosen == best).to(torch.float64)
    return torch.where(best >= 0, hit, torch.full_like(hit, float("nan")))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def nanmean(value: Tensor) -> float:
    """Mean ignoring ``nan``; ``nan`` when nothing is finite."""
    value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(value)
    if not bool(finite.any()):
        return float("nan")
    return float(value[finite].mean().item())


def nanmedian(value: Tensor) -> float:
    value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(value)
    if not bool(finite.any()):
        return float("nan")
    return float(value[finite].median().item())


def bootstrap_mean_ci(
    values: Tensor,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean of ``values`` (NaN-tolerant).

    Deterministic given ``seed``: the same inputs always produce the same
    interval, which matters because these numbers end up in the paper.
    """
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    values = values[torch.isfinite(values)]
    n = int(values.numel())
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean = float(values.mean().item())
    if n == 1 or n_boot <= 0:
        return {"mean": mean, "lo": mean, "hi": mean, "n": n}

    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randint(0, n, (int(n_boot), n), generator=generator)
    means = values[index].mean(dim=1)
    lo = float(torch.quantile(means, alpha / 2.0).item())
    hi = float(torch.quantile(means, 1.0 - alpha / 2.0).item())
    return {"mean": mean, "lo": lo, "hi": hi, "n": n}


def paired_bootstrap_diff_ci(
    a: Tensor,
    b: Tensor,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap CI of ``mean(a - b)`` for paired observations.

    Used for questions of the form "are the candidates better than the plan the
    original planner picked?" where both numbers come from the same task.
    """
    a = torch.as_tensor(a, dtype=torch.float64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(
            f"paired inputs must have the same length, got {a.numel()} vs {b.numel()}"
        )
    diff = a - b
    valid = torch.isfinite(diff)
    diff = diff[valid]
    n = int(diff.numel())
    if n == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean_diff = float(diff.mean().item())
    if n == 1 or n_boot <= 0:
        return {"mean_diff": mean_diff, "lo": mean_diff, "hi": mean_diff, "n": n}
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randint(0, n, (int(n_boot), n), generator=generator)
    means = diff[index].mean(dim=1)
    return {
        "mean_diff": mean_diff,
        "lo": float(torch.quantile(means, alpha / 2.0).item()),
        "hi": float(torch.quantile(means, 1.0 - alpha / 2.0).item()),
        "n": n,
    }


def candidate_ranking_summary(
    model_scores: Tensor,
    real_returns: Tensor,
    *,
    baseline_index: int = 0,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """All stage-2 statistics in one dictionary.

    Keys are stable so the analysis script can write them to JSON/CSV without
    knowing which of them happened to be defined.
    """
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    gain = oracle_candidate_gain(returns, baseline_index=baseline_index)
    tau = kendall_tau(scores, returns)
    accuracy = pairwise_accuracy(scores, returns)
    hit = best_candidate_hit(scores, returns)
    gain_ci = bootstrap_mean_ci(gain, n_boot=n_boot, seed=seed)
    return {
        "n_decisions": int(gain.numel()),
        "n_candidates": int(scores.shape[0]),
        "oracle_gain_mean": gain_ci["mean"],
        "oracle_gain_lo": gain_ci["lo"],
        "oracle_gain_hi": gain_ci["hi"],
        "oracle_gain_median": nanmedian(gain),
        "oracle_gain_positive_rate": _rate(gain > 0),
        "kendall_tau_mean": nanmean(tau),
        "pairwise_accuracy_mean": nanmean(accuracy),
        "best_candidate_hit_rate": nanmean(hit),
    }


def _rate(mask: Tensor) -> float:
    mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if mask.numel() == 0:
        return float("nan")
    return float(mask.to(torch.float64).mean().item())


def group_summary(
    groups: Iterable[str],
    model_scores: Tensor,
    real_returns: Tensor,
    *,
    baseline_index: int = 0,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """``candidate_ranking_summary`` computed separately per group label.

    ``groups`` labels each *decision* (i.e. each column of the ``(K, B)``
    matrices); stage 2 is reported separately for failed and successful tasks.
    """
    groups = list(groups)
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    if len(groups) != scores.shape[1]:
        raise ValueError(
            f"got {len(groups)} group labels for {scores.shape[1]} decisions"
        )
    unique = sorted({str(g) for g in groups})
    out: Dict[str, Dict[str, float]] = {}
    for label in unique:
        index = [i for i, g in enumerate(groups) if str(g) == label]
        if not index:
            continue
        out[label] = candidate_ranking_summary(
            scores[:, index],
            returns[:, index],
            baseline_index=baseline_index,
            n_boot=n_boot,
            seed=seed,
        )
    return out


def selection_utility(
    model_scores: Tensor, real_returns: Tensor, baseline_index: int = 0
) -> Dict[str, float]:
    """**Regret-style** view: what would *choosing by the model* have delivered?

    ``regret = R_H(P_baseline) - R_H(argmin_J model_score)`` (positive = the
    model's pick executed worse than the original planner's pick).  Reported
    next to the oracle gain so the two failure modes stay separable:

    * large oracle gain + large regret  -> the candidates are good, ranking is bad;
    * small oracle gain                 -> the candidates themselves are the
      bottleneck, and a better ranker cannot help.
    """
    scores = _as_candidate_matrix(model_scores, "model_scores")
    returns = _as_candidate_matrix(real_returns, "real_returns")
    masked = scores.clone()
    masked[~torch.isfinite(masked)] = float("inf")
    chosen = masked.argmin(dim=0)
    picked = returns.gather(0, chosen.view(1, -1)).squeeze(0)
    baseline = returns[int(baseline_index)]
    regret = baseline - picked
    finite = torch.isfinite(regret)
    return {
        "model_pick_regret_mean": nanmean(regret),
        "model_pick_regret_median": nanmedian(regret),
        "model_pick_worse_rate": _rate(finite & (regret > 0)),
        "model_pick_better_rate": _rate(finite & (regret < 0)),
    }


# --------------------------------------------------------------------------- #
# I/O helpers shared by the collection / oracle scripts
# --------------------------------------------------------------------------- #
def read_candidate_groups(path: Any) -> List[Dict[str, Any]]:
    """Load the tidy candidate table written by ``collect_plan_candidates.py``.

    Each line is one decision::

        {"task_id": 4, "decision_id": 3, "horizon": 40,
         "candidates": [{"candidate_id": 0, "model_score": 0.8,
                         "real_return": 2.0, ...}, ...]}

    Lines that do not match this schema raise ``ValueError``; silently skipping
    them would bias the oracle statistics.
    """
    import json
    from pathlib import Path

    path = Path(path)
    groups: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "candidates" not in record or not isinstance(
                record["candidates"], list
            ):
                raise ValueError(
                    f"{path}:{line_number} is not a candidate group record"
                )
            groups.append(record)
    return groups


def groups_to_matrices(
    groups: Sequence[Dict[str, Any]],
    *,
    horizon: Optional[int] = None,
    score_key: str = "model_score",
    return_key: str = "real_return",
) -> Tuple[Tensor, Tensor, List[int], List[int]]:
    """Stack candidate groups into ``(K, B)`` score / return matrices.

    Groups with different candidate counts are padded with ``nan``, which the
    statistics above treat as "not evaluated" rather than as zero.
    ``NaN`` groups (e.g. all candidates unlabelled) are still returned so the
    caller can report coverage instead of silently dropping decisions.
    """
    if horizon is not None:
        groups = [g for g in groups if int(g.get("horizon", -1)) == int(horizon)]
    if not groups:
        empty = torch.zeros((0, 0), dtype=torch.float64)
        return empty, empty, [], []

    k_max = max(len(g.get("candidates") or []) for g in groups)
    batch = len(groups)
    scores = torch.full((k_max, batch), float("nan"), dtype=torch.float64)
    returns = torch.full((k_max, batch), float("nan"), dtype=torch.float64)
    task_ids: List[int] = []
    decision_ids: List[int] = []
    for column, group in enumerate(groups):
        task_ids.append(int(group.get("task_id", -1)))
        decision_ids.append(int(group.get("decision_id", -1)))
        for row, candidate in enumerate(group.get("candidates") or []):
            score = candidate.get(score_key)
            value = candidate.get(return_key)
            if score is not None:
                scores[row, column] = float(score)
            if value is not None:
                returns[row, column] = float(value)
    return scores, returns, task_ids, decision_ids


def label_by_task(task_ids: Sequence[int]) -> List[str]:
    """Group label per decision: ``task_<id>``."""
    return [f"task_{int(task_id)}" for task_id in task_ids]

