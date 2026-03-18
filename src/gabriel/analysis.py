from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in columns})


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    if method == "pearson":
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.empty:
        return float("nan")
    observed = float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean())
    labels = sorted(set(pair.iloc[:, 0]).union(set(pair.iloc[:, 1])))
    pa = pair.iloc[:, 0].value_counts(normalize=True).reindex(labels, fill_value=0.0)
    pb = pair.iloc[:, 1].value_counts(normalize=True).reindex(labels, fill_value=0.0)
    expected = float((pa * pb).sum())
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _macro_f1(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.empty:
        return float("nan")
    labels = sorted(set(pair.iloc[:, 0]).union(set(pair.iloc[:, 1])))
    scores: List[float] = []
    for lab in labels:
        tp = int(((pair.iloc[:, 0] == lab) & (pair.iloc[:, 1] == lab)).sum())
        fp = int(((pair.iloc[:, 0] != lab) & (pair.iloc[:, 1] == lab)).sum())
        fn = int(((pair.iloc[:, 0] == lab) & (pair.iloc[:, 1] != lab)).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _krippendorff_alpha_numeric(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    n_items = arr.shape[0]
    item_disagreements = []
    pooled = []
    for i in range(n_items):
        vals = arr[i, ~np.isnan(arr[i])]
        pooled.extend(vals.tolist())
        m = len(vals)
        if m <= 1:
            continue
        item_disagreements.append(np.sum((vals[:, None] - vals[None, :]) ** 2) / (m - 1))
    if not pooled:
        return float("nan")
    do = float(np.sum(item_disagreements) / max(1, len(item_disagreements))) if item_disagreements else 0.0
    pooled_arr = np.asarray(pooled, dtype=float)
    if len(pooled_arr) <= 1:
        return float("nan")
    de = float(np.sum((pooled_arr[:, None] - pooled_arr[None, :]) ** 2) / (len(pooled_arr) - 1))
    if de == 0:
        return 1.0
    return 1.0 - do / de


def _icc_oneway_random(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    valid_rows = mask.sum(axis=1) == arr.shape[1]
    arr = arr[valid_rows]
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return float("nan")
    n, k = arr.shape
    row_means = arr.mean(axis=1)
    grand = arr.mean()
    ms_between = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_within = np.sum((arr - row_means[:, None]) ** 2) / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    if denom == 0:
        return 1.0
    return float((ms_between - ms_within) / denom)


def reliability(
    data: pd.DataFrame | str,
    *,
    item_col: str = "id",
    run_col: str = "run",
    value_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in {item_col, run_col, "text", "entity_name"}]
    records = []
    for col in value_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        wide = pd.DataFrame({item_col: df[item_col], run_col: df[run_col], col: numeric}).pivot_table(index=item_col, columns=run_col, values=col, aggfunc="mean")
        vals = wide.to_numpy(dtype=float)
        records.append({
            "attribute": col,
            "krippendorff_alpha": _krippendorff_alpha_numeric(vals),
            "icc1": _icc_oneway_random(vals),
            "n_items": int(wide.shape[0]),
            "n_runs": int(wide.shape[1]),
        })
    return pd.DataFrame(records)


def validate(
    df: pd.DataFrame,
    *,
    gpt_col: str,
    human_col: str,
    task_type: str = "continuous",
    strata: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"task_type": task_type}
    if task_type == "continuous":
        a = pd.to_numeric(df[gpt_col], errors="coerce")
        b = pd.to_numeric(df[human_col], errors="coerce")
        pair = pd.concat([a, b], axis=1).dropna()
        out["metrics"] = {
            "pearson": _safe_corr(pair.iloc[:, 0], pair.iloc[:, 1], "pearson"),
            "spearman": _safe_corr(pair.iloc[:, 0], pair.iloc[:, 1], "spearman"),
            "mae": float((pair.iloc[:, 0] - pair.iloc[:, 1]).abs().mean()) if not pair.empty else float("nan"),
            "n": int(len(pair)),
        }
    else:
        pair = df[[gpt_col, human_col]].dropna()
        out["metrics"] = {
            "f1_macro": _macro_f1(pair[gpt_col], pair[human_col]),
            "cohen_kappa": _cohen_kappa(pair[gpt_col], pair[human_col]),
            "n": int(len(pair)),
        }
        out["confusion_matrix"] = pd.crosstab(pair[human_col], pair[gpt_col], dropna=False)
    if strata:
        strat_records = []
        for keys, sub in df.groupby(list(strata), dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            metrics = validate(sub, gpt_col=gpt_col, human_col=human_col, task_type=task_type)["metrics"]
            record = {col: key for col, key in zip(strata, keys)}
            record.update(metrics)
            strat_records.append(record)
        out["stratified"] = pd.DataFrame(strat_records)
    return out


def robustness(
    df: pd.DataFrame,
    *,
    score_col: str,
    group_col: str,
    reference_group: Optional[str] = None,
    n_bootstrap: int = 200,
    random_state: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    pivot = df.pivot_table(index=df.index, columns=group_col, values=score_col, aggfunc="mean")
    corr = pivot.corr(method="spearman")
    summary = df.groupby(group_col)[score_col].agg(["mean", "std", "count"]).reset_index()
    boot = []
    for group, sub in df.groupby(group_col):
        vals = pd.to_numeric(sub[score_col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            boot.append({group_col: group, "ci_lower": np.nan, "ci_upper": np.nan})
            continue
        means = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(max(1, n_bootstrap))]
        boot.append({group_col: group, "ci_lower": float(np.quantile(means, 0.025)), "ci_upper": float(np.quantile(means, 0.975))})
    out = {"group_summary": summary.merge(pd.DataFrame(boot), on=group_col, how="left"), "correlation": corr}
    if reference_group is not None and reference_group in set(df[group_col]):
        ref = pd.to_numeric(df.loc[df[group_col] == reference_group, score_col], errors="coerce")
        compare = []
        for group, sub in df.groupby(group_col):
            vals = pd.to_numeric(sub[score_col], errors="coerce")
            compare.append({group_col: group, "mean_difference_vs_reference": float(vals.mean() - ref.mean())})
        out["vs_reference"] = pd.DataFrame(compare)
    return out
