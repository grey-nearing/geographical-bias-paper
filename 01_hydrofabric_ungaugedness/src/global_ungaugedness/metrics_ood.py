"""Family 5: Hydroclimatic / Feature-Space Representation (OOD) Metrics.

Computes strictly scale-invariant (feature-space) out-of-distribution anomaly
scores measuring how well each gauge tier covers the global hydroclimatic
attribute distribution of land catchments:
  - Standardized Euclidean k-NN distance (k=1 and k=5) in globally standardized
    hydroclimatic + physiographic feature space (mean-imputed and z-scored on
    the global target domain so scales are identical across tiers).
  - Fraction of land area, runoff, population, and flood exposure falling
    outside the gauge tier's 95th and 99th percentile in-distribution k-NN
    distance envelope (leave-one-out self-distances among training gauges).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from global_ungaugedness.metrics_2d import weighted_quantiles


def standardize_features(
    X_target: np.ndarray,
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaNs with global target medians and z-score using global target mean/std."""
    Xt = np.asarray(X_target, dtype=np.float64).copy()
    Xg = np.asarray(X_train, dtype=np.float64).copy()

    col_med = np.nanmedian(Xt, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)

    for j in range(Xt.shape[1]):
        bad_t = ~np.isfinite(Xt[:, j])
        if np.any(bad_t):
            Xt[bad_t, j] = col_med[j]
        bad_g = ~np.isfinite(Xg[:, j])
        if np.any(bad_g):
            Xg[bad_g, j] = col_med[j]

    mu = np.mean(Xt, axis=0)
    sd = np.std(Xt, axis=0)
    keep = sd > 1e-12
    Xt = (Xt[:, keep] - mu[keep]) / sd[keep]
    Xg = (Xg[:, keep] - mu[keep]) / sd[keep]
    return Xt, Xg


def compute_ood_metrics(
    X_target: np.ndarray,
    X_train: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    domain_mask: np.ndarray | None = None,
    k_neighbors: tuple[int, ...] = (1, 5),
    workers: int = 8,
    target_group_ids: np.ndarray | None = None,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Compute Family 5 hydroclimatic OOD representation metrics.

    Returns:
        records: Tidy metric summary rows across all weights.
        dists_by_k: Dict mapping k -> per-unit mean k-NN standardized distance array.
    """
    n = X_target.shape[0]
    if X_train.shape[0] < 2:
        return [], {}

    Xt, Xg = standardize_features(X_target, X_train)
    # Normalize by sqrt(D) so distances are dimensionless RMS z-scores per feature
    scale = np.sqrt(float(Xt.shape[1]))
    Xt /= scale
    Xg /= scale

    tree = cKDTree(Xg)
    max_k = min(max(k_neighbors), Xg.shape[0])

    # Target distances to training gauges (exact deduplication by group ID if provided)
    if target_group_ids is not None and len(target_group_ids) == n:
        _, first_idx, inv_idx = np.unique(target_group_ids, return_index=True, return_inverse=True)
        d_uniq, _ = tree.query(Xt[first_idx], k=max_k, workers=workers)
        if max_k == 1:
            d_uniq = d_uniq[:, None]
        d_target = d_uniq[inv_idx]
    else:
        d_target, _ = tree.query(Xt, k=max_k, workers=workers)
        if max_k == 1:
            d_target = d_target[:, None]

    # Leave-one-out in-distribution distances among training gauges (k + 1 to skip self)
    loo_k = min(max_k + 1, Xg.shape[0])
    d_train_loo, _ = tree.query(Xg, k=loo_k, workers=workers)
    if loo_k == 1:
        d_train_loo = d_train_loo[:, None]

    m = np.ones(n, dtype=bool) if domain_mask is None else np.asarray(domain_mask, dtype=bool)
    idx = np.flatnonzero(m)

    records: list[dict] = []
    dists_by_k: dict[int, np.ndarray] = {}

    for k in k_neighbors:
        k_eff = min(k, max_k)
        dk_target = np.mean(d_target[:, :k_eff], axis=1)
        dists_by_k[k] = dk_target

        # LOO distances: drop col 0 (self distance 0) if available
        if loo_k > 1:
            dk_loo = np.mean(d_train_loo[:, 1 : min(k_eff + 1, loo_k)], axis=1)
        else:
            dk_loo = d_train_loo[:, 0]

        p95_thresh = float(np.percentile(dk_loo, 95))
        p99_thresh = float(np.percentile(dk_loo, 99))

        dk_m = dk_target[idx]
        for w_name, w_full in weights_dict.items():
            if w_full is None:
                continue
            w = np.asarray(w_full[idx], dtype=np.float64)
            w_tot = float(np.sum(w))
            if w_tot <= 0.0:
                continue

            q25, q50, q75 = weighted_quantiles(dk_m, w, (0.25, 0.50, 0.75))
            mean_d = float(np.dot(w, dk_m) / w_tot)
            frac_gt_p95 = float(np.sum(w[dk_m > p95_thresh]) / w_tot)
            frac_gt_p99 = float(np.sum(w[dk_m > p99_thresh]) / w_tot)

            for metric_name, val in [
                (f"U_OOD_knn{k}_mean", mean_d),
                (f"U_OOD_knn{k}_q25", q25),
                (f"U_OOD_knn{k}_q50", q50),
                (f"U_OOD_knn{k}_q75", q75),
                (f"U_OOD_knn{k}_frac_gt_train_p95", frac_gt_p95),
                (f"U_OOD_knn{k}_frac_gt_train_p99", frac_gt_p99),
            ]:
                records.append({
                    "family": "F5_hydroclimatic_ood",
                    "metric": metric_name,
                    "ratio_basis": "feature_space",
                    "snapping_mode": "standardized_knn",
                    "weight": w_name,
                    "scale_param": float(k),
                    "value": float(val),
                    "total_weight": w_tot,
                })

    return records, dists_by_k
