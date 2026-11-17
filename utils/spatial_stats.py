"""Spatially adjusted correlation inference (Moran's I & Dutilleul / CRH N_eff).

Implements exact Clifford-Richardson-Hemon (1989) / Dutilleul (1993) modified
effective sample size (N_eff) and spatially adjusted two-sided p-values for
country-level Spearman rank correlations using k-nearest spherical neighbors
(great-circle distance on country centroids) and simultaneous autoregressive
(SAR) maximum-likelihood spatial covariance matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr, t as student_t


def build_spherical_knn_weights(
    lat_deg: np.ndarray, lon_deg: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
  """Builds row-standardized k-NN spherical weight matrix W and great-circle dist matrix (km)."""
  lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
  lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
  n = len(lat)
  k_eff = min(k, max(1, n - 1))

  # Exact unit-sphere 3D coordinates -> exact great-circle distance
  xyz = np.column_stack([
      np.cos(lat) * np.cos(lon),
      np.cos(lat) * np.sin(lon),
      np.sin(lat),
  ])
  cos_ang = np.clip(xyz @ xyz.T, -1.0, 1.0)
  dist_km = 6371.0088 * np.arccos(cos_ang)
  np.fill_diagonal(dist_km, np.inf)

  W = np.zeros((n, n), dtype=np.float64)
  nn_idx = np.argpartition(dist_km, kth=k_eff, axis=1)[:, :k_eff]
  for i in range(n):
    W[i, nn_idx[i]] = 1.0 / k_eff
  np.fill_diagonal(dist_km, 0.0)
  return W, dist_km


def morans_i(v: np.ndarray, W: np.ndarray) -> float:
  """Computes global Moran's I for vector v under row-standardized spatial weights W."""
  z = np.asarray(v, dtype=np.float64)
  z = z - np.mean(z)
  denom = float(z @ z)
  if denom <= 0.0:
    return 0.0
  s0 = float(W.sum())
  if s0 <= 0.0:
    return 0.0
  return float((len(z) / s0) * (z @ (W @ z)) / denom)


def _fit_sar_rho_mle(z: np.ndarray, W: np.ndarray, eigvals: np.ndarray) -> float:
  """Fits simultaneous autoregressive (SAR) spatial parameter rho in [0, 0.95] via exact profile MLE."""
  z_c = z - np.mean(z)
  n = len(z_c)
  if float(z_c @ z_c) <= 1e-15:
    return 0.0
  Wz = W @ z_c
  I_val = morans_i(z_c, W)
  if I_val <= 0.0:
    return 0.0

  # Upper bound strictly inside 1 / max(real(eigval)) for invertibility
  max_ev = float(np.max(np.real(eigvals)))
  hi = min(0.95, (0.98 / max_ev) if max_ev > 0 else 0.95)

  def neg_loglik(rho: float) -> float:
    e = z_c - rho * Wz
    sse = float(e @ e)
    if sse <= 1e-30:
      return 1e30
    # Exact log|det(I - rho W)| = sum log|1 - rho * lambda_i|
    logdet = float(np.sum(np.log(np.abs(1.0 - rho * eigvals))))
    return 0.5 * n * np.log(sse / n) - logdet

  res = minimize_scalar(neg_loglik, bounds=(0.0, hi), method='bounded')
  return float(np.clip(res.x, 0.0, hi))


def _sar_corr_matrix(rho: float, W: np.ndarray) -> np.ndarray:
  """Returns correlation matrix implied by SAR covariance (I - rho W)^{-1} (I - rho W)^{-T}."""
  n = W.shape[0]
  if rho <= 1e-8:
    return np.eye(n, dtype=np.float64)
  A = np.eye(n, dtype=np.float64) - rho * W
  invA = np.linalg.inv(A)
  cov = invA @ invA.T
  sd = np.sqrt(np.maximum(np.diag(cov), 1e-15))
  return cov / np.outer(sd, sd)


def spatial_spearman_neff(
    x: np.ndarray,
    y: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
  """Computes Spearman r, naive p, Moran's I (x & y), Dutilleul/CRH N_eff, and p_spatial.

  Rank-transforms both x and y before fitting SAR spatial correlation matrices R_x, R_y
  on k-NN spherical weights W, then applies the exact Dutilleul (1993) / Clifford-Richardson-
  Hemon (1989) variance formula:
      Var(r) = tr(B R_x B R_y) / (tr(B R_x) * tr(B R_y)),  N_eff = 1 + 1 / Var(r)
  where B = I - (1/N) 1 1^T, yielding df_eff = N_eff - 2 and two-sided Student-t p_spatial.
  """
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  lat = np.asarray(lat_deg, dtype=np.float64)
  lon = np.asarray(lon_deg, dtype=np.float64)
  ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
  x, y, lat, lon = x[ok], y[ok], lat[ok], lon[ok]
  n = int(len(x))
  if n < 6:
    return {
        'n': n,
        'spearman_r': np.nan,
        'p_naive': np.nan,
        'morans_i_x': np.nan,
        'morans_i_y': np.nan,
        'sar_rho_x': np.nan,
        'sar_rho_y': np.nan,
        'n_eff': np.nan,
        'p_spatial': np.nan,
    }

  rs, p_naive = spearmanr(x, y)
  rs = float(rs)
  p_naive = float(p_naive)

  rx = rankdata(x)
  ry = rankdata(y)
  W, _ = build_spherical_knn_weights(lat, lon, k=k)
  ix = morans_i(rx, W)
  iy = morans_i(ry, W)

  eigvals = np.linalg.eigvals(W)
  rho_x = _fit_sar_rho_mle(rx, W, eigvals)
  rho_y = _fit_sar_rho_mle(ry, W, eigvals)

  Rx = _sar_corr_matrix(rho_x, W)
  Ry = _sar_corr_matrix(rho_y, W)

  # Centering matrix B = I - (1/n) 1 1^T -> B M = M - col_mean(M)
  BRx = Rx - Rx.mean(axis=0, keepdims=True)
  BRy = Ry - Ry.mean(axis=0, keepdims=True)
  tr_x = float(np.trace(BRx))
  tr_y = float(np.trace(BRy))
  # tr(A B) = sum(A * B.T)
  tr_xy = float(np.sum(BRx * BRy.T))

  if tr_xy > 1e-12 and tr_x > 1e-12 and tr_y > 1e-12:
    n_eff = float(np.clip(1.0 + (tr_x * tr_y) / tr_xy, 3.0, float(n)))
  else:
    n_eff = float(n)

  df_eff = max(1.0, n_eff - 2.0)
  if abs(rs) >= 1.0 - 1e-12:
    p_spatial = 0.0
  else:
    t_stat = abs(rs) * np.sqrt(df_eff / (1.0 - rs * rs))
    p_spatial = float(2.0 * student_t.sf(t_stat, df=df_eff))

  return {
      'n': n,
      'k_nn': int(k),
      'spearman_r': rs,
      'p_naive': p_naive,
      'morans_i_x': float(ix),
      'morans_i_y': float(iy),
      'sar_rho_x': float(rho_x),
      'sar_rho_y': float(rho_y),
      'n_eff': float(n_eff),
      'p_spatial': float(p_spatial),
  }
