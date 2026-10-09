"""Deterministic outlier detection — pure functions, zero side effects.

No network, no file system, no database, no global state.
All random seeds are fixed for reproducibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean as _mean
from statistics import median as _median
from statistics import stdev as _stdev

from materials_screening.models import MaterialRecord
from materials_screening.sub_agents.outlier_detection.models import (
    ALLOWED_MULTIVARIATE_METHODS,
    ALLOWED_PROPERTIES,
    ALLOWED_UNIVARIATE_METHODS,
    DEFAULT_IQR_THRESHOLD,
    DEFAULT_ZSCORE_THRESHOLD,
    DistributionStats,
    MultivariateOutlierRecord,
    MultivariateOutlierReport,
    PropertyOutlierRecord,
    PropertyOutlierReport,
)

# Fixed seed for deterministic Isolation Forest
_FIXED_SEED = 0
_ISOLATION_FOREST_SCORE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_property_values(
    records: Sequence[MaterialRecord],
    property_name: str,
) -> tuple[list[float], list[MaterialRecord], list[str]]:
    """Extract non-None values for *property_name* from *records*.

    Returns (values, valid_records, warnings).
    """
    values: list[float] = []
    valid: list[MaterialRecord] = []
    warnings: list[str] = []
    for rec in records:
        val = getattr(rec, property_name, None)
        if val is None:
            warnings.append(
                f"{rec.material_id or rec.formula_pretty}: "
                f"{property_name} is missing — excluded from analysis"
            )
        elif isinstance(val, (int, float)) and not math.isfinite(val):
            warnings.append(
                f"{rec.material_id or rec.formula_pretty}: "
                f"{property_name}={val} is non-finite — excluded"
            )
        else:
            values.append(float(val))
            valid.append(rec)
    return values, valid, warnings


def _validate_property(property_name: str) -> None:
    """Raise ValueError if *property_name* is not in the whitelist."""
    if property_name not in ALLOWED_PROPERTIES:
        raise ValueError(
            f"Property {property_name!r} not in allowed set: "
            f"{sorted(ALLOWED_PROPERTIES)}"
        )


def _validate_univariate_method(method: str) -> None:
    if method not in ALLOWED_UNIVARIATE_METHODS:
        raise ValueError(
            f"Method {method!r} not allowed. Choose from: "
            f"{sorted(ALLOWED_UNIVARIATE_METHODS)}"
        )


def _validate_multivariate_method(method: str) -> None:
    if method not in ALLOWED_MULTIVARIATE_METHODS:
        raise ValueError(
            f"Method {method!r} not allowed. Choose from: "
            f"{sorted(ALLOWED_MULTIVARIATE_METHODS)}"
        )


def _material_label(rec: MaterialRecord) -> str:
    """Human-readable formula and identifier for a material."""
    formula = rec.formula_pretty or "unknown"
    return f"{formula} ({rec.material_id})" if rec.material_id else formula


def _sort_key(values: list[float]) -> list[float]:
    return sorted(values)


def _quartiles(sorted_vals: list[float]) -> tuple[float, float, float, float]:
    """Return (q1, median, q3, iqr) using the inclusive/Tukey method."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("cannot compute quartiles from empty data")
    mid = n // 2
    if n % 2 == 0:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid:]
    else:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid + 1 :]
    q1 = float(_median(lower) if lower else sorted_vals[0])
    med = float(_median(sorted_vals))
    q3 = float(_median(upper) if upper else sorted_vals[-1])
    return q1, med, q3, q3 - q1


# ---------------------------------------------------------------------------
# Single-property outlier detection
# ---------------------------------------------------------------------------


def detect_property_outliers(
    records: Sequence[MaterialRecord],
    property_name: str,
    method: str = "zscore",
    threshold: float | None = None,
) -> PropertyOutlierReport:
    """Detect single-property outliers via Z-score or IQR.

    Pure function — no side effects.

    Args:
        records: Material records to analyse.
        property_name: Whitelisted property (band_gap_ev, etc.).
        method: ``"zscore"`` (default threshold 2.0) or ``"iqr"`` (default 1.5).
        threshold: Override the default threshold.

    Returns:
        PropertyOutlierReport with distribution stats and per-record flags.
    """
    _validate_property(property_name)
    _validate_univariate_method(method)

    if method == "zscore":
        threshold = threshold if threshold is not None else DEFAULT_ZSCORE_THRESHOLD
    else:
        threshold = threshold if threshold is not None else DEFAULT_IQR_THRESHOLD

    values, valid_records, warnings = _extract_property_values(records, property_name)

    if len(values) < 10:
        warnings.append(
            f"Small sample size (n={len(values)}): {method} outlier detection "
            "has limited statistical power; a non-outlier result does not rule "
            "out an extreme material."
        )

    if len(values) < 2:
        return PropertyOutlierReport(
            property_name=property_name,
            method=method,
            threshold=threshold,
            distribution=DistributionStats(
                count=len(values),
                mean=float(values[0]) if values else 0.0,
                median=float(values[0]) if values else 0.0,
                std=0.0,
                q1=float(values[0]) if values else 0.0,
                q3=float(values[0]) if values else 0.0,
                iqr=0.0,
                min_value=float(values[0]) if values else 0.0,
                max_value=float(values[0]) if values else 0.0,
            ),
            records=(),
            warnings=tuple(warnings),
        )

    sorted_vals = _sort_key(values)
    mu = _mean(values)
    sigma = _stdev(values) if len(values) > 1 else 0.0
    q1, med, q3, iqr = _quartiles(sorted_vals)

    outlier_records: list[PropertyOutlierRecord] = []
    for rec, val in zip(valid_records, values, strict=True):
        if method == "zscore":
            z = (val - mu) / sigma if sigma > 0 else 0.0
            is_outlier = abs(z) > threshold
            direction = "high" if z > 0 else "low"
        else:  # iqr
            z = None
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            is_outlier = val < lower or val > upper
            direction = "high" if val > upper else "low" if val < lower else "low"

        outlier_records.append(
            PropertyOutlierRecord(
                material_id=rec.material_id,
                material_label=_material_label(rec),
                property=property_name,
                value=float(val),
                z_score=round(z, 6) if z is not None else None,
                is_outlier=is_outlier,
                direction=direction,
            )
        )

    return PropertyOutlierReport(
        property_name=property_name,
        method=method,
        threshold=threshold,
        distribution=DistributionStats(
            count=len(values),
            mean=round(mu, 6),
            median=round(med, 6),
            std=round(sigma, 6),
            q1=round(q1, 6),
            q3=round(q3, 6),
            iqr=round(iqr, 6),
            min_value=round(sorted_vals[0], 6),
            max_value=round(sorted_vals[-1], 6),
        ),
        records=tuple(outlier_records),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Multivariate outlier detection
# ---------------------------------------------------------------------------


def _build_matrix(
    records: Sequence[MaterialRecord],
    properties: Sequence[str],
) -> tuple[list[list[float]], list[MaterialRecord], list[str]]:
    """Build a complete-case matrix (n_samples × m_features).

    Returns (matrix, valid_records, warnings). Rows with any missing value
    are dropped with a warning.
    """
    all_values: list[list[float | None]] = []
    for rec in records:
        row: list[float | None] = []
        for prop in properties:
            val = getattr(rec, prop, None)
            if val is None or not math.isfinite(float(val)):
                row.append(None)
            else:
                row.append(float(val))
        all_values.append(row)

    matrix: list[list[float]] = []
    valid_records: list[MaterialRecord] = []
    warnings: list[str] = []

    for rec, row in zip(records, all_values, strict=True):
        if any(v is None for v in row):
            missing = [p for p, v in zip(properties, row, strict=True) if v is None]
            warnings.append(
                f"{_material_label(rec)}: missing {missing} — excluded"
            )
        else:
            matrix.append([float(v) for v in row])  # type: ignore[arg-type]
            valid_records.append(rec)

    return matrix, valid_records, warnings


def _standardise(matrix: list[list[float]]) -> list[list[float]]:
    """Z-score standardise each column in-place-like, return new matrix."""
    if not matrix or not matrix[0]:
        return matrix
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = [sum(row[c] for row in matrix) / n_rows for c in range(n_cols)]
    stds: list[float] = []
    for c in range(n_cols):
        var = sum((row[c] - means[c]) ** 2 for row in matrix) / n_rows
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return [
        [(row[c] - means[c]) / stds[c] for c in range(n_cols)]
        for row in matrix
    ]


def _covariance(matrix: list[list[float]]) -> list[list[float]]:
    """Compute covariance matrix (m×m) from n×m data matrix."""
    if not matrix or not matrix[0]:
        return []
    n = len(matrix)
    m = len(matrix[0])
    means = [sum(row[c] for row in matrix) / n for c in range(m)]
    cov: list[list[float]] = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            s = sum(
                (row[i] - means[i]) * (row[j] - means[j])
                for row in matrix
            )
            val = s / n  # population covariance
            cov[i][j] = val
            cov[j][i] = val
    return cov


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Invert a small square matrix via Gaussian elimination with partial pivoting.

    Raises ValueError for singular matrices.
    """
    n = len(matrix)
    # Augmented matrix [A | I]
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]

    for col in range(n):
        # Partial pivoting
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-15:
            raise ValueError("covariance matrix is singular — cannot invert")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]


def _mahalanobis(
    matrix: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """Compute Mahalanobis distances for each row.

    Returns (distances, inv_cov).
    """
    if not matrix or not matrix[0]:
        return [], []

    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = [sum(row[c] for row in matrix) / n_rows for c in range(n_cols)]
    cov = _covariance(matrix)
    inv_cov = _invert_matrix(cov)

    distances: list[float] = []
    for row in matrix:
        diff = [row[c] - means[c] for c in range(n_cols)]
        # diff^T @ inv_cov @ diff
        dist_sq = 0.0
        for i in range(n_cols):
            row_sum = 0.0
            for j in range(n_cols):
                row_sum += inv_cov[i][j] * diff[j]
            dist_sq += diff[i] * row_sum
        distances.append(math.sqrt(max(0.0, dist_sq)))

    return distances, inv_cov


def _chi2_critical(m: int, alpha: float = 0.05) -> float:
    """Approximate chi-squared critical value for *m* degrees of freedom.

    Uses the Wilson-Hilferty approximation for small m (1–4).
    """
    if m < 1:
        return 0.0
    # Wilson-Hilferty: χ²_α ≈ m * (1 - 2/(9m) + z_α * √(2/(9m)))^3
    # z_α for α=0.05 (one-sided upper) ≈ 1.64485
    z_alpha = 1.6448536269514722
    term = 1.0 - 2.0 / (9.0 * m) + z_alpha * math.sqrt(2.0 / (9.0 * m))
    return m * (term ** 3)


def _abnormal_properties(
    row: list[float],
    means: list[float],
    inv_cov: list[list[float]],
    properties: Sequence[str],
) -> tuple[tuple[str, ...], str]:
    """Identify which properties contribute most to the Mahalanobis distance."""
    n = len(row)
    diffs = [row[c] - means[c] for c in range(n)]
    # Contribution per property: diff_i * Σ_j inv_cov[i][j] * diff_j
    contributions: list[tuple[str, float]] = []
    for i in range(n):
        contrib = 0.0
        for j in range(n):
            contrib += inv_cov[i][j] * diffs[j]
        contrib = abs(diffs[i] * contrib)
        contributions.append((str(properties[i]), contrib))

    contributions.sort(key=lambda x: -x[1])
    # Take properties with above-average contribution
    avg = sum(c for _, c in contributions) / n if n > 0 else 0
    abnormal = tuple(name for name, c in contributions if c > avg)

    props_display = [f"{n}={diffs[i]:.3f}" for i, n in enumerate(properties)]
    reason = "+".join(props_display)
    return abnormal, reason


def _mahalanobis_outliers(
    matrix: list[list[float]],
    valid_records: list[MaterialRecord],
    properties: Sequence[str],
) -> list[MultivariateOutlierRecord]:
    """Run Mahalanobis detection and return outlier records.

    Falls back to Euclidean distance when the covariance matrix is singular.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0])

    # Standardise
    std_matrix = _standardise(matrix)

    # Mahalanobis on standardised data; fall back to Euclidean if singular
    try:
        distances, inv_cov = _mahalanobis(std_matrix)
    except ValueError:
        # Singular covariance — use Euclidean distance on standardised data
        means_euc = [
            sum(row[c] for row in std_matrix) / n_rows
            for c in range(n_cols)
        ]
        distances = [
            math.sqrt(sum((row[c] - means_euc[c]) ** 2 for c in range(n_cols)))
            for row in std_matrix
        ]
        inv_cov = [
            [1.0 if i == j else 0.0 for j in range(n_cols)]
            for i in range(n_cols)
        ]

    means = [
        sum(row[c] for row in std_matrix) / n_rows for c in range(n_cols)
    ]

    # Chi-squared critical value for outlier threshold (p < 0.05)
    critical = math.sqrt(_chi2_critical(n_cols))

    results: list[MultivariateOutlierRecord] = []
    for i, (rec, row_std) in enumerate(
        zip(valid_records, std_matrix, strict=True)
    ):
        dist = distances[i]
        abnormal, reason = _abnormal_properties(
            row_std, means, inv_cov, properties
        )
        results.append(
            MultivariateOutlierRecord(
                material_id=rec.material_id,
                material_label=_material_label(rec),
                anomaly_score=round(dist, 6),
                is_outlier=dist > critical,
                abnormal_properties=abnormal,
                outlier_reason=reason,
            )
        )

    return results


def _isolation_forest_outliers(
    matrix: list[list[float]],
    valid_records: list[MaterialRecord],
    properties: Sequence[str],
    n_trees: int = 100,
    sample_size: int = 256,
) -> list[MultivariateOutlierRecord]:
    """Deterministic Isolation Forest using a hash-based split sequence.

    Fixed random_seed = 0 guarantees reproducibility.
    """
    n_rows = len(matrix)
    if n_rows < 3:
        # Too few samples — mark all as non-outliers
        return [
            MultivariateOutlierRecord(
                material_id=rec.material_id,
                material_label=_material_label(rec),
                anomaly_score=0.0,
                is_outlier=False,
                abnormal_properties=(),
                outlier_reason="insufficient samples for detection",
            )
            for rec in valid_records
        ]

    n_cols = len(matrix[0])

    path_lengths = [0.0] * n_rows
    subsample_size = min(sample_size, n_rows)
    max_depth = int(math.ceil(math.log2(max(subsample_size, 2))))

    for tree_idx in range(n_trees):
        seed = _FIXED_SEED + tree_idx
        tree_indices = _deterministic_sample(n_rows, subsample_size, seed)
        for row_index, row in enumerate(matrix):
            path_lengths[row_index] += _path_length(
                matrix=matrix,
                indices=tree_indices,
                seed=seed,
                n_cols=n_cols,
                target_row=row,
                depth=0,
                max_depth=max_depth,
            )

    # Average path length
    avg_paths = [pl / n_trees for pl in path_lengths]

    c_n = _expected_path_length(subsample_size)

    # Anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))
    results: list[MultivariateOutlierRecord] = []

    col_means = [sum(row[c] for row in matrix) / n_rows for c in range(n_cols)]
    col_stds = [
        math.sqrt(sum((row[c] - col_means[c]) ** 2 for row in matrix) / n_rows)
        for c in range(n_cols)
    ]
    for i, rec in enumerate(valid_records):
        score = 2.0 ** (-avg_paths[i] / c_n) if c_n > 0 else 0.0
        # Identify abnormal properties: those > 2σ from column means
        row = matrix[i]
        abnormal: list[str] = []
        for c in range(n_cols):
            if col_stds[c] > 1e-12:
                z = (row[c] - col_means[c]) / col_stds[c]
                if abs(z) > 2.0:
                    abnormal.append(str(properties[c]))

        props_display = [
            f"{n}={row[c]:.3f}" for c, n in enumerate(properties)
        ]
        reason = "+".join(props_display)

        results.append(
            MultivariateOutlierRecord(
                material_id=rec.material_id,
                material_label=_material_label(rec),
                anomaly_score=round(score, 6),
                is_outlier=score > _ISOLATION_FOREST_SCORE_THRESHOLD,
                abnormal_properties=tuple(abnormal),
                outlier_reason=reason,
            )
        )

    return results


def _lcg(seed: int) -> int:
    return (seed * 1103515245 + 12345) & 0x7FFFFFFF


def _deterministic_sample(
    population_size: int,
    sample_size: int,
    seed: int,
) -> list[int]:
    """Return a deterministic sample without replacement."""
    indices = list(range(population_size))
    for index in range(sample_size):
        seed = _lcg(seed)
        selected = index + seed % (population_size - index)
        indices[index], indices[selected] = indices[selected], indices[index]
    return indices[:sample_size]


def _expected_path_length(n: int) -> float:
    """Expected unsuccessful BST search length used by Isolation Forest."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _path_length(
    matrix: list[list[float]],
    indices: list[int],
    seed: int,
    n_cols: int,
    target_row: list[float],
    depth: int,
    max_depth: int,
) -> float:
    """Simulate path length for one sample in one tree (deterministic)."""
    if depth >= max_depth or len(indices) <= 1:
        return float(depth) + _expected_path_length(len(indices))

    # Deterministic split: use seed to pick feature and threshold
    def _lcg(s: int) -> int:
        return (s * 1103515245 + 12345) & 0x7FFFFFFF

    s = _lcg(seed + depth * 7777 + len(indices))
    varying = [
        feature
        for feature in range(n_cols)
        if min(matrix[idx][feature] for idx in indices)
        < max(matrix[idx][feature] for idx in indices)
    ]
    if not varying:
        return float(depth) + _expected_path_length(len(indices))
    feature = varying[s % len(varying)]
    lower = min(matrix[idx][feature] for idx in indices)
    upper = max(matrix[idx][feature] for idx in indices)
    s = _lcg(s)
    split = lower + (s / 0x7FFFFFFF) * (upper - lower)

    # Split indices
    left = [idx for idx in indices if matrix[idx][feature] < split]
    right = [idx for idx in indices if matrix[idx][feature] >= split]

    if target_row[feature] < split:
        if not left:
            return float(depth)
        return _path_length(
            matrix, left, seed, n_cols, target_row, depth + 1, max_depth,
        )
    else:
        if not right:
            return float(depth)
        return _path_length(
            matrix, right, seed, n_cols, target_row, depth + 1, max_depth,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_multivariate_outliers(
    records: Sequence[MaterialRecord],
    properties: Sequence[str],
    method: str = "mahalanobis",
) -> MultivariateOutlierReport:
    """Detect multivariate outliers via Mahalanobis distance or Isolation Forest.

    Pure function — no side effects.

    Args:
        records: Material records to analyse.
        properties: 2–4 whitelisted property names.
        method: ``"mahalanobis"`` (default) or ``"isolation_forest"``.

    Returns:
        MultivariateOutlierReport with per-record anomaly scores.
    """
    if len(properties) < 2:
        raise ValueError("multivariate detection requires at least 2 properties")
    if len(properties) > 4:
        raise ValueError("multivariate detection supports at most 4 properties")
    for prop in properties:
        _validate_property(prop)
    _validate_multivariate_method(method)

    matrix, valid_records, warnings = _build_matrix(records, properties)
    threshold = (
        math.sqrt(_chi2_critical(len(properties)))
        if method == "mahalanobis"
        else _ISOLATION_FOREST_SCORE_THRESHOLD
    )

    if len(matrix) < 10:
        warnings.append(
            f"Small sample size (n={len(matrix)}): {method} multivariate "
            "outlier detection has limited statistical power."
        )

    if len(matrix) < 3:
        return MultivariateOutlierReport(
            properties=tuple(properties),
            method=method,
            threshold=round(threshold, 6),
            records=(),
            warnings=tuple(warnings)
            + ("insufficient complete cases for multivariate detection (need ≥3)",),
        )

    if method == "mahalanobis":
        outlier_records = _mahalanobis_outliers(matrix, valid_records, properties)
    else:
        outlier_records = _isolation_forest_outliers(matrix, valid_records, properties)

    return MultivariateOutlierReport(
        properties=tuple(properties),
        method=method,
        threshold=round(threshold, 6),
        records=tuple(outlier_records),
        warnings=tuple(warnings),
    )
