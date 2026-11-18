import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter

# A: Univariate (column-by-column)
# ----------------------------------

def kolmogorov_smirnov_test(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 30_000,
    random_state: int = 42,
) -> dict:
    """Calculate the two-sample Kolmogorov-Smirnov (KS) test column-wise

    between two dataframes.

    Returns a dictionary mapping each column to its KS statistic (0.0 to 1.0),
    which represents the maximum vertical distance between the empirical CDFs.
    """
    results = {}
    rng = np.random.default_rng(random_state)
    
    for col in columns:
        if col not in df_p.columns or col not in df_q.columns:
            continue

        p_vals = df_p[col].dropna().to_numpy()
        q_vals = df_q[col].dropna().to_numpy()

        if len(p_vals) == 0 or len(q_vals) == 0:
            continue

        if len(p_vals) > max_samples:
            p_vals = rng.choice(p_vals, max_samples, replace=False)
        if len(q_vals) > max_samples:
            q_vals = rng.choice(q_vals, max_samples, replace=False)

        # ks_2samp returns a result object containing .statistic and .pvalue
        ks_result = ks_2samp(p_vals, q_vals)

        # Store the KS statistic (acting as a normalized distance metric)
        results[col] = float(ks_result.statistic)

    return results

def wasserstein(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 15_000,
    random_state: int = 42,
) -> dict:
    """
    Calculate the Wasserstein distance column-wise for the input dataframes.
    Returns a number between 0 (closest distance) and +inf per column, which scales with and has the unit of the input values.
    """
    results = {}
    rng = np.random.default_rng(random_state)
    
    for col in columns:
        if col not in df_p.columns or col not in df_q.columns:
            continue
            
        p_vals = df_p[col].dropna().to_numpy()
        q_vals = df_q[col].dropna().to_numpy()

        if len(p_vals) == 0 or len(q_vals) == 0:
            continue

        if len(p_vals) > max_samples:
            p_vals = rng.choice(p_vals, max_samples, replace=False)
        if len(q_vals) > max_samples:
            q_vals = rng.choice(q_vals, max_samples, replace=False)

        # Works directly on raw data (no binning needed)
        results[col] = float(wasserstein_distance(p_vals, q_vals))
    return results

def population_stability_index(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    num_bins: int = 50,
    eps: float = 1e-10,
) -> dict:
    """Calculate the Population Stability Index (PSI) column-wise

    between two dataframes (baseline df_p vs. target df_q).

    Returns a dictionary mapping each column to its PSI value.
    Standard rule-of-thumb thresholds:
- PSI < 0.10: No significant distribution change
- 0.10 <= PSI < 0.25: Moderate shift
- PSI >= 0.25: Significant shift / instability
    """
    results = {}
    for col in columns:
        if col not in df_p.columns or col not in df_q.columns:
            continue

        p_vals = df_p[col].dropna().to_numpy()
        q_vals = df_q[col].dropna().to_numpy()

        if len(p_vals) == 0 or len(q_vals) == 0:
            continue

        # Define bin edges based on the baseline dataframe (df_p) using quantiles
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(p_vals, quantiles)

        # Handle edge cases where duplicate bin edges occur (e.g., sparse features)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            results[col] = 0.0
            continue

        # Widen boundaries slightly to capture any out-of-bounds values in df_q
        bin_edges[0] = min(bin_edges[0], np.min(q_vals)) - 1e-5
        bin_edges[-1] = max(bin_edges[-1], np.max(q_vals)) + 1e-5

        # Compute binned counts
        p_counts, _ = np.histogram(p_vals, bins=bin_edges)
        q_counts, _ = np.histogram(q_vals, bins=bin_edges)

        # Convert to proportions (probabilities)
        p_props = p_counts / len(p_vals)
        q_props = q_counts / len(q_vals)

        # Apply smoothing epsilon to avoid division by zero or log(0)
        p_props = np.where(p_props == 0, eps, p_props)
        q_props = np.where(q_props == 0, eps, q_props)

        # Re-normalize proportions after smoothing
        p_props /= np.sum(p_props)
        q_props /= np.sum(q_props)

        # Calculate PSI: sum((Actual - Expected) * ln(Actual / Expected))
        # Expected = p_props, Actual = q_props
        psi_value = np.sum((q_props - p_props) * np.log(q_props / p_props))
        results[col] = float(psi_value)

    return results

def _kl_div_or_js_dist(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins: int = 50,
    epsilon: float = 1e-10,
    metric: str = "",
) -> dict:
    """Calculates column-wise divergence or distance between two pandas DataFrames P and Q.
    Using default natural logarithm (base = e).

    Parameters
    -----
- df_p: Reference DataFrame (P)
- df_q: Comparison DataFrame (Q)
- columns: which columns to evaluate
- bins: Number of bins for continuous numerical columns
- epsilon: Smoothing factor to prevent zero-probability errors
- metric: "kl" for asymmetric Kullback-Leibler Divergence (P || Q), "js" for symmetric Jensen-Shannon Distance

    Returns
    -----
- dict of divergence/distance values for demanded columns.
    """
    results = {}

    for col in columns:
        if col not in df_p.columns or col not in df_q.columns:
            continue

        p_data = df_p[col].dropna()
        q_data = df_q[col].dropna()

        if len(p_data) == 0 or len(q_data) == 0:
            continue

        min_val = min(p_data.min(), q_data.min())
        max_val = max(p_data.max(), q_data.max())

        # If all values are identical, distance/divergence is 0
        if min_val == max_val:
            results[col] = 0.0
            continue

        bin_edges = np.linspace(min_val, max_val, bins + 1)
        p_counts, _ = np.histogram(p_data, bins=bin_edges)
        q_counts, _ = np.histogram(q_data, bins=bin_edges)

        # Convert counts to probability distributions and apply smoothing
        p_prob = p_counts.astype(float) + epsilon
        q_prob = q_counts.astype(float) + epsilon

        p_prob /= p_prob.sum()
        q_prob /= q_prob.sum()

        # Calculate based on selected metric
        if metric == "kl":
            # Compute KL Divergence: D_KL(P || Q) (Asymmetric, unbounded)
            results[col] = float(entropy(p_prob, q_prob))
        elif metric == "js":
            # Compute Jensen-Shannon Distance (Symmetric, bounded between 0 and 1)
            results[col] = float(jensenshannon(p_prob, q_prob))
        else:
            raise ValueError("Unsupported metric. Choose 'kl' or 'js'.")

    return results

def kl_divergence(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins: int = 50,
    epsilon: float = 1e-10,
) -> dict:
    """Calculates the asymmetric Kullback-Leibler Divergence (P || Q) column-wise between two pandas DataFrames P and Q.
    Returns a Value between 0 (least divergence) and +inf, per column.
    
    Parameters
    -----
- df_p: Reference DataFrame (P)
- df_q: Comparison DataFrame (Q)
- columns: which columns to evaluate
- bins: Number of bins for continuous numerical columns
- epsilon: Smoothing factor to prevent zero-probability errors

    Returns
    -----
- dict of divergence values for demanded columns.
    """
    return _kl_div_or_js_dist(df_p=df_p, df_q=df_q, columns=columns, bins=bins, epsilon=epsilon, metric="kl")

def js_distance(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins: int = 50,
    epsilon: float = 1e-10,
) -> dict:
    """Calculates the symmetric Jensen-Shannon Distance column-wise between two pandas DataFrames P and Q.
    Returns a value between 0 (closest distance) and 0.832 = sqrt(ln 2), per column. 

    Parameters
    -----
- df_p: Reference DataFrame (P)
- df_q: Comparison DataFrame (Q)
- columns: which columns to evaluate
- bins: Number of bins for continuous numerical columns
- epsilon: Smoothing factor to prevent zero-probability errors

    Returns
    -----
- dict of distance values for demanded columns.
    """

    return _kl_div_or_js_dist(df_p=df_p, df_q=df_q, columns=columns, bins=bins, epsilon=epsilon, metric="js")


# B: Multivariate (columns combined)
# ----------------------------------

def energy_distance(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 10_000,
    random_state: int = 42,
) -> float:
    """Calculate the multivariate Energy Distance between two dataframes.
    
    Using Euclidean distance matrices, it measures the statistical distance 
    between two multi-dimensional probability distributions.
    
    If datasets exceed max_samples, they are randomly subsampled to keep 
    O(N^2) pairwise distance calculations efficient.
    
    Returns an energy distance value >= 0.0 (0 means identical distributions).
    """
    valid_cols = [c for c in columns if c in df_p.columns and c in df_q.columns]
    if not valid_cols:
        raise ValueError("No matching columns found in both DataFrames.")
        
    # MEMORY FIX: Subsample the DataFrame directly BEFORE converting to numpy or allocating
    df_p_sub = df_p[valid_cols]
    if len(df_p_sub) > max_samples:
        df_p_sub = df_p_sub.sample(n=max_samples, random_state=random_state)
        
    df_q_sub = df_q[valid_cols]
    if len(df_q_sub) > max_samples:
        df_q_sub = df_q_sub.sample(n=max_samples, random_state=random_state)
        
    X = df_p_sub.to_numpy(dtype=np.float64)
    Y = df_q_sub.to_numpy(dtype=np.float64)
        
    n = len(X)
    m = len(Y)
    if n == 0 or m == 0:
        raise ValueError("One or both DataFrames are empty for selected columns.")
        
    # Compute pairwise Euclidean distance matrices using vectorization
    XX_sq = np.sum(X**2, axis=1, keepdims=True)
    YY_sq = np.sum(Y**2, axis=1, keepdims=True)
    
    dist_XY = np.sqrt(np.clip(XX_sq + YY_sq.T - 2 * np.dot(X, Y.T), 0, None))
    dist_XX = np.sqrt(np.clip(XX_sq + XX_sq.T - 2 * np.dot(X, X.T), 0, None))
    dist_YY = np.sqrt(np.clip(YY_sq + YY_sq.T - 2 * np.dot(Y, Y.T), 0, None))
    
    term_xy = np.sum(dist_XY) / (n * m)
    term_xx = np.sum(dist_XX) / (n * n)
    term_yy = np.sum(dist_YY) / (m * m)
    
    energy_dist_sq = 2.0 * term_xy - term_xx - term_yy
    
    return float(np.sqrt(np.maximum(0.0, energy_dist_sq)))

def maximum_mean_discrepancy(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 10_000,
    random_state: int = 42,
) -> float:
    """Calculate the multivariate Maximum Mean Discrepancy (MMD)

    between two dataframes using an RBF kernel and the median heuristic.

    If datasets exceed max_samples, they are randomly subsampled to keep
    O(N^2) pairwise distance matrix computations efficient.

    Returns a squared MMD value >= 0.0 (0 means identical distributions).
    """
    valid_cols = [c for c in columns if c in df_p.columns and c in df_q.columns]
    if not valid_cols:
        raise ValueError("No matching columns found in both DataFrames.")

    # MEMORY FIX: Subsample the DataFrame directly BEFORE converting to numpy or allocating
    df_p_sub = df_p[valid_cols]
    if len(df_p_sub) > max_samples:
        df_p_sub = df_p_sub.sample(n=max_samples, random_state=random_state)
        
    df_q_sub = df_q[valid_cols]
    if len(df_q_sub) > max_samples:
        df_q_sub = df_q_sub.sample(n=max_samples, random_state=random_state)

    X = df_p_sub.to_numpy(dtype=np.float64)
    Y = df_q_sub.to_numpy(dtype=np.float64)

    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        raise ValueError("One or both DataFrames are empty for the selected columns.")

    # Combine data to compute the RBF kernel bandwidth via median heuristic
    XY_pool = np.vstack([X, Y])
    XX_pool = np.sum(XY_pool**2, axis=1, keepdims=True)
    dist_sq_pool = XX_pool + XX_pool.T - 2 * np.dot(XY_pool, XY_pool.T)
    dist_sq_pool = np.clip(dist_sq_pool, 0, None)

    median_dist = np.median(dist_sq_pool[dist_sq_pool > 0])
    gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0

    def compute_rbf_kernel(A, B, gamma):
        AA = np.sum(A**2, axis=1, keepdims=True)
        BB = np.sum(B**2, axis=1, keepdims=True)
        AB = np.dot(A, B.T)
        dist_sq = AA + BB.T - 2 * AB
        return np.exp(-gamma * np.clip(dist_sq, 0, None))

    K_XX = compute_rbf_kernel(X, X, gamma)
    K_YY = compute_rbf_kernel(Y, Y, gamma)
    K_XY = compute_rbf_kernel(X, Y, gamma)

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    term1 = np.sum(K_XX) / (n * (n - 1)) if n > 1 else 0.0
    term2 = np.sum(K_YY) / (m * (m - 1)) if m > 1 else 0.0
    term3 = 2 * np.sum(K_XY) / (n * m)

    mmd_sq = term1 + term2 - term3

    return float(np.maximum(0.0, mmd_sq))

def c2st(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 20_000,
    random_state: int = 42,
) -> float:
    """Make a multivariate Classifier-Two-Sample-Test (C2ST) across all specified columns.
    
    Returns a single number between 0.5 (minimally distinct data) and 1.0 (maximally distinct data).
    """
    cols = [c for c in columns if c in df_p.columns and c in df_q.columns]
    if not cols:
        raise ValueError("No matching columns found in both DataFrames.")
        
    p_sample_size = min(len(df_p), max_samples)
    q_sample_size = min(len(df_q), max_samples)
    
    # MEMORY FIX: Sample the dataframe subset directly instead of running dropna on 2M rows first
    p_subset = df_p[cols].sample(n=p_sample_size, random_state=random_state)
    q_subset = df_q[cols].sample(n=q_sample_size, random_state=random_state)
    
    X = pd.concat([p_subset, q_subset], axis=0)
    y = np.array([0] * len(p_subset) + [1] * len(q_subset))
    
    clf = RandomForestClassifier(n_estimators=20, max_depth=8, n_jobs=-1, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    return float(scores.mean())

# C: Combinations
# ---------------

def evaluate_univariate_metrics(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins: int = 10,
    eps: float = 1e-10,
    verbose: bool = True,
) -> dict:
    """Run all univariate column-wise metrics safely with error handling."""
    if verbose:
        print("Running univariate metrics (KS, Wasserstein, PSI, KL, JS)...")
    
    results = {}
    
    metrics_to_run = {
        "kolmogorov_smirnov": lambda: kolmogorov_smirnov_test(df_p, df_q, columns),
        "wasserstein": lambda: wasserstein(df_p, df_q, columns),
        "psi": lambda: population_stability_index(df_p, df_q, columns, num_bins=bins, eps=eps),
        "kl_divergence": lambda: kl_divergence(df_p, df_q, columns, bins=bins, epsilon=eps),
        "js_distance": lambda: js_distance(df_p, df_q, columns, bins=bins, epsilon=eps),
    }

    for name, func in metrics_to_run.items():
        try:
            results[name] = func()
        except Exception as e:
            # Catch exceptions gracefully so other metrics still run
            results[name] = {"error": str(e)}
    
    if verbose:
        print("Univariate metrics complete.")

    return results

def evaluate_multivariate_metrics(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    max_samples: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Run all multivariate joint distribution metrics safely with error handling."""
    if verbose:
        print("Calculating multivariate metrics (Energy distance, MMD, C2ST)...")

    results = {}
    
    metrics_to_run = {
        "energy_distance": lambda: energy_distance(df_p, df_q, columns, max_samples=max_samples, random_state=random_state),
        "mmd": lambda: maximum_mean_discrepancy(df_p, df_q, columns, max_samples=max_samples, random_state=random_state),
        "c2st": lambda: c2st(df_p, df_q, columns, max_samples=max_samples, random_state=random_state),
    }

    for name, func in metrics_to_run.items():
        try:
            results[name] = func()
        except Exception as e:
            results[name] = {"error": str(e)}

    if verbose:
        print("Multivariate metrics complete.")

    return results

def evaluate_all_metrics(
    df_p: pd.DataFrame,
    df_q: pd.DataFrame,
    columns: set = {"ekin", "x", "y", "z", "ux", "uy", "uz"},
    bins: int = 10,
    eps: float = 1e-10,
    max_samples: int = 50000,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Master wrapper that safely evaluates both univariate and multivariate suites."""
    if verbose:
        print("Starting evaluation of all univariate and multivariate metrics...")
    
    return {
        "univariate": evaluate_univariate_metrics(df_p, df_q, columns=columns, bins=bins, eps=eps, verbose=verbose),
        "multivariate": evaluate_multivariate_metrics(df_p, df_q, columns=columns, max_samples=max_samples, random_state=random_state, verbose=verbose),
    }

# OTHER
# -----

def smooth_elbow(x_array, y_array, plot=False):
    """
    Calculate x position of elbow of a falling curve y(x). That is where it has its maximum bend.
    It first smoothes the curve.
    """
    
    # Smooth the data
    smoothed_y = savgol_filter(y_array, window_length=5, polyorder=2)

    # Normalize coordinates to find the elbow mathematically
    x_norm = (x_array - x_array[0]) / (x_array[-1] - x_array[0])
    y_norm = (smoothed_y - smoothed_y[0]) / (smoothed_y[-1] - smoothed_y[0])

    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)

    vecs = np.column_stack((x_norm - x_norm[0], y_norm - y_norm[0]))
    scalar_projections = np.dot(vecs, line_vec)
    projections = np.outer(scalar_projections, line_vec)
    perpendicular_vecs = vecs - projections
    distances = np.linalg.norm(perpendicular_vecs, axis=1)

    elbow_idx = np.argmax(distances)
    optimal_x = x_array[elbow_idx]
    optimal_y = smoothed_y[elbow_idx]

    if plot:
        # Calculate the projection point on the baseline in original coordinates (for drawing)
        P0 = np.array([x_array[0], smoothed_y[0]])          # Start point
        P1 = np.array([x_array[-1], smoothed_y[-1]])        # End point
        P_elbow = np.array([x_array[elbow_idx], smoothed_y[elbow_idx]]) # Elbow point

        chord_vec = P1 - P0
        chord_unit = chord_vec / np.linalg.norm(chord_vec)
        elbow_vec = P_elbow - P0
        proj_length = np.dot(elbow_vec, chord_unit)
        P_proj = P0 + proj_length * chord_unit  # Closest point on the baseline to the elbow

        # Plotting everything
        plt.figure(figsize=(10, 6))

        # Plot raw and smoothed curves
        plt.plot(x_array, y_array, 'o', color='gray', alpha=0.5, label='Raw Y (Noisy)')
        plt.plot(x_array, smoothed_y, 'b-', linewidth=2, label='Smoothed Trendline')

        # Draw the baseline straight line (chord) from start to end
        plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'k--', label='Baseline (Chord Line)', alpha=0.7)

        # Draw the perpendicular distance line from the elbow to the baseline
        plt.plot([P_elbow[0], P_proj[0]], [P_elbow[1], P_proj[1]], 'r-', linewidth=2.5, label='Max Distance (Elbow Gap)')

        # Highlight the exact elbow point
        plt.plot(optimal_x, optimal_y, 'ro', markersize=9, label=f'Optimal X: {optimal_x}')

        # Labels and formatting
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Visualizing the Elbow Method & Maximum Perpendicular Distance")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    return optimal_x

# %%
# Example data with bumps/noise
# n_fit = np.array([100, 300, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000])
# metric = np.array([5.2, 4.8, 3.1, 2.9, 1.6, 1.9, 0.9, 1.1, 0.6, 0.58, 0.57])
# print(smooth_elbow(x_array=n_fit, y_array=metric, plot=True))
