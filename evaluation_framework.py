"""
evaluation_framework.py
------------------------------------------------------------
Author: Akeru Chukwudifu Carl (16805097)
Module: 7005SCN Individual Research Project
"""

import argparse
import logging
import sys
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("evaluation_framework")

CRITERIA_ORDER = ["recall", "accuracy", "precision", "f1", "fpr_inverse"]


UNIFORM_PRIOR = {c: 1.0 / len(CRITERIA_ORDER) for c in CRITERIA_ORDER}


# ---------------------------------------------------------------------------
# STEP 1: DATA VALIDATION
# ---------------------------------------------------------------------------
VALID_CATEGORIES = {
    "Banking/URL Phishing", "Spear Phishing",
    "Business Email Compromise", "Mass/Email Phishing",
}
VALID_EVIDENCE_TYPES = {"primary", "secondary"}


def validate_dataset(df: pd.DataFrame) -> list:
    """Check the loaded dataset for structural problems before any
    scoring happens. Returns a list of human-readable issue strings;
    an empty list means the dataset passed all checks."""
    issues = []

    unknown_categories = set(df["category"].unique()) - VALID_CATEGORIES
    if unknown_categories:
        issues.append(f"Unknown category value(s) found: {unknown_categories}")

    if "evidence_type" in df.columns:
        unknown_evidence = set(df["evidence_type"].dropna().unique()) - VALID_EVIDENCE_TYPES
        if unknown_evidence:
            issues.append(f"Unknown evidence_type value(s) found: {unknown_evidence}")

    for col in ["accuracy", "precision", "recall", "f1"]:
        out_of_range = df[(df[col].notna()) & ((df[col] < 0) | (df[col] > 100))]
        if len(out_of_range):
            issues.append(f"{col}: {len(out_of_range)} row(s) outside the valid 0-100 range")

    negative_fpr = df[(df["fpr"].notna()) & (df["fpr"] < 0)]
    if len(negative_fpr):
        issues.append(f"fpr: {len(negative_fpr)} row(s) with a negative value")

    duplicate_check_cols = ["category", "study", "model", "dataset"]
    if all(c in df.columns for c in duplicate_check_cols):
        dupes = df[df.duplicated(subset=duplicate_check_cols, keep=False)]
        if len(dupes):
            issues.append(f"{len(dupes)} duplicate row(s) found (same category, study, model, dataset)")

    return issues


# ---------------------------------------------------------------------------
# STEP 2: LOADING AND SCORING
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{path}'. Run this script from the directory containing "
            f"phishing_detection_results.csv, or pass --data /path/to/file.csv."
        ) from None
    except pd.errors.EmptyDataError:
        raise ValueError(f"'{path}' is empty or not a valid CSV file.") from None

    required_cols = {"category", "study", "model", "accuracy", "precision", "recall", "f1", "fpr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"'{path}' is missing required column(s): {sorted(missing)}")

    for col in ["accuracy", "precision", "recall", "f1", "fpr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fpr_inverse"] = 100 - df["fpr"]
    if "evidence_type" not in df.columns:
        df["evidence_type"] = "secondary"
    return df


def compute_score_row(row, weights: dict) -> float:
    """Weighted score for one row under an arbitrary weighting scheme,
    re-normalising across whichever metrics are actually present."""
    available = {}
    for key in weights:
        val = row[key]
        if pd.notna(val) and weights[key] > 0:
            available[key] = val
    if not available:
        return np.nan
    weight_sum = sum(weights[k] for k in available)
    score = sum(row[k] * (weights[k] / weight_sum) for k in available)
    return round(score, 2)


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, group in df.groupby("category"):
        row = {"category": cat, "n_studies": len(group),
               "n_primary": (group["evidence_type"] == "primary").sum(),
               "n_secondary": (group["evidence_type"] == "secondary").sum()}
        for metric in ["accuracy", "precision", "recall", "f1", "fpr", "DES"]:
            vals = group[metric].dropna()
            row[f"{metric}_mean"] = round(vals.mean(), 2) if len(vals) else np.nan
            row[f"{metric}_min"] = round(vals.min(), 2) if len(vals) else np.nan
            row[f"{metric}_max"] = round(vals.max(), 2) if len(vals) else np.nan
            row[f"{metric}_n"] = len(vals)
            row[f"{metric}_low_confidence"] = len(vals) < 3
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("category")
    summary = summary.sort_values("DES_mean", ascending=False)
    return summary


# ---------------------------------------------------------------------------
# STEP 3: USE-CASE-PARAMETRISED WEIGHT OPTIMISATION (GENETIC ALGORITHM)
# ---------------------------------------------------------------------------
USE_CASE_PROFILES = {
    "General-Purpose Detection": {
        "cost_fn": 1, "cost_fp": 1,
        "description": ("No specific deployment context assumed, so a missed "
                         "detection and a false alarm are given equal cost (1:1) "
                         "rather than any particular asymmetry. Used as this "
                         "project's default, headline reporting lens wherever a "
                         "single figure is quoted without a named use case."),
    },
    "BEC / Wire-Fraud Interception": {
        "cost_fn": 1000, "cost_fp": 1,
        "description": ("Low volume, extremely high stakes per incident. A missed "
                         "BEC attempt can trigger a large fraudulent wire transfer; "
                         "a false positive only costs an analyst a few minutes of "
                         "review. Recall should dominate the weighting."),
    },
    "Banking/URL Login Protection": {
        "cost_fn": 50, "cost_fp": 20,
        "description": ("Missing a phishing site risks credential theft and account "
                         "takeover, but blocking legitimate banking sessions too "
                         "aggressively causes customer friction and support cost. "
                         "Cost is more balanced between false negatives and false "
                         "positives than in the BEC case."),
    },
    "Mass-Email Gateway Filtering": {
        "cost_fn": 5, "cost_fp": 20,
        "description": ("High volume, low stakes per individual message. Over-"
                         "blocking legitimate mail at scale causes alert fatigue and "
                         "erodes trust in the filter faster than an occasional missed "
                         "phishing email costs the organisation. False positives are "
                         "weighted as more costly than false negatives here."),
    },
}

GA_POP_SIZE = 60
GA_GENERATIONS = 150
GA_MUTATION_STRENGTH = 0.08
GA_MUTATION_RATE = 0.3
GA_ELITE_FRACTION = 0.2
GA_REG_LAMBDA = 0.25          # strength of the tie-breaking pull toward the uniform prior
GA_RANDOM_SEED = 42
GA_MULTISTART_SEEDS = tuple(range(10))  # seeds tried by multistart_optimise_weights()


def expected_cost_row(row, cost_fn: float, cost_fp: float) -> float:
    """Expected operational cost of a result, using recall as the true
    positive rate (so 1-recall is the false negative rate) and fpr
    directly. Returns NaN if either required metric is missing."""
    if pd.isna(row["recall"]) or pd.isna(row["fpr"]):
        return np.nan
    fn_rate = 1 - (row["recall"] / 100)
    fp_rate = row["fpr"] / 100
    return fn_rate * cost_fn + fp_rate * cost_fp


def _rank_array(values: np.ndarray) -> np.ndarray:
    """Simple average-rank implementation (1 = best/highest value),
    used to compute Spearman rank correlation without a scipy dependency."""
    order = values.argsort()[::-1]
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra, rb = _rank_array(a), _rank_array(b)
    ra_c, rb_c = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra_c ** 2).sum() * (rb_c ** 2).sum())
    return float((ra_c * rb_c).sum() / denom) if denom > 0 else 0.0


def _random_simplex_population(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """n random weight vectors of length dim, each non-negative and
    summing to 1 (sampled uniformly over the simplex via the standard
    exponential-normalisation trick)."""
    raw = rng.exponential(1.0, size=(n, dim))
    return raw / raw.sum(axis=1, keepdims=True)


def _prepare_cost_matrix(cost_df: pd.DataFrame) -> np.ndarray:
    """Pre-extracts the cost-eligible rows into a fixed numpy matrix
    (n_rows x 5, columns in CRITERIA_ORDER, NaN preserved) once per GA run,
    so the fitness function can score every candidate in the population
    with pure numpy rather than a pandas .apply() per candidate per
    generation, which is the dominant cost otherwise (population size x
    generations x row count x apply-overhead)."""
    return cost_df[list(CRITERIA_ORDER)].to_numpy(dtype=float)


def fitness_vectorised(population: np.ndarray, value_matrix: np.ndarray,
                        expected_cost: np.ndarray, prior_vec: np.ndarray) -> np.ndarray:
    """Row-level fitness for an entire population at once. value_matrix is
    (n_rows x 5); population is (pop_size x 5). Missing metrics (NaN) are
    excluded per row via re-normalisation, matching compute_score_row's
    behaviour, but done here as batched numpy operations instead of one
    pandas .apply() call per candidate."""
    available = ~np.isnan(value_matrix)  # (n_rows, 5)
    filled = np.nan_to_num(value_matrix, nan=0.0)
    weight_sums = available.astype(float) @ population.T  # (n_rows, pop_size)
    weight_sums = weight_sums.T  # (pop_size, n_rows)
    numer = (available.astype(float) * filled) @ population.T  # (n_rows, pop_size)
    numer = numer.T  # (pop_size, n_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        des = np.where(weight_sums > 0, numer / weight_sums, np.nan)  # (pop_size, n_rows)

    fitness_scores = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        valid = ~np.isnan(des[i]) & ~np.isnan(expected_cost)
        if valid.sum() < 3:
            fitness_scores[i] = -np.inf
            continue
        corr = spearman_corr(des[i][valid], -expected_cost[valid])
        penalty = GA_REG_LAMBDA * np.linalg.norm(population[i] - prior_vec)
        fitness_scores[i] = corr - penalty
    return fitness_scores


def ga_optimise_weights(df: pd.DataFrame, cost_fn: float, cost_fp: float,
                         seed: int = GA_RANDOM_SEED) -> tuple:
    """Runs the Genetic Algorithm once, for one seed, for one use-case
    profile, and returns (best_weights_dict, best_fitness, fitness_history).
    A single run of this function can converge to a local optimum rather
    than the best achievable one (Section 4.6.2 of the report demonstrates
    a concrete case where this happens); `multistart_optimise_weights()`
    below is what this project actually uses for every reported result."""
    rng = np.random.default_rng(seed)
    cost_df = df.dropna(subset=["recall", "fpr"]).copy()
    prior_vec = np.array([UNIFORM_PRIOR[c] for c in CRITERIA_ORDER])
    value_matrix = _prepare_cost_matrix(cost_df)
    expected_cost = cost_df.apply(lambda r: expected_cost_row(r, cost_fn, cost_fp), axis=1).to_numpy(dtype=float)

    population = _random_simplex_population(GA_POP_SIZE, len(CRITERIA_ORDER), rng)
    n_elite = max(2, int(GA_POP_SIZE * GA_ELITE_FRACTION))
    history = []

    for _generation in range(GA_GENERATIONS):
        scores = fitness_vectorised(population, value_matrix, expected_cost, prior_vec)
        order = scores.argsort()[::-1]
        population = population[order]
        scores = scores[order]
        history.append(scores[0])

        elites = population[:n_elite]
        children = [elites[i] for i in range(n_elite)]
        while len(children) < GA_POP_SIZE:
            p1, p2 = elites[rng.integers(n_elite)], elites[rng.integers(n_elite)]
            alpha = rng.uniform(0.3, 0.7)
            child = alpha * p1 + (1 - alpha) * p2
            if rng.random() < GA_MUTATION_RATE:
                child = child + rng.normal(0, GA_MUTATION_STRENGTH, size=child.shape)
                child = np.clip(child, 0, None)
                if child.sum() == 0:
                    child = _random_simplex_population(1, len(CRITERIA_ORDER), rng)[0]
                else:
                    child = child / child.sum()
            children.append(child)
        population = np.array(children)

    final_scores = fitness_vectorised(population, value_matrix, expected_cost, prior_vec)
    best_idx = final_scores.argmax()
    best_weights = {c: float(w) for c, w in zip(CRITERIA_ORDER, population[best_idx])}
    return best_weights, float(final_scores[best_idx]), history


def multistart_optimise_weights(df: pd.DataFrame, cost_fn: float, cost_fp: float,
                                 seeds=GA_MULTISTART_SEEDS) -> tuple:
    """Runs ga_optimise_weights() once per seed and keeps whichever run
    achieves the best fitness. This is the function every reported result
    in this project actually uses. Also returns a `diagnostics` dict
    (per-seed fitness and weights, the winning seed, and per-metric
    standard deviation across seeds) so seed-sensitivity is visible and
    reportable rather than hidden behind a single number."""
    results = []
    for seed in seeds:
        weights, fitness, history = ga_optimise_weights(df, cost_fn, cost_fp, seed=seed)
        results.append({"seed": seed, "weights": weights, "fitness": fitness, "history": history})

    best = max(results, key=lambda r: r["fitness"])
    fitness_values = np.array([r["fitness"] for r in results])
    weight_matrix = np.array([[r["weights"][c] for c in CRITERIA_ORDER] for r in results])

    diagnostics = {
        "n_seeds": len(seeds),
        "fitness_mean": float(fitness_values.mean()),
        "fitness_std": float(fitness_values.std()),
        "fitness_min": float(fitness_values.min()),
        "fitness_max": float(fitness_values.max()),
        "weight_std_by_metric": {c: float(weight_matrix[:, i].std()) for i, c in enumerate(CRITERIA_ORDER)},
        "winning_seed": best["seed"],
        "all_results": results,
    }
    return best["weights"], best["fitness"], best["history"], diagnostics


def grid_search_weights(df: pd.DataFrame, cost_fn: float, cost_fp: float, step: float = 0.05) -> tuple:
    """Exhaustively enumerates every weight vector on the simplex at the
    given step resolution (e.g. step=0.05 means every non-negative
    multiple of 5% that sums to 100%) and scores every one with the exact
    same fitness function the GA uses. This is a brute-force cross-check,
    not a replacement for the GA: at step=0.05 across 5 metrics there are
    10,626 candidate points, which is small enough to fully enumerate in
    well under a second, but a finer step (e.g. 0.01) would already reach
    over 4.5 million points, which is where the GA's search strategy
    starts to pay for itself. Returns (best_weights_dict, best_fitness,
    n_points_evaluated)."""
    n_steps = int(round(1 / step))
    dim = len(CRITERIA_ORDER)

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions(total - i, parts - 1):
                yield (i,) + rest

    cost_df = df.dropna(subset=["recall", "fpr"]).copy()
    value_matrix = _prepare_cost_matrix(cost_df)
    expected_cost = cost_df.apply(lambda r: expected_cost_row(r, cost_fn, cost_fp), axis=1).to_numpy(dtype=float)
    prior_vec = np.array([UNIFORM_PRIOR[c] for c in CRITERIA_ORDER])

    combos = np.array(list(compositions(n_steps, dim)), dtype=float) * step
    scores = fitness_vectorised(combos, value_matrix, expected_cost, prior_vec)
    best_idx = scores.argmax()
    best_weights = {c: float(w) for c, w in zip(CRITERIA_ORDER, combos[best_idx])}
    return best_weights, float(scores[best_idx]), len(combos)


def loocv_weight_stability(df: pd.DataFrame, cost_fn: float, cost_fp: float,
                            seeds=GA_MULTISTART_SEEDS) -> pd.DataFrame:
    """Leave-one-out weight stability check. With as few as 9 cost-eligible
    rows in this dataset, a genuine held-out predictive validation is not
    achievable; what this function checks instead is a more honest
    question given the data size: how much does the optimised weighting
    move when any single cost-eligible row is dropped and the search is
    re-run on the remaining rows? Large swings would indicate the reported
    weights are fragile artefacts of one or two rows rather than a stable
    pattern across the evidence. Returns one row per left-out study."""
    cost_df = df.dropna(subset=["recall", "fpr"]).copy()
    rows = []
    for idx in cost_df.index:
        remaining = df.drop(index=idx)
        weights, fitness, _, _ = multistart_optimise_weights(remaining, cost_fn, cost_fp, seeds=seeds)
        row = {"left_out_study": df.loc[idx, "study"], "left_out_category": df.loc[idx, "category"],
               "fitness": fitness}
        row.update({c: weights[c] for c in CRITERIA_ORDER})
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def cost_ratio_sensitivity(df: pd.DataFrame, profile_name: str,
                            multipliers=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Re-runs the GA for one named use-case profile at several multiples of
    its base cost ratio (default: half, base, and double), to test whether
    the resulting category ranking is sensitive to the exact cost ratio
    chosen. Since the GA takes a cost ratio, rather than a hand-asserted
    weighting, as its only input, this is the natural robustness question:
    does the ranking survive plausible uncertainty in that ratio."""
    profile = USE_CASE_PROFILES[profile_name]
    rows = []
    for m in multipliers:
        cost_fn = profile["cost_fn"] * m
        best_weights, _, _, _ = multistart_optimise_weights(df, cost_fn, profile["cost_fp"])
        scored = df.copy()
        scored["DES"] = scored.apply(lambda r: compute_score_row(r, best_weights), axis=1)
        summary = category_summary(scored)
        row = {"cost_ratio": f"{cost_fn:.0f}:{profile['cost_fp']:.0f}", "multiplier": m}
        for cat in summary.index:
            row[cat] = summary.loc[cat, "DES_mean"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("multiplier")


def run_all_use_cases(df: pd.DataFrame) -> dict:
    """Runs the multi-start GA once per use-case profile and, for each,
    re-runs the existing scoring pipeline (compute_score_row /
    category_summary) with that use case's optimised weights. Returns a
    dict keyed by use-case name."""
    results = {}
    for name, profile in USE_CASE_PROFILES.items():
        best_weights, best_fitness, history, diagnostics = multistart_optimise_weights(
            df, profile["cost_fn"], profile["cost_fp"])
        scored = df.copy()
        scored["DES"] = scored.apply(lambda r: compute_score_row(r, best_weights), axis=1)
        summary = category_summary(scored)
        n_cost_eligible = df.dropna(subset=["recall", "fpr"]).groupby("category").size()
        results[name] = {
            "weights": best_weights,
            "fitness": best_fitness,
            "history": history,
            "diagnostics": diagnostics,
            "summary": summary,
            "n_cost_eligible": n_cost_eligible,
            "cost_fn": profile["cost_fn"],
            "cost_fp": profile["cost_fp"],
        }
    return results


# ---------------------------------------------------------------------------
# STEP 4: CHARTS
# ---------------------------------------------------------------------------
def plot_sample_sizes(df: pd.DataFrame, outpath: str):
    counts = df.groupby("category").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index, counts.values, color="#2E5395")
    ax.set_ylabel("Number of independent results")
    ax.set_title("Evidence Base Size by Phishing Category")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=12, ha="right")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v), ha="center", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_primary_vs_secondary(df: pd.DataFrame, outpath: str):
    both = df[df["category"].isin(["Banking/URL Phishing", "Mass/Email Phishing"])]
    pivot = both.groupby(["category", "evidence_type"])["accuracy"].mean().unstack()
    pivot = pivot[["secondary", "primary"]]
    x = np.arange(len(pivot.index))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, pivot["secondary"], width, label="Secondary (literature)", color="#1F3864")
    ax.bar(x + width / 2, pivot["primary"], width, label="Primary (author-tested)", color="#C0504D")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Primary vs Secondary Evidence: Mean Accuracy\n(categories with both evidence types)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=8, ha="right")
    ax.set_ylim(60, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_ga_convergence(histories: dict, outpath: str):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#1F3864", "#C0504D", "#2E9E6B", "#D9822B"]
    for i, (name, hist) in enumerate(histories.items()):
        ax.plot(range(1, len(hist) + 1), hist, label=name, color=colors[i % len(colors)])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (rank correlation minus uniform-prior penalty)")
    ax.set_title("GA Convergence: Best-of-10-Seeds per Use-Case Profile")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_weights_by_usecase(weights_table: pd.DataFrame, outpath: str):
    criteria = CRITERIA_ORDER
    x = np.arange(len(criteria))
    width = 0.8 / len(weights_table)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#1F3864", "#C0504D", "#2E9E6B", "#D9822B"]
    for i, (usecase, row) in enumerate(weights_table.iterrows()):
        vals = [row[c] * 100 for c in criteria]
        ax.bar(x + (i - (len(weights_table) - 1) / 2) * width, vals, width,
               label=usecase, color=colors[i % len(colors)])
    ax.set_ylabel("GA-Optimised Weight (%)")
    ax.set_title("GA-Optimised DES Weights by Use-Case Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=12, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_category_scores_by_usecase(scores_table: pd.DataFrame, outpath: str):
    categories = scores_table.index.tolist()
    usecases = scores_table.columns.tolist()
    x = np.arange(len(categories))
    width = 0.8 / len(usecases)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#1F3864", "#C0504D", "#2E9E6B", "#D9822B"]
    for i, uc in enumerate(usecases):
        vals = scores_table[uc].fillna(0)
        ax.bar(x + (i - (len(usecases) - 1) / 2) * width, vals, width, label=uc, color=colors[i % len(colors)])
    ax.set_ylabel("DES (%)")
    ax.set_title("Category DES Under Each Use-Case's GA-Optimised Weights")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=12, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_cost_ratio_sensitivity(general_sens: pd.DataFrame, bec_sens: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1F3864", "#C0504D", "#2E9E6B", "#D9822B"]
    for ax, table, title in zip(axes, [general_sens, bec_sens],
                                 ["General-Purpose Detection", "BEC / Wire-Fraud Interception"]):
        categories = [c for c in table.columns if c not in ("cost_ratio",)]
        for i, cat in enumerate(categories):
            ax.plot(table.index, table[cat], marker="o", label=cat, color=colors[i % len(colors)])
        ax.set_xlabel("Cost-ratio multiplier (0.5x - 2x base ratio)")
        ax.set_ylabel("DES (%)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Category Ranking Stability Across Cost-Ratio Uncertainty")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_radar(summary: pd.DataFrame, outpath: str):
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#1F3864", "#C0504D", "#2E9E6B", "#D9822B"]
    for i, cat in enumerate(summary.index):
        vals = [summary.loc[cat, m] if pd.notna(summary.loc[cat, m]) else 0 for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=cat, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.08, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(60, 100)
    ax.set_title("Multi-Metric Profile by Phishing Category", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Category-comparative evaluation framework for AI-based phishing detection results.")
    parser.add_argument("--data", default="phishing_detection_results.csv",
                         help="Path to the results CSV (default: phishing_detection_results.csv)")
    parser.add_argument("--outdir", default=".", help="Directory to write output CSVs and charts to (default: current directory)")
    parser.add_argument("--n-starts", type=int, default=len(GA_MULTISTART_SEEDS),
                         help="Number of random seeds tried per use case by the multi-start GA (default: 10)")
    parser.add_argument("--skip-loocv", action="store_true",
                         help="Skip the leave-one-out weight-stability check (it re-runs the multi-start GA once per "
                              "cost-eligible row, so it is the slowest part of a full run)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug-level logging")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(levelname)s: %(message)s")

    seeds = tuple(range(args.n_starts))
    outdir = args.outdir

    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    issues = validate_dataset(df)
    if issues:
        logger.warning("Data validation issues found:")
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("Data validation passed: no issues found.")

    logger.info("GA regularisation anchor (neutral, equal-weight prior - not a reported scheme):")
    for crit, w in UNIFORM_PRIOR.items():
        logger.info("  %-12s: %.2f%%", crit, w * 100)
    logger.info("This project does not report DES using this prior directly; every reported "
                "DES figure below comes from the multi-start GA, per use case.")

    plot_sample_sizes(df, f"{outdir}/chart_evidence_base_size.png")
    plot_primary_vs_secondary(df, f"{outdir}/chart_primary_vs_secondary.png")

   
    logger.info("Running multi-start GA weight optimisation (%d seeds/use case)...", len(seeds))
    uc_results = run_all_use_cases(df)

    weights_rows, scores_cols, histories, diagnostics_rows = {}, {}, {}, {}
    for name, res in uc_results.items():
        logger.info("--- Use case: %s (cost_FN=%s, cost_FP=%s) ---", name, res["cost_fn"], res["cost_fp"])
        logger.info("  Cost-eligible evidence per category: %s", res["n_cost_eligible"].to_dict())
        diag = res["diagnostics"]
        logger.info("  Winning seed: %s | fitness=%.4f (range across %d seeds: %.4f - %.4f)",
                    diag["winning_seed"], res["fitness"], diag["n_seeds"], diag["fitness_min"], diag["fitness_max"])
        for crit, w in res["weights"].items():
            logger.info("    %-12s: %.2f%%  (std across seeds: %.2f pp)",
                        crit, w * 100, diag["weight_std_by_metric"][crit] * 100)
        weights_rows[name] = res["weights"]
        scores_cols[name] = res["summary"]["DES_mean"]
        histories[name] = res["history"]
        diagnostics_rows[name] = {
            "winning_seed": diag["winning_seed"], "fitness_mean": diag["fitness_mean"],
            "fitness_std": diag["fitness_std"], "fitness_min": diag["fitness_min"], "fitness_max": diag["fitness_max"],
        }

    weights_table = pd.DataFrame(weights_rows).T
    scores_table = pd.DataFrame(scores_cols)

    weights_table.to_csv(f"{outdir}/ga_optimised_weights_by_usecase.csv")
    scores_table.to_csv(f"{outdir}/category_des_by_usecase.csv")
    pd.DataFrame(histories).to_csv(f"{outdir}/ga_fitness_history.csv", index_label="generation")
    pd.DataFrame(diagnostics_rows).T.to_csv(f"{outdir}/ga_seed_robustness.csv")

    plot_ga_convergence(histories, f"{outdir}/chart_ga_convergence.png")
    plot_weights_by_usecase(weights_table, f"{outdir}/chart_ga_weights_by_usecase.png")
    plot_category_scores_by_usecase(scores_table, f"{outdir}/chart_category_scores_by_usecase.png")

    logger.info("GA-optimised weights by use case:\n%s", weights_table.round(4))
    logger.info("Category DES by use case:\n%s", scores_table.round(2))

    default_summary = uc_results["General-Purpose Detection"]["summary"]
    default_summary.to_csv(f"{outdir}/category_summary_default.csv")
    plot_radar(default_summary, f"{outdir}/chart_radar_profile.png")

    
    logger.info("Running grid-search cross-check (5%% resolution, %d candidate points)...", 10626)
    grid_rows = []
    for name, profile in USE_CASE_PROFILES.items():
        grid_weights, grid_fitness, n_points = grid_search_weights(df, profile["cost_fn"], profile["cost_fp"])
        ga_fitness = uc_results[name]["fitness"]
        row = {"use_case": name, "n_grid_points": n_points, "grid_fitness": grid_fitness,
               "ga_fitness": ga_fitness, "gap_grid_minus_ga": grid_fitness - ga_fitness}
        row.update({f"grid_{c}": grid_weights[c] for c in CRITERIA_ORDER})
        grid_rows.append(row)
        logger.info("  %s: GA fitness=%.4f, grid fitness=%.4f, gap=%.4f",
                    name, ga_fitness, grid_fitness, grid_fitness - ga_fitness)
    pd.DataFrame(grid_rows).set_index("use_case").to_csv(f"{outdir}/grid_search_crosscheck.csv")

    
    if not args.skip_loocv:
        logger.info("Running leave-one-out weight-stability check (this re-runs the multi-start "
                     "GA once per cost-eligible row and is the slowest step)...")
        loocv_rows = {}
        for name, profile in USE_CASE_PROFILES.items():
            loocv_result = loocv_weight_stability(df, profile["cost_fn"], profile["cost_fp"], seeds=seeds)
            loocv_rows[name] = loocv_result
            logger.info("  %s: weight std across %d leave-one-out refits: %s",
                        name, len(loocv_result),
                        {c: round(loocv_result[c].std(), 3) for c in CRITERIA_ORDER})
        pd.concat(loocv_rows, names=["use_case", "row"]).to_csv(f"{outdir}/loocv_weight_stability.csv")
    else:
        logger.info("Skipping leave-one-out weight-stability check (--skip-loocv).")

    
    logger.info("Running cost-ratio sensitivity analysis...")
    all_sens = {}
    for name in USE_CASE_PROFILES:
        sens_table = cost_ratio_sensitivity(df, name)
        all_sens[name] = sens_table
        logger.info("--- %s ---\n%s", name, sens_table.round(2))

    combined_sens = pd.concat(all_sens, names=["use_case", "multiplier"])
    combined_sens.to_csv(f"{outdir}/cost_ratio_sensitivity.csv")
    plot_cost_ratio_sensitivity(all_sens["General-Purpose Detection"],
                                 all_sens["BEC / Wire-Fraud Interception"],
                                 f"{outdir}/chart_cost_ratio_sensitivity.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
