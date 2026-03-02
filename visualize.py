"""
TED-LW Visualization Module (v2 — Interference Metrics)
========================================================
Generates four publication-quality plots from pipeline results:

  Graph A: Capability Frontier (LW vs Accuracy, logistic fit)
  Graph B: Interference Time Series (success vs failure comparison)
  Graph C: Choke Point (IG + Rolling Wander Ratio dual-axis)
  Graph D: Interference Cliff (Peak_IG vs Accuracy — the smoking gun)

Usage:
    python visualize.py [--results-dir ./results]
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import curve_fit

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize")

# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
})

MODEL_COLORS = {
    "Qwen3.5-35B-A3B": "#58a6ff",
    "Nanbeige4.1-3B": "#f78166",
    "Nemotron-3-Nano-30B-A3B": "#7ee787",
    "LFM2-24B-A2B": "#d2a8ff",
}

FALLBACK_COLORS = ["#79c0ff", "#ffa657", "#56d364", "#bc8cff"]


# ---------------------------------------------------------------------------
# Graph A: Capability Frontier
# ---------------------------------------------------------------------------

def plot_capability_frontier(df: pd.DataFrame, output_path: str):
    """Total Latent Work (log) vs Accuracy with logistic fit per model."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, model in enumerate(sorted(df["Model_Name"].unique())):
        color = MODEL_COLORS.get(model, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        mdf = df[df["Model_Name"] == model].copy()
        mdf["Total_Latent_Work"] = pd.to_numeric(mdf["Total_Latent_Work"], errors="coerce")
        mdf = mdf.dropna(subset=["Total_Latent_Work"])
        if mdf.empty:
            continue

        x = mdf["Total_Latent_Work"].values
        y = mdf["Is_Correct"].astype(int).values
        y_jitter = y + np.random.uniform(-0.03, 0.03, size=len(y))
        ax.scatter(x, y_jitter, c=color, alpha=0.3, s=20, edgecolors="none")

        if len(np.unique(y)) > 1 and len(x) >= 10:
            try:
                x_log = np.log10(np.maximum(x, 1e-6))
                def logistic(t, a, b): return expit(a * t + b)
                popt, _ = curve_fit(logistic, x_log, y.astype(float), p0=[-1.0, 5.0], maxfev=5000)
                x_smooth = np.linspace(x_log.min(), x_log.max(), 200)
                ax.plot(10 ** x_smooth, logistic(x_smooth, *popt), color=color, linewidth=2.5, label=model)
            except (RuntimeError, ValueError):
                ax.plot([], [], color=color, linewidth=2.5, label=model)
        else:
            ax.axhline(y=y.mean(), color=color, linewidth=1.5, linestyle="--", label=f"{model} ({y.mean():.0%})")

    ax.set_xscale("log")
    ax.set_xlabel("Total Latent Work (ICV × (1+IG))", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title("Capability Frontier: Accuracy vs Problem Difficulty", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(-0.08, 1.08)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graph A saved: {output_path}")


# ---------------------------------------------------------------------------
# Graph B: Interference Time Series (success vs failure)
# ---------------------------------------------------------------------------

def plot_interference_comparison(
    traces_dir: str, df: pd.DataFrame, output_path: str,
    target_model: str = "Nanbeige4.1-3B",
):
    """Side-by-side IG time series for a success vs failure from the same model."""
    model_df = df[df["Model_Name"] == target_model]
    if model_df.empty:
        target_model = df["Model_Name"].iloc[0]
        model_df = df[df["Model_Name"] == target_model]

    successes = model_df[model_df["Is_Correct"] == True]
    failures = model_df[model_df["Is_Correct"] == False]

    success_trace = _load_best_trace(traces_dir, target_model, successes)
    failure_trace = _load_best_trace(traces_dir, target_model, failures)

    if success_trace is None and failure_trace is None:
        logger.warning("No traces for interference comparison. Skipping Graph B.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    titles = ["Successful Solve", "Failed Solve"]
    traces = [success_trace, failure_trace]

    for ax, trace, title in zip(axes, traces, titles):
        if trace is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14, color="#8b949e")
            ax.set_title(f"{title} — {target_model}", fontsize=12)
            continue

        ig = trace.get("ig_per_token", np.array([]))
        csi = trace.get("csi_per_token", np.array([]))
        n = len(ig)
        if n < 5:
            ax.text(0.5, 0.5, "Too few tokens", ha="center", va="center", fontsize=14, color="#8b949e")
            ax.set_title(f"{title}", fontsize=12)
            continue

        steps = np.arange(n)

        # Rolling average of IG
        window = min(config.IG_SMOOTHING_WINDOW, n)
        ig_smooth = np.convolve(ig, np.ones(window) / window, mode="same")

        ax.plot(steps, ig, alpha=0.15, color="#58a6ff", linewidth=0.5)
        ax.plot(steps, ig_smooth, color="#58a6ff", linewidth=2, label="IG (smoothed)")
        if len(csi) == n:
            csi_smooth = np.convolve(csi, np.ones(window) / window, mode="same")
            ax.plot(steps, csi_smooth, color="#f78166", linewidth=2, linestyle="--", label="CSI (smoothed)")

        ax.set_xlabel("Token Step", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"{title} — {target_model}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Interference Gauge: Success vs Failure", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graph B saved: {output_path}")


def _load_best_trace(traces_dir: str, model_name: str, subset_df: pd.DataFrame) -> dict:
    """Load the trace .npz with the most tokens from a DataFrame subset."""
    if subset_df.empty:
        return None
    sorted_df = subset_df.sort_values("Total_Tokens_Generated", ascending=False)
    for _, row in sorted_df.iterrows():
        pid = int(row["Problem_ID"])
        path = os.path.join(traces_dir, f"{model_name}_problem_{pid:04d}.npz")
        if os.path.exists(path):
            try:
                data = np.load(path)
                if data["ig_per_token"].size > 0:
                    return dict(data)
            except Exception as e:
                logger.warning(f"Failed to load trace {path}: {e}")
    return None


# ---------------------------------------------------------------------------
# Graph C: Choke Point (IG + Outlier Pressure dual-axis)
# ---------------------------------------------------------------------------

def plot_choke_point(
    traces_dir: str, df: pd.DataFrame, output_path: str,
    target_model: str = "LFM2-24B-A2B",
):
    """IG and Outlier Pressure (L_inf) for a failed high-difficulty run."""
    model_df = df[df["Model_Name"] == target_model]
    if model_df.empty:
        target_model = df["Model_Name"].iloc[0]
        model_df = df[df["Model_Name"] == target_model]

    failures = model_df[model_df["Is_Correct"] == False].copy()
    failures["Total_Latent_Work"] = pd.to_numeric(failures["Total_Latent_Work"], errors="coerce")
    if failures.empty:
        mdf = model_df.copy()
        mdf["Total_Latent_Work"] = pd.to_numeric(mdf["Total_Latent_Work"], errors="coerce")
        failures = mdf.nlargest(5, "Total_Latent_Work")

    trace = _load_best_trace(traces_dir, target_model, failures)
    if trace is None:
        logger.warning("No traces for choke point. Skipping Graph C.")
        return

    ig = trace.get("ig_per_token", np.array([]))
    l_inf = trace.get("l_inf_per_token", np.array([]))
    n = len(ig)
    if n < 20 or len(l_inf) != n:
        logger.warning("Too few tokens or missing L_inf data for choke point. Skipping.")
        return

    steps = np.arange(n)
    window = min(config.IG_SMOOTHING_WINDOW, n)
    ig_smooth = np.convolve(ig, np.ones(window) / window, mode="same")
    l_inf_smooth = np.convolve(l_inf, np.ones(window) / window, mode="same")

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # IG (blue, left axis)
    color_blue = "#58a6ff"
    ax1.plot(steps, ig_smooth, color=color_blue, linewidth=2, alpha=0.9)
    ax1.set_xlabel("Token Generation Step", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Interference Gauge (IG, smoothed)", color=color_blue, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_blue)
    ax1.fill_between(steps, ig_smooth, alpha=0.08, color=color_blue)

    # L_inf Outlier Pressure (red, right axis)
    ax2 = ax1.twinx()
    color_red = "#f78166"
    ax2.plot(steps, l_inf_smooth, color=color_red, linewidth=1.5, alpha=0.9)
    ax2.set_ylabel("Outlier Pressure (L∞ Norm, smoothed)", color=color_red, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_red)

    # Find choke point
    if n > window * 2:
        ig_late = ig_smooth[n // 2:]
        if len(ig_late) > 10:
            peak_idx = n // 2 + np.argmax(ig_late)
            if ig_smooth[peak_idx] > 0.5:
                ax1.axvline(x=peak_idx, color="#ffd700", linewidth=1.5, linestyle="--", alpha=0.7)
                ax1.annotate(
                    "Interference Spike", xy=(peak_idx, ig_smooth[peak_idx]),
                    xytext=(peak_idx + 50, ig_smooth.max() * 0.95),
                    fontsize=10, color="#ffd700", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ffd700"),
                )

    ax1.set_title(f"Choke Point Analysis: Interference vs Outlier Pressure — {target_model}", fontsize=16, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graph C saved: {output_path}")


# ---------------------------------------------------------------------------
# Graph D: Interference Cliff (the smoking gun)
# ---------------------------------------------------------------------------

def plot_interference_cliff(df: pd.DataFrame, output_path: str) -> dict:
    """
    Peak_IG vs Accuracy — universal phase transition plot.
    Returns dict of {model_name: tau} where tau is the interference ceiling
    (Peak_IG at 50% accuracy from logistic fit).
    """
    tau_per_model = {}

    if "Peak_IG" not in df.columns:
        logger.warning("Peak_IG column not found. Skipping Graph D.")
        return tau_per_model

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, model in enumerate(sorted(df["Model_Name"].unique())):
        color = MODEL_COLORS.get(model, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        mdf = df[df["Model_Name"] == model].copy()
        mdf["Peak_IG"] = pd.to_numeric(mdf["Peak_IG"], errors="coerce")
        mdf = mdf.dropna(subset=["Peak_IG"])
        if mdf.empty:
            continue

        x = mdf["Peak_IG"].values
        y = mdf["Is_Correct"].astype(int).values
        y_jitter = y + np.random.uniform(-0.03, 0.03, size=len(y))
        ax.scatter(x, y_jitter, c=color, alpha=0.4, s=30, edgecolors="none", label=model)

        if len(np.unique(y)) > 1 and len(x) >= 10:
            try:
                def logistic(t, a, b): return expit(a * t + b)
                popt, _ = curve_fit(logistic, x, y.astype(float), p0=[-10.0, 5.0], maxfev=5000)
                x_smooth = np.linspace(0, 1, 200)
                ax.plot(x_smooth, logistic(x_smooth, *popt), color=color, linewidth=2.5, alpha=0.9)

                # τ = interference ceiling (50% accuracy point)
                tau = -popt[1] / popt[0] if abs(popt[0]) > 1e-6 else None
                if tau is not None and 0 < tau < 1:
                    tau_per_model[model] = tau
                    ax.axvline(x=tau, color=color, linewidth=1, linestyle=":", alpha=0.5)
                    ax.annotate(
                        f"τ={tau:.2f}", xy=(tau, 0.5),
                        xytext=(tau + 0.03, 0.55), fontsize=9, color=color,
                        fontweight="bold",
                    )
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Logistic fit failed for {model}: {e}")

    ax.axvspan(0.7, 1.0, alpha=0.08, color="#f85149", label="Danger Zone (IG > 0.7)")
    ax.set_xlabel("Peak Interference Gauge (cosine similarity of updates)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title("The Interference Cliff: Accuracy Collapses at Update Collapse", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Graph D saved: {output_path}")

    return tau_per_model


# ---------------------------------------------------------------------------
# Unit of Difficulty: Λ (problem) and τ (model) summary
# ---------------------------------------------------------------------------

def compute_difficulty_summary(
    df: pd.DataFrame,
    tau_per_model: dict,
    output_path: str,
):
    """
    Compute and print the dual unit of difficulty:
      Λ (Lambda) = mean LW per token for each problem (model-independent)
      τ (Tau)    = interference ceiling per model (from Graph D logistic)

    Outputs a summary table to logger and saves to CSV.
    """
    df = df.copy()
    df["Total_Latent_Work"] = pd.to_numeric(df["Total_Latent_Work"], errors="coerce")
    df["Total_Tokens_Generated"] = pd.to_numeric(df["Total_Tokens_Generated"], errors="coerce")

    # --- Λ per problem (averaged across models) ---
    df["LW_per_token"] = df["Total_Latent_Work"] / df["Total_Tokens_Generated"].clip(lower=1)
    lambda_per_problem = df.groupby("Problem_ID")["LW_per_token"].mean()

    # --- Summary table per model ---
    summary_rows = []
    for model in sorted(df["Model_Name"].unique()):
        mdf = df[df["Model_Name"] == model]
        accuracy = mdf["Is_Correct"].mean()
        solved = mdf[mdf["Is_Correct"] == True]
        failed = mdf[mdf["Is_Correct"] == False]

        tau = tau_per_model.get(model, float("nan"))
        mean_lw_solved = solved["Total_Latent_Work"].mean() if not solved.empty else 0
        mean_lw_failed = failed["Total_Latent_Work"].mean() if not failed.empty else 0

        summary_rows.append({
            "Model": model,
            "Tau_Ceiling": f"{tau:.3f}" if not np.isnan(tau) else "N/A",
            "Mean_LW_Solved": f"{mean_lw_solved:.1f}",
            "Mean_LW_Failed": f"{mean_lw_failed:.1f}",
            "Accuracy": f"{accuracy:.1%}",
            "N_Problems": len(mdf),
        })

    # Print to log
    logger.info("\n" + "=" * 80)
    logger.info("UNIT OF DIFFICULTY SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"{'Model':<30} {'τ (Ceiling)':>12} {'Λ Solved':>10} {'Λ Failed':>10} "
        f"{'Accuracy':>10} {'N':>6}"
    )
    logger.info("-" * 80)
    for row in summary_rows:
        logger.info(
            f"{row['Model']:<30} {row['Tau_Ceiling']:>12} {row['Mean_LW_Solved']:>10} "
            f"{row['Mean_LW_Failed']:>10} {row['Accuracy']:>10} {row['N_Problems']:>6}"
        )
    logger.info("=" * 80)
    logger.info(
        f"Λ (problem difficulty) range: "
        f"{lambda_per_problem.min():.2f} — {lambda_per_problem.max():.2f} "
        f"(median {lambda_per_problem.median():.2f})"
    )
    logger.info("=" * 80)

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Difficulty summary saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate TED-LW visualizations")
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "limo_latent_work_results.csv")
    traces_dir = os.path.join(args.results_dir, "traces")
    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} results from {csv_path}")
    logger.info(f"Models: {df['Model_Name'].unique().tolist()}")
    logger.info(f"Accuracy:\n{df.groupby('Model_Name')['Is_Correct'].mean().to_string()}")

    plot_capability_frontier(df, os.path.join(plots_dir, "graph_a_capability_frontier.png"))
    plot_interference_comparison(traces_dir, df, os.path.join(plots_dir, "graph_b_interference_comparison.png"))
    plot_choke_point(traces_dir, df, os.path.join(plots_dir, "graph_c_choke_point.png"))
    tau_map = plot_interference_cliff(df, os.path.join(plots_dir, "graph_d_interference_cliff.png"))

    # Unit of Difficulty summary
    compute_difficulty_summary(
        df, tau_map,
        os.path.join(args.results_dir, "difficulty_summary.csv"),
    )

    logger.info(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
