"""
TED-LW Pipeline Orchestrator
=============================
Main CLI entry point that runs the full pipeline:
  Phase 1: Batched Streaming Inference
  Phase 2: Answer Grading
  Phase 3: CSV + .npz Export
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from typing import List, Dict

import numpy as np
import torch

import config
import dataset
import inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TED-LW: Unsupervised Latent Work Stress-Test Pipeline"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma-separated model short names to run, or 'all'. "
            f"Available: {', '.join(config.MODELS.keys())}"
        ),
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit the number of LIMO problems (default: all 817)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.MAX_BATCH_SIZE,
        help=f"Batch size for inference (default: {config.MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.RESULTS_DIR,
        help=f"Output directory (default: {config.RESULTS_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV, skipping already-completed model+problem pairs",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help=(
            "Two-phase stratification: run the most capable model (Qwen3.5-35B-A3B) on all "
            "problems first to rank by LW (unbiased \u039b), then run remaining models on only "
            "the top-100 heaviest and bottom-100 lightest problems. Fits in ~45min on H200."
        ),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "CUDA device ID to use (e.g. 0 or 1). For 2×H200 parallelism, run two "
            "instances:\n"
            "  GPU 0: python run_pipeline.py --models Qwen3.5-35B-A3B,Nanbeige4.1-3B --gpu-id 0\n"
            "  GPU 1: python run_pipeline.py --models Nemotron-3-Nano-30B-A3B,LFM2-24B-A2B --gpu-id 1\n"
            "Both write to the same CSV; use --resume to merge."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 1 + 2: Inference + Grading
# ---------------------------------------------------------------------------

def run_model(
    model_spec: config.ModelSpec,
    problems: List[Dict],
    batch_size: int,
    traces_dir: str,
    completed_pairs: set,
) -> List[Dict]:
    """
    Run a single model on all problems, extract metrics, and grade answers.

    Returns a list of result dicts ready for CSV export.
    """
    logger.info(f"{'='*60}")
    logger.info(f"MODEL: {model_spec.name} ({model_spec.hf_id})")
    logger.info(f"{'='*60}")

    # Filter out already-completed problems if resuming
    remaining = [
        p for p in problems
        if (model_spec.name, p["problem_id"]) not in completed_pairs
    ]

    if not remaining:
        logger.info(f"All problems already completed for {model_spec.name}, skipping")
        return []

    logger.info(f"Problems to process: {len(remaining)} (skipped {len(problems) - len(remaining)})")

    # Initialize engine
    engine = inference.InferenceEngine(model_spec, batch_size=batch_size)
    engine.load_model()

    # Run inference in batches
    all_results = []
    batches = dataset.batch_problems(remaining, batch_size)
    total_batches = len(batches)

    t_start = time.time()

    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"Batch {batch_idx + 1}/{total_batches} "
            f"(problems {batch[0]['problem_id']}-{batch[-1]['problem_id']})"
        )

        results = engine.run_batch(batch)

        # Grade answers
        for result, problem in zip(results, batch):
            result.is_correct = dataset.grade_answer(
                result.generated_text, problem["answer"]
            )

        all_results.extend(results)

        # Progress logging
        elapsed = time.time() - t_start
        done = len(all_results)
        if done > 0:
            eta = (elapsed / done) * (len(remaining) - done)
            logger.info(
                f"Progress: {done}/{len(remaining)} | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                f"Correct so far: {sum(1 for r in all_results if r.is_correct)}/{done}"
            )

    engine.unload_model()

    # Convert results to export format
    csv_rows = []
    for result in all_results:
        # Save trace arrays as .npz
        trace_path = os.path.join(
            traces_dir,
            f"{model_spec.name}_problem_{result.problem_id:04d}.npz",
        )
        _save_trace(result, trace_path)

        csv_rows.append({
            "Model_Name": result.model_name,
            "Problem_ID": result.problem_id,
            "Is_Correct": result.is_correct,
            "Total_Tokens_Generated": result.total_tokens,
            "Total_Latent_Work": f"{result.total_latent_work:.2f}",
            "Peak_IG": f"{result.peak_ig:.6f}",
            "Mean_IG": f"{result.mean_ig:.6f}",
            "Peak_CSI": f"{result.peak_csi:.6f}",
            "Peak_L_Inf": f"{result.peak_l_inf:.2f}",
            "Wander_Ratio": f"{result.wander_ratio:.4f}",
            "Truncated": result.truncated,
            "Error": result.error or "",
        })

    total_time = time.time() - t_start
    correct = sum(1 for r in all_results if r.is_correct)
    logger.info(
        f"Model {model_spec.name} complete: "
        f"{correct}/{len(all_results)} correct ({100*correct/max(1,len(all_results)):.1f}%) "
        f"in {total_time:.0f}s"
    )

    return csv_rows


def _save_trace(result: inference.RunResult, path: str):
    """Save per-run trace arrays as compressed .npz."""
    try:
        np.savez_compressed(
            path,
            lw_per_token=np.array(result.lw_per_token, dtype=np.float32),
            icv_per_token=np.array(result.icv_per_token, dtype=np.float32),
            ig_per_token=np.array(result.ig_per_token, dtype=np.float32),
            csi_per_token=np.array(result.csi_per_token, dtype=np.float32),
            l_inf_per_token=np.array(result.l_inf_per_token, dtype=np.float32),
            total_latent_work=result.total_latent_work,
            peak_ig=result.peak_ig,
            mean_ig=result.mean_ig,
            peak_csi=result.peak_csi,
            peak_l_inf=result.peak_l_inf,
            wander_ratio=result.wander_ratio,
            total_tokens=result.total_tokens,
            is_correct=result.is_correct,
        )
    except Exception as e:
        logger.warning(f"Failed to save trace to {path}: {e}")


# ---------------------------------------------------------------------------
# Phase 3: CSV Export
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "Model_Name",
    "Problem_ID",
    "Is_Correct",
    "Total_Tokens_Generated",
    "Total_Latent_Work",
    "Peak_IG",
    "Mean_IG",
    "Peak_CSI",
    "Peak_L_Inf",
    "Wander_Ratio",
    "Truncated",
    "Error",
]


def load_existing_csv(csv_path: str) -> tuple:
    """Load existing CSV rows and completed (model, problem_id) pairs."""
    rows = []
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                completed.add((row["Model_Name"], int(row["Problem_ID"])))
        logger.info(f"Loaded {len(rows)} existing results from {csv_path}")
    return rows, completed


def write_csv(csv_path: str, rows: List[Dict], append: bool = False):
    """Write result rows to CSV."""
    mode = "a" if append else "w"
    write_header = not append or not os.path.exists(csv_path)

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve models
    if args.models.lower() == "all":
        model_specs = list(config.MODELS.values())
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        model_specs = []
        for name in model_names:
            if name in config.MODELS:
                model_specs.append(config.MODELS[name])
            else:
                logger.error(
                    f"Unknown model: {name}. "
                    f"Available: {', '.join(config.MODELS.keys())}"
                )
                sys.exit(1)

    # Pin to specific GPU if requested (for multi-GPU parallelism)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logger.info(f"Pinned to GPU {args.gpu_id} (CUDA_VISIBLE_DEVICES={args.gpu_id})")

    # Create output directories
    results_dir = args.output_dir
    traces_dir = os.path.join(results_dir, "traces")
    plots_dir = os.path.join(results_dir, "plots")
    csv_path = os.path.join(results_dir, "limo_latent_work_results.csv")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load dataset
    problems = dataset.load_limo(max_problems=args.max_problems)

    # Load existing results if resuming
    existing_rows = []
    completed_pairs = set()
    if args.resume:
        existing_rows, completed_pairs = load_existing_csv(csv_path)

    # Run each model
    t_global = time.time()
    all_new_rows = []

    if args.stratify:
        # ---------------------------------------------------------------
        # STRATIFIED MODE: Two-phase run for ~45min H200 budget
        # Phase A: Run fastest model on ALL problems to rank by LW
        # Phase B: Run remaining models on top-100 + bottom-100 only
        # ---------------------------------------------------------------
        ranker_key = "Qwen3.5-35B-A3B"
        ranker_spec = config.MODELS[ranker_key]

        logger.info("=" * 60)
        logger.info("STRATIFIED MODE: Phase A — Ranking all problems with %s", ranker_key)
        logger.info("=" * 60)

        ranker_rows = run_model(
            model_spec=ranker_spec,
            problems=problems,
            batch_size=args.batch_size,
            traces_dir=traces_dir,
            completed_pairs=completed_pairs,
        )
        all_new_rows.extend(ranker_rows)

        if ranker_rows:
            write_csv(csv_path, ranker_rows, append=bool(existing_rows or completed_pairs))
            completed_pairs.update(
                (row["Model_Name"], int(row["Problem_ID"]))
                for row in ranker_rows
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Rank problems by Total_Latent_Work
        ranked = sorted(
            ranker_rows,
            key=lambda r: float(r.get("Total_Latent_Work", 0)),
        )

        if len(ranked) >= 200:
            light_ids = {int(r["Problem_ID"]) for r in ranked[:100]}
            heavy_ids = {int(r["Problem_ID"]) for r in ranked[-100:]}
        else:
            light_ids = {int(r["Problem_ID"]) for r in ranked[:len(ranked) // 2]}
            heavy_ids = {int(r["Problem_ID"]) for r in ranked[len(ranked) // 2:]}

        stratified_ids = light_ids | heavy_ids
        stratified_problems = [p for p in problems if p["problem_id"] in stratified_ids]

        logger.info(
            f"Phase A complete. Selected {len(light_ids)} light + "
            f"{len(heavy_ids)} heavy = {len(stratified_problems)} problems "
            f"for remaining models."
        )

        # Phase B: Remaining models on stratified subset
        remaining_specs = [
            spec for spec in model_specs if spec.name != ranker_key
        ]

        for model_spec in remaining_specs:
            model_rows = run_model(
                model_spec=model_spec,
                problems=stratified_problems,
                batch_size=args.batch_size,
                traces_dir=traces_dir,
                completed_pairs=completed_pairs,
            )
            all_new_rows.extend(model_rows)

            if model_rows:
                write_csv(csv_path, model_rows, append=True)
                completed_pairs.update(
                    (row["Model_Name"], int(row["Problem_ID"]))
                    for row in model_rows
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # ---------------------------------------------------------------
        # FULL MODE: All models × all problems
        # ---------------------------------------------------------------
        for model_spec in model_specs:
            model_rows = run_model(
                model_spec=model_spec,
                problems=problems,
                batch_size=args.batch_size,
                traces_dir=traces_dir,
                completed_pairs=completed_pairs,
            )
            all_new_rows.extend(model_rows)

            if model_rows:
                write_csv(csv_path, model_rows, append=bool(existing_rows or completed_pairs))
                completed_pairs.update(
                    (row["Model_Name"], int(row["Problem_ID"]))
                    for row in model_rows
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - t_global

    # Summary
    total_rows = len(existing_rows) + len(all_new_rows)
    logger.info(f"{'='*60}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total results: {total_rows} rows")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Traces: {traces_dir}")
    logger.info(f"Run 'python visualize.py' to generate plots")


if __name__ == "__main__":
    main()
