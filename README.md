# Unit of Difficulty

**Unsupervised Latent Work Stress-Test & Interference Profiling for Language Models**

Quantifies the intrinsic computational cost of reasoning problems and maps the failure points of language models by measuring interference in their residual streams — without any human labels or supervised probes.

## The Core Idea

When a language model struggles with a problem, its residual stream vectors start collapsing into a narrow cone — consecutive updates become increasingly parallel (high cosine similarity), outlier dimensions explode, and the model's reasoning trajectory begins to wander. This pipeline captures that exact signal.

Two quantities emerge:

| Symbol | Name | What it measures |
|--------|------|-----------------|
| **Λ** (Lambda) | Problem Difficulty | Mean Latent Work per token, averaged across models. Model-independent. |
| **τ** (Tau) | Interference Ceiling | Peak interference at which a model's accuracy drops to 50%. Architecture-dependent. |

**The phase transition predicate:** When `Peak_IG(model, problem) > τ_model` → the model will fail. Computable *before checking the answer*.

## Metrics

Only **3 sparse layer hooks** (`L//4`, `L//2`, `L-1`) — preserves vLLM kernel fusion throughput.

| Metric | Formula | Signal |
|--------|---------|--------|
| **ICV** | `‖x_final − x_early‖₂` | Total residual displacement |
| **IG** | `cos(Δ_t, Δ_{t-1})` | Update collapse → interference |
| **CSI** | `cos(Δ_early→mid, Δ_mid→final)` | Redundant processing |
| **L∞** | `max(‖x_final‖)` | Outlier dimension explosion |
| **LW** | `ICV × (1 + IG)` | Difficulty weighted by interference |
| **Wander** | `Σ‖v_t‖ / ‖x_N − x_0‖` | Path efficiency |

## Models

All models share a ~2-3B active parameter budget per token, but use radically different architectures:

| Model | Active Params | Architecture |
|-------|--------------|-------------|
| Qwen3.5-35B-A3B | ~3B | Standard MoE |
| Nanbeige4.1-3B | 3B | Dense |
| Nemotron-3-Nano-30B-A3B | ~3B | Mamba-Transformer Hybrid MoE |
| LFM2-24B-A2B | ~2B | Conv-GQA MoE |

## Dataset

[GAIR/LIMO](https://huggingface.co/datasets/GAIR/LIMO) — 817 Olympiad and AIME-level math problems.

## Quick Start

```bash
pip install -r requirements.txt

# Single GPU
python run_pipeline.py --models all

# 2× H200 (parallel, ~35 min with stratification)
# GPU 0 — Qwen ranks all 817 problems for unbiased Λ
python run_pipeline.py --models Qwen3.5-35B-A3B,Nanbeige4.1-3B --gpu-id 0 --stratify

# GPU 1
python run_pipeline.py --models Nemotron-3-Nano-30B-A3B,LFM2-24B-A2B --gpu-id 1 --stratify --resume

# Generate plots + difficulty summary
python visualize.py
```

## Output

| File | Description |
|------|-------------|
| `limo_latent_work_results.csv` | Master ledger (all metrics per model × problem) |
| `difficulty_summary.csv` | Per-model τ, Λ (solved/failed), accuracy |
| `graph_a_capability_frontier.png` | LW vs Accuracy with logistic fit |
| `graph_b_interference_comparison.png` | IG time series: success vs failure |
| `graph_c_choke_point.png` | IG + Wander Ratio dual-axis |
| `graph_d_interference_cliff.png` | Peak_IG vs Accuracy with τ annotations |

## CLI Flags

| Flag | Description |
|------|-------------|
| `--models` | Comma-separated model names or `all` |
| `--max-problems` | Limit number of LIMO problems |
| `--batch-size` | Override batch size (default 16, auto-halves on OOM) |
| `--stratify` | Two-phase: Qwen ranks all problems, others get top/bottom 100 |
| `--gpu-id` | Pin to specific CUDA device for multi-GPU parallelism |
| `--resume` | Skip already-completed model+problem pairs |

## Project Structure

```
├── config.py          # Model registry, hyperparameters
├── dataset.py         # LIMO loader + ∖boxed{} answer grading
├── metrics.py         # ICV, IG, CSI, L∞, LW, Wander Ratio
├── hooks.py           # Architecture-agnostic layer discovery
├── inference.py       # 3-hook streaming engine (nnsight + vLLM)
├── run_pipeline.py    # CLI orchestrator
├── visualize.py       # Graphs A-D + difficulty summary
├── task.spec          # Full mathematical specification
└── tests/
    └── test_metrics.py
```

## License

MIT
