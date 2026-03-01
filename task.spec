# TED-LW v2 Specification: Unsupervised Latent Work Stress-Test & Interference Profiling

## 1. Overview & Objective

Write a Python pipeline to quantify the structural computational load and failure points
of 4 sparse/hybrid language models. We measure Interference Gauge (IG), Cross-Segment
Interference (CSI), Outlier Pressure (L∞), revised Latent Work (LW), and Wander Ratio
as these models attempt to solve 817 LIMO reasoning problems. The pipeline outputs a
master CSV, per-run trace files, four visualizations, and a difficulty summary table.

## 2. Environment & Hardware

- Hardware: 2× NVIDIA H200 (141GB VRAM each)
- Frameworks: torch (CUDA), nnsight (v0.6+), vllm, scipy, matplotlib, pandas
- Dataset: GAIR/LIMO (817 advanced math problems)
- Models:
  - Qwen/Qwen3.5-35B-A3B (MoE, ~3B active)
  - Nanbeige/Nanbeige4.1-3B (Dense, 3B)
  - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (Mamba-Transformer hybrid MoE, ~3B active)
  - LiquidAI/LFM2-24B-A2B (Conv-GQA MoE, ~2B active)

## 3. Mathematical Definitions & Tracking

We intercept the residual stream vector x_t^(l) ∈ ℝ^d at 3 sparse checkpoint layers
and token generation step t. Let L = total layers.

### Hook Architecture: 3 Sparse Checkpoints

Only hook layer outputs at:
- l_early = ⌊L/4⌋
- l_mid   = ⌊L/2⌋  
- l_final = L-1

This preserves vLLM kernel fusion throughput (3 hooks vs 48).

### Metrics

1. **Inter-Checkpoint Velocity (ICV)**: Total residual displacement through the model.
   ICV_t = ||x_t^(final) - x_t^(early)||₂

2. **Interference Gauge (IG)**: Cosine similarity between consecutive token-step
   update vectors. Measures update collapse.
   - Δ_t = x_t^(final) - x_t^(early)
   - IG_t = max(0, cos(Δ_t, Δ_{t-1}))
   - Signal: IG ≈ 0 → orthogonal updates (healthy). IG → 1.0 → narrow cone (saturated).

3. **Cross-Segment Interference (CSI)**: Cosine similarity between the early→mid
   and mid→final update segments at the same token step.
   - Δ_early = x_t^(mid) - x_t^(early)
   - Δ_late  = x_t^(final) - x_t^(mid)
   - CSI_t = max(0, cos(Δ_early, Δ_late))
   - Signal: CSI → 1.0 → early and late processing stuck in same subspace.

4. **Outlier Pressure (L∞)**: Maximum absolute value of any coordinate in x_t^(final).
   - Signal: L∞ explosion → outlier dimensions saturated as global memory buffers.

5. **Latent Work (revised)**:
   LW_t = ICV_t × (1 + IG_t)
   When the model works hard (high ICV) AND updates collapse (high IG), LW spikes.

6. **Total Latent Work**:
   Total_LW = Σ_{t=1}^{N} LW_t

7. **Wander Ratio**: Path efficiency of the final-layer residual stream.
   Wander = (Σ ||x_t - x_{t-1}||₂) / ||x_N - x_0||₂

## 4. The Unit of Difficulty

Two dual quantities formally define difficulty:

### Λ (Lambda) — Problem Difficulty (model-independent)
Λ_p = (1/T) × Σ_t ICV_t × (1 + IG_t)

Mean LW per token for problem p, averaged across all models. Two 500-token problems
can have wildly different Λ if one keeps interference low and the other saturates.

### τ (Tau) — Model Interference Ceiling (architecture-dependent)
τ_m = Peak_IG value at which model m's accuracy drops to 50%

Derived from Graph D's logistic regression fit. Architecture-dependent — Mamba may
give a different τ than GQA attention.

### Phase Transition Predicate
A model m fails on problem p with high probability when:
  Peak_IG(m, p) > τ_m

Computable before checking the answer. Failure predicted from residual geometry alone.

## 5. Optimization & Memory Constraints

- Batch Size: Hardcap 16, auto-halve on OOM.
- Inference Engine: nnsight + vLLM backend (dispatch=True) for PagedAttention.
- vLLM Failure: ABORT with actionable error. NO silent HuggingFace fallback (50x slower).
- Streaming Math: Compute ICV, IG, CSI, L∞ immediately per token. No global caching.
- GPU Pinning: --gpu-id flag sets CUDA_VISIBLE_DEVICES for multi-GPU parallelism.
- All extracted scalars moved off GPU via .cpu().item() instantly.

## 6. Pipeline Execution

### Phase 1: Batched Streaming Inference
- Loop through 4 models. Process LIMO in batches of 16.
- Generation: max_new_tokens=8000, temperature=0.0 (greedy).
- --stratify mode: Run Qwen3.5-35B-A3B (most capable) on ALL 817 problems first, rank by LW
  to get unbiased Λ, keep top-100 (hardest) + bottom-100 (easiest), run remaining 3 models
  on those 200 only. Using the weakest model (Nanbeige) as ranker would poison Λ — its
  confusion on hard problems inflates LW artificially.

### Phase 2: Dynamic Metric Extraction
- 3-hook streaming extraction per token step.
- Grade final text against LIMO ground truth (\boxed{} extraction).

### Phase 3: Logging & Export
Master CSV: limo_latent_work_results.csv
Columns:
- Model_Name
- Problem_ID
- Is_Correct (Boolean)
- Total_Tokens_Generated
- Total_Latent_Work
- Peak_IG
- Mean_IG
- Peak_CSI
- Peak_L_Inf
- Wander_Ratio
- Truncated
- Error

Per-run .npz traces: lw_per_token, icv_per_token, ig_per_token, csi_per_token, l_inf_per_token.

Difficulty summary: difficulty_summary.csv with per-model τ, Λ (solved/failed), accuracy.

## 7. Visualization Deliverables

### Graph A: Capability Frontier (Scatter Plot)
- X: Total_Latent_Work (log scale)
- Y: Accuracy (0/1, logistic regression fit per model)
- Series: One color per model

### Graph B: Interference Comparison (Time Series)
- Side-by-side IG + CSI time series for one success vs one failure from same model
- Rolling average smoothed (window=50)

### Graph C: Choke Point (Dual-Axis Time Series)
- Select a failed high-LW run from LFM2-24B-A2B
- Left Y-axis (blue): IG (smoothed)
- Right Y-axis (red): Rolling Wander Ratio
- Annotate where IG spikes and Wander explodes

### Graph D: The Interference Cliff (The Smoking Gun)
- X: Peak_IG
- Y: Accuracy (0/1)
- Logistic regression fit per model with τ annotated
- Red danger zone shading for IG > 0.7
- Summary: Universal phase transition where accuracy collapses when IG > τ

## 8. Deployment (2× H200)

```bash
# Terminal 1 (GPU 0) — Qwen runs all 817 as ranker, Nanbeige gets 200 stratified
python run_pipeline.py --models Qwen3.5-35B-A3B,Nanbeige4.1-3B --gpu-id 0 --stratify

# Terminal 2 (GPU 1)
python run_pipeline.py --models Nemotron-3-Nano-30B-A3B,LFM2-24B-A2B --gpu-id 1 --stratify --resume

# After both complete:
python visualize.py
```

Expected time: ~35 min with --stratify on 2× H200.

## 9. Why SVD Was Removed (v1 → v2 Errata)

The original spec used SVD-based TED (Truncated SVD on [W=50, d_model] window).
Three fatal flaws:

1. SVD rank cap: rank([50 × d_model]) ≤ 50. TED/d_model ≈ 0.016 max.
   Graph D showed a flat line near zero. Mathematically impossible to approach the "cliff."

2. Superposition: Models pack N >> d features non-orthogonally (Elhage et al., 2022).
   There is no brick wall at ρ = TED/d_model = 1.0. Failure comes from simultaneous
   interference, not from exceeding orthogonal capacity.

3. vLLM hook overhead: Hooking all 48 layers at every token step = 384K Python
   callbacks per problem. Breaks kernel fusion, throughput collapses.

Fix: Replace SVD with cosine similarity (IG/CSI) + L∞ outlier pressure.
Hook only 3 layers instead of all. Mathematically sound, engineering-sound.