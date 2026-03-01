"""
TED-LW Inference Engine (v2 — 3 Sparse Hooks)
===============================================
Streaming inference with interference-based metric extraction.
Only hooks 3 layer checkpoints (early, mid, final) to preserve
vLLM kernel fusion throughput.

Supports two backends:
  1. nnsight + vLLM (preferred): PagedAttention, high throughput
  2. nnsight + HF Transformers (emergency fallback, requires user approval)

Extracts per-token: ICV, IG, CSI, L∞, and revised LW on-the-fly.
"""

import gc
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch

import config
import hooks
import metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single problem's inference + metric extraction."""
    problem_id: int
    model_name: str
    generated_text: str = ""
    total_tokens: int = 0
    is_correct: bool = False

    # Aggregate metrics
    total_latent_work: float = 0.0
    peak_ig: float = 0.0           # Peak interference gauge
    mean_ig: float = 0.0           # Mean interference gauge
    peak_csi: float = 0.0          # Peak cross-segment interference
    peak_l_inf: float = 0.0        # Peak outlier pressure (L∞)
    wander_ratio: float = 0.0

    # Trace arrays (for .npz export)
    lw_per_token: List[float] = field(default_factory=list)
    icv_per_token: List[float] = field(default_factory=list)
    ig_per_token: List[float] = field(default_factory=list)
    csi_per_token: List[float] = field(default_factory=list)
    l_inf_per_token: List[float] = field(default_factory=list)

    # Status
    truncated: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Runs inference with streaming metric extraction using 3 sparse hooks.
    """

    def __init__(
        self,
        model_spec: config.ModelSpec,
        batch_size: int = config.MAX_BATCH_SIZE,
    ):
        self.model_spec = model_spec
        self.batch_size = batch_size
        self.model = None
        self.num_layers = 0
        self.checkpoint_indices = (0, 0, 0)  # (early, mid, final)
        self._d_model = 0

    def load_model(self):
        """Load the model using the appropriate backend."""
        logger.info(
            f"Loading model: {self.model_spec.hf_id} "
            f"(backend={self.model_spec.backend})"
        )

        if self.model_spec.backend == "vllm":
            self._load_vllm()
        else:
            self._load_hf()

        logger.info(
            f"Model loaded. Layers={self.num_layers}, "
            f"d_model={self._d_model}, "
            f"checkpoints={self.checkpoint_indices}"
        )

    def _load_vllm(self):
        """Load model via nnsight's vLLM backend."""
        try:
            from nnsight.modeling.vllm import VLLM
            self.model = VLLM(
                self.model_spec.hf_id,
                dispatch=True,
                tensor_parallel_size=1,
            )
        except Exception as e:
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"FATAL: vLLM backend failed for {self.model_spec.hf_id}.\n"
                f"Error: {e}\n\n"
                f"The HuggingFace fallback is ~50x slower (20+ hours vs 30 min)\n"
                f"and will NOT be used automatically.\n\n"
                f"Options:\n"
                f"  1. Fix vLLM installation: pip install vllm>=0.12.0\n"
                f"  2. Skip this model: --models <other_models>\n"
                f"  3. Use a different model with confirmed vLLM support\n"
                f"{'='*60}"
            ) from e

        self._discover_architecture()

    def _load_hf(self):
        """Load model via nnsight's HuggingFace Transformers backend."""
        from nnsight import LanguageModel

        self.model = LanguageModel(
            self.model_spec.hf_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._discover_architecture()

    def _discover_architecture(self):
        """Discover layers, d_model, and set checkpoint indices."""
        layers = hooks.discover_layers(self.model)
        self.num_layers = len(layers)
        self.checkpoint_indices = hooks.get_checkpoint_indices(self.num_layers)

        # Discover d_model from model config
        model_config = None
        for attr in ["config", "_config", "model_config"]:
            if hasattr(self.model, attr):
                model_config = getattr(self.model, attr)
                break

        if model_config is not None:
            self._d_model = getattr(
                model_config, "hidden_size",
                getattr(model_config, "d_model",
                        getattr(model_config, "n_embd", 0))
            )

        if self._d_model == 0:
            first_layer = layers[0]
            for p in first_layer.parameters():
                if p.ndim == 2:
                    self._d_model = p.shape[-1]
                    break

        if self._d_model == 0:
            self._d_model = 4096
            logger.warning(f"Could not detect d_model, assuming {self._d_model}")

    def unload_model(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed")

    # -----------------------------------------------------------------------
    # Batch Inference
    # -----------------------------------------------------------------------

    def run_batch(self, problems: List[Dict]) -> List[RunResult]:
        """
        Run inference on a batch with OOM auto-recovery.
        """
        current_batch_size = min(self.batch_size, len(problems))

        while current_batch_size >= 1:
            try:
                return self._run_batch_inner(problems, current_batch_size)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"OOM at batch_size={current_batch_size}. "
                    f"Halving to {current_batch_size // 2}."
                )
                gc.collect()
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

        logger.error("OOM even at batch_size=1. Returning error results.")
        return [
            RunResult(
                problem_id=p["problem_id"],
                model_name=self.model_spec.name,
                error="OOM at batch_size=1",
            )
            for p in problems
        ]

    def _run_batch_inner(
        self, problems: List[Dict], batch_size: int,
    ) -> List[RunResult]:
        """Process problems in sub-batches."""
        from dataset import format_prompt

        all_results = []
        sub_batches = [
            problems[i : i + batch_size]
            for i in range(0, len(problems), batch_size)
        ]

        for batch_idx, batch in enumerate(sub_batches):
            prompts = [format_prompt(p["question"]) for p in batch]

            logger.info(
                f"  Sub-batch {batch_idx + 1}/{len(sub_batches)} "
                f"({len(batch)} problems)"
            )

            t0 = time.time()
            batch_results = self._run_single_subbatch(batch, prompts)
            elapsed = time.time() - t0

            total_tokens = sum(r.total_tokens for r in batch_results)
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Sub-batch done: {total_tokens} tokens in {elapsed:.1f}s "
                f"({tok_per_sec:.0f} tok/s)"
            )

            all_results.extend(batch_results)

        return all_results

    def _run_single_subbatch(
        self, problems: List[Dict], prompts: List[str],
    ) -> List[RunResult]:
        """
        TRUE batched inference with PROXY-SAFE metric extraction.

        Inside nnsight's tracer.iter, all tensors are Proxy objects that
        build a computation graph. You CANNOT call .item(), .cpu(), or
        use Python if-statements on them. All math must use PyTorch ops.

        Strategy:
          1. Compute metrics as proxy ops (torch.norm, cosine_similarity)
          2. .save() only scalar results (near-zero VRAM)
          3. Extract .value[b].item() AFTER the trace block completes
        """
        B = len(problems)
        early_idx, mid_idx, final_idx = self.checkpoint_indices

        results = [
            RunResult(problem_id=p["problem_id"], model_name=self.model_spec.name)
            for p in problems
        ]

        # Saved proxy references — populated during tracing, read after
        step_proxies = []           # List[dict] of saved scalar proxies per step
        wander_velocities = []      # List of saved velocity proxies
        first_residuals_proxy = None
        last_residuals_proxy = None

        try:
            with self.model.trace(
                prompts,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=False,
            ) as tracer:
                prev_deltas = None
                prev_finals = None

                for gen_step in tracer.iter:
                    # --- Extract proxy residuals [B, d_model] ---
                    r_e = hooks.get_nnsight_layer_accessor(self.model, early_idx).output
                    r_m = hooks.get_nnsight_layer_accessor(self.model, mid_idx).output
                    r_f = hooks.get_nnsight_layer_accessor(self.model, final_idx).output

                    if isinstance(r_e, tuple): r_e = r_e[0]
                    if isinstance(r_m, tuple): r_m = r_m[0]
                    if isinstance(r_f, tuple): r_f = r_f[0]

                    r_early = r_e[:, -1, :]   # [B, d_model]
                    r_mid   = r_m[:, -1, :]
                    r_final = r_f[:, -1, :]

                    # --- Metrics as PROXY OPS (no .item(), no if) ---
                    # ICV: L2 norm of displacement [B]
                    icv = torch.norm(r_final - r_early, p=2, dim=-1)

                    # IG: cosine similarity of consecutive updates [B]
                    delta_full = r_final - r_early
                    if prev_deltas is None:
                        ig = icv * 0.0  # Zero tensor, same shape, same device
                    else:
                        ig = torch.nn.functional.cosine_similarity(
                            delta_full, prev_deltas, dim=-1
                        )
                        ig = torch.relu(ig)  # Clamp negative to 0

                    # CSI: cosine similarity of model halves [B]
                    delta_e2m = r_mid - r_early
                    delta_m2f = r_final - r_mid
                    csi = torch.nn.functional.cosine_similarity(
                        delta_e2m, delta_m2f, dim=-1
                    )
                    csi = torch.relu(csi)

                    # L∞: max absolute value [B]
                    l_inf = torch.max(torch.abs(r_final), dim=-1).values

                    # --- .save() SCALAR proxies only (near-zero VRAM) ---
                    step_proxies.append({
                        "icv": icv.save(),
                        "ig": ig.save(),
                        "csi": csi.save(),
                        "l_inf": l_inf.save(),
                    })

                    # --- Wander Ratio state ---
                    if first_residuals_proxy is None:
                        first_residuals_proxy = r_final.save()

                    if prev_finals is not None:
                        v = torch.norm(r_final - prev_finals, p=2, dim=-1)
                        wander_velocities.append(v.save())

                    last_residuals_proxy = r_final.save()

                    prev_deltas = delta_full
                    prev_finals = r_final

                    tracer.next()

        except AttributeError:
            logger.info("tracer.iter not available, falling back to per-problem hooks")
            fallback_results = []
            for prob, prompt in zip(problems, prompts):
                res = RunResult(
                    problem_id=prob["problem_id"],
                    model_name=self.model_spec.name,
                )
                try:
                    self._run_single_problem_fallback(prompt, res)
                except Exception as e:
                    logger.error(f"Fallback error on problem {prob['problem_id']}: {e}")
                    res.error = str(e)
                fallback_results.append(res)
            return fallback_results
        except Exception as e:
            logger.error(f"Tracing error in batch: {e}", exc_info=True)
            for res in results:
                res.error = str(e)
            return results

        # ==================================================================
        # AFTER TRACE — graph executed, proxies have real values
        # ==================================================================
        try:
            batch_outputs = tracer.output if hasattr(tracer, "output") else [""] * B
            if isinstance(batch_outputs, str):
                batch_outputs = [batch_outputs]
        except Exception:
            batch_outputs = [""] * B

        for b in range(B):
            res = results[b]
            res.generated_text = str(batch_outputs[b]) if b < len(batch_outputs) else ""
            res.total_tokens = len(step_proxies)

            for step_dict in step_proxies:
                icv_val = step_dict["icv"].value[b].item()
                ig_val  = step_dict["ig"].value[b].item()
                csi_val = step_dict["csi"].value[b].item()
                linf_val = step_dict["l_inf"].value[b].item()

                res.icv_per_token.append(icv_val)
                res.ig_per_token.append(ig_val)
                res.csi_per_token.append(csi_val)
                res.l_inf_per_token.append(linf_val)
                res.lw_per_token.append(icv_val * (1.0 + ig_val))

                res.peak_ig = max(res.peak_ig, ig_val)
                res.peak_csi = max(res.peak_csi, csi_val)
                res.peak_l_inf = max(res.peak_l_inf, linf_val)

            res.total_latent_work = sum(res.lw_per_token)
            if res.ig_per_token:
                res.mean_ig = sum(res.ig_per_token) / len(res.ig_per_token)

            # Wander Ratio
            if first_residuals_proxy is not None and last_residuals_proxy is not None:
                total_velocity = sum(v.value[b].item() for v in wander_velocities)
                displacement = torch.norm(
                    last_residuals_proxy.value[b] - first_residuals_proxy.value[b], p=2
                ).item()
                res.wander_ratio = total_velocity / max(displacement, 1e-12)

            res.truncated = (res.total_tokens >= config.MAX_NEW_TOKENS)

        return results

    # -----------------------------------------------------------------------
    # Fallback: PyTorch forward hooks (for nnsight API variations)
    # -----------------------------------------------------------------------

    def _run_single_problem_fallback(self, prompt: str, result: RunResult):
        """
        Fallback using direct PyTorch forward hooks on only 3 layers.
        Used when nnsight's tracer.iter API isn't available.
        """
        early_idx, mid_idx, final_idx = self.checkpoint_indices
        layers = hooks.discover_layers(self.model)

        icv_per_token = []
        ig_per_token = []
        csi_per_token = []
        l_inf_per_token = []
        lw_per_token = []

        prev_delta = None
        prev_final_residual = None
        first_final_residual = None
        last_final_residual = None
        velocity_sum = 0.0
        peak_ig = 0.0
        peak_csi = 0.0
        peak_l_inf = 0.0
        token_count = 0

        # Hook only the 3 checkpoint layers
        _captured = {}

        def make_hook(label):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    _captured[label] = output[0][:, -1, :].detach()
                else:
                    _captured[label] = output[:, -1, :].detach()
            return hook_fn

        hook_handles = [
            layers[early_idx].register_forward_hook(make_hook("early")),
            layers[mid_idx].register_forward_hook(make_hook("mid")),
            layers[final_idx].register_forward_hook(make_hook("final")),
        ]

        try:
            if hasattr(self.model, "tokenizer"):
                tokenizer = self.model.tokenizer
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_spec.hf_id)

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(
                next(layers[0].parameters()).device
            )

            for step in range(config.MAX_NEW_TOKENS):
                _captured.clear()

                with torch.no_grad():
                    outputs = self.model._model(input_ids)

                if "early" not in _captured or "final" not in _captured:
                    break

                token_count += 1

                r_early = _captured["early"][0].cpu()
                r_mid = _captured.get("mid", _captured["early"])[0].cpu()
                r_final = _captured["final"][0].cpu()

                # ICV
                icv = metrics.compute_icv(r_final, r_early)
                icv_per_token.append(icv)

                # Update vectors
                delta_full = r_final - r_early
                delta_early_to_mid = r_mid - r_early
                delta_mid_to_final = r_final - r_mid

                # IG
                ig = metrics.compute_interference(delta_full, prev_delta) if prev_delta is not None else 0.0
                ig_per_token.append(ig)
                peak_ig = max(peak_ig, ig)

                # CSI
                csi = metrics.compute_csi(delta_early_to_mid, delta_mid_to_final)
                csi_per_token.append(csi)
                peak_csi = max(peak_csi, csi)

                # L∞
                l_inf, _ = metrics.compute_outlier_pressure(r_final)
                l_inf_per_token.append(l_inf)
                peak_l_inf = max(peak_l_inf, l_inf)

                # LW
                lw = metrics.compute_latent_work(icv, ig)
                lw_per_token.append(lw)

                # Wander state
                if first_final_residual is None:
                    first_final_residual = r_final.clone()
                if prev_final_residual is not None:
                    velocity_sum += metrics.compute_icv(r_final, prev_final_residual)

                prev_final_residual = r_final
                last_final_residual = r_final
                prev_delta = delta_full

                # Next token
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if hasattr(tokenizer, "eos_token_id") and next_token.item() == tokenizer.eos_token_id:
                    break

        finally:
            for h in hook_handles:
                h.remove()

        # Final metrics
        wander = 0.0
        if first_final_residual is not None and last_final_residual is not None:
            wander = metrics.compute_wander_ratio(velocity_sum, first_final_residual, last_final_residual)

        try:
            generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            generated_text = ""

        result.generated_text = generated_text
        result.total_tokens = token_count
        result.total_latent_work = metrics.compute_total_lw(lw_per_token)
        result.peak_ig = peak_ig
        result.mean_ig = sum(ig_per_token) / max(1, len(ig_per_token))
        result.peak_csi = peak_csi
        result.peak_l_inf = peak_l_inf
        result.wander_ratio = wander
        result.lw_per_token = lw_per_token
        result.icv_per_token = icv_per_token
        result.ig_per_token = ig_per_token
        result.csi_per_token = csi_per_token
        result.l_inf_per_token = l_inf_per_token
        result.truncated = (token_count >= config.MAX_NEW_TOKENS)
