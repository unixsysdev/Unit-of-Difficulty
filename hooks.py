"""
TED-LW Hook Management
======================
Architecture-agnostic discovery of residual stream layers
and hook attachment for metric extraction.
"""

import logging
from typing import List, Tuple, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer Discovery
# ---------------------------------------------------------------------------

# Common attribute paths where transformer blocks live across architectures
_LAYER_ATTR_PATHS = [
    # Standard HuggingFace Transformers
    ("model", "layers"),           # Llama, Qwen, Mistral, etc.
    ("model", "decoder", "layers"),
    ("transformer", "h"),          # GPT-2, GPT-J, etc.
    ("transformer", "layers"),
    ("gpt_neox", "layers"),        # GPT-NeoX / Pythia
    # Nemotron / Mamba-Transformer hybrids
    ("model", "layers"),
    ("backbone", "layers"),
    # LFM / Liquid
    ("model", "blocks"),
    ("blocks",),
    # Generic fallback
    ("layers",),
]


def discover_layers(model) -> List[nn.Module]:
    """
    Dynamically discover the ordered list of residual block modules
    in an arbitrary transformer-family architecture.

    Walks common attribute paths, then falls back to recursive search
    for repeated nn.Module subclasses that look like transformer blocks.

    Args:
        model: The loaded model (nnsight LanguageModel or VLLM wrapper).
               We access the underlying torch model via model.model or
               the model object itself.

    Returns:
        Ordered list of layer/block nn.Module instances.

    Raises:
        RuntimeError: If no layers can be discovered.
    """
    # Get the underlying torch model from nnsight wrappers
    root = _unwrap_model(model)

    # Strategy 1: Walk known attribute paths
    for path in _LAYER_ATTR_PATHS:
        obj = root
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                logger.info(
                    f"Discovered {len(obj)} layers via path: "
                    f"{'.'.join(path)}"
                )
                return list(obj)
        except AttributeError:
            continue

    # Strategy 2: Recursive search for the largest ModuleList
    candidates = []
    for name, module in root.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            candidates.append((name, module))

    if candidates:
        # Pick the largest ModuleList (most likely the main layer stack)
        best_name, best_list = max(candidates, key=lambda x: len(x[1]))
        logger.info(
            f"Discovered {len(best_list)} layers via recursive search: "
            f"'{best_name}'"
        )
        return list(best_list)

    raise RuntimeError(
        f"Cannot discover layers in model architecture. "
        f"Top-level modules: {[n for n, _ in root.named_children()]}"
    )


def _unwrap_model(model) -> nn.Module:
    """Unwrap nnsight / vLLM wrappers to get the raw torch model."""
    # nnsight LanguageModel stores the model in .model
    if hasattr(model, "_model"):
        return model._model
    if hasattr(model, "model"):
        inner = model.model
        # Some wrappers nest further
        if hasattr(inner, "model"):
            return inner.model
        return inner
    return model


# ---------------------------------------------------------------------------
# Checkpoint Indices
# ---------------------------------------------------------------------------

def get_checkpoint_indices(num_layers: int) -> Tuple[int, int, int]:
    """
    Compute layer indices for the three TED sampling checkpoints.

    Returns:
        (early_idx, mid_idx, final_idx) — all 0-indexed.

    Examples:
        32 layers → (7, 15, 31)
        48 layers → (11, 23, 47)
    """
    early = max(0, num_layers // 4 - 1)
    mid = max(0, num_layers // 2 - 1)
    final = num_layers - 1
    return (early, mid, final)


# ---------------------------------------------------------------------------
# Module Path Discovery for nnsight Tracing
# ---------------------------------------------------------------------------

def get_layer_module_path(model, layer_idx: int) -> str:
    """
    Get the dotted attribute path to a specific layer's output,
    suitable for nnsight tracing.

    This inspects the model to find the correct path (e.g.,
    'model.layers.15' or 'transformer.h.15').

    Args:
        model: The nnsight-wrapped model.
        layer_idx: 0-indexed layer number.

    Returns:
        Dotted path string like 'model.layers.15'.
    """
    root = _unwrap_model(model)

    for path in _LAYER_ATTR_PATHS:
        obj = root
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > layer_idx:
                full_path = ".".join(path) + f".{layer_idx}"
                return full_path
        except AttributeError:
            continue

    raise RuntimeError(f"Cannot find module path for layer {layer_idx}")


def get_nnsight_layer_accessor(model, layer_idx: int):
    """
    Return a reference to the nnsight-proxied layer module,
    navigating the model's attribute tree.

    Args:
        model: The nnsight model (LanguageModel or VLLM).
        layer_idx: 0-indexed layer number.

    Returns:
        The nnsight module proxy for the specified layer.
    """
    # Try known paths on the nnsight model object directly
    paths_to_try = [
        lambda m, i: m.model.layers[i],
        lambda m, i: m.model.decoder.layers[i],
        lambda m, i: m.transformer.h[i],
        lambda m, i: m.transformer.layers[i],
        lambda m, i: m.gpt_neox.layers[i],
        lambda m, i: m.model.blocks[i],
        lambda m, i: m.backbone.layers[i],
        lambda m, i: m.blocks[i],
        lambda m, i: m.layers[i],
    ]

    for accessor in paths_to_try:
        try:
            layer = accessor(model, layer_idx)
            return layer
        except (AttributeError, IndexError, TypeError):
            continue

    raise RuntimeError(
        f"Cannot access layer {layer_idx} via nnsight proxy. "
        f"Model type: {type(model)}"
    )
