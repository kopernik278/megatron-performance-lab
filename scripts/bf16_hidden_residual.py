"""Keep Transformer hidden/residual activations in BF16. Parameters stay FP32."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


BF16_HIDDEN_DTYPE = torch.bfloat16
WRAPPER_FLAG = "_bf16_hidden_residual_wrapped"
ORIGINAL_FORWARD_ATTR = "_bf16_hidden_residual_original_forward"


def _cast_hidden(hidden_states: Any) -> Any:
    if hasattr(hidden_states, "unwrap"):
        hidden_states = hidden_states.unwrap()
    if not torch.is_tensor(hidden_states):
        return hidden_states
    if not hidden_states.is_floating_point():
        return hidden_states
    if hidden_states.dtype == BF16_HIDDEN_DTYPE:
        return hidden_states
    return hidden_states.to(dtype=BF16_HIDDEN_DTYPE)


def enable_bf16_hidden_residual_stream(model: nn.Module) -> dict[str, Any]:
    """Cast decoder-input activations to BF16 once.

    Residual connections copy the incoming hidden state. After this wrap,
    that residual is BF16, so Megatron's ``x.to(residual.dtype)`` no longer
    promotes layer outputs back to FP32. Master weights and optimizer
    state are not modified.
    """
    decoder = model.decoder
    if getattr(decoder, WRAPPER_FLAG, False):
        return {
            "already_wrapped": True,
            "wrapped_module": "decoder",
            "target_dtype": "bfloat16",
        }

    original_forward = decoder.forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        if args:
            hidden_states, *rest = args
            return original_forward(_cast_hidden(hidden_states), *rest, **kwargs)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = _cast_hidden(kwargs["hidden_states"])
        return original_forward(*args, **kwargs)

    decoder.forward = wrapped_forward  # type: ignore[method-assign]
    setattr(decoder, WRAPPER_FLAG, True)
    setattr(decoder, ORIGINAL_FORWARD_ATTR, original_forward)
    return {
        "already_wrapped": False,
        "wrapped_module": "decoder",
        "target_dtype": "bfloat16",
        "parameter_storage": "unchanged",
    }


def assert_fp32_master_weights(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    ]
