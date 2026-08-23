"""Keep Transformer hidden/residual activations in BF16. Parameters stay FP32."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.jit import jit_fuser


BF16_HIDDEN_DTYPE = torch.bfloat16
WRAPPER_FLAG = "_bf16_hidden_residual_wrapped"
ORIGINAL_FORWARD_ATTR = "_bf16_hidden_residual_original_forward"
ORIGINAL_SELF_ATTN_BDA_ATTR = "_bf16_hidden_original_self_attn_bda"
ORIGINAL_MLP_BDA_ATTR = "_bf16_hidden_original_mlp_bda"


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


def _bias_dropout_add_func(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    """Run BDA after independently aligning x and bias to the residual dtype."""
    x, bias = x_with_bias
    inplace = (
        not training
        and not x.requires_grad
        and not residual.requires_grad
        and (bias is None or not bias.requires_grad)
    )

    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
    if bias is not None and bias.dtype != residual.dtype:
        bias = bias.to(residual.dtype)

    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training, inplace=inplace)
    if inplace:
        out.add_(residual)
    else:
        out = residual + out
    return out


def bias_dropout_add_unfused(training: bool):
    def _bias_dropout_add(
        x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
        residual: torch.Tensor,
        prob: float,
    ) -> torch.Tensor:
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


@jit_fuser
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)


@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)


def get_persistent_bf16_bias_dropout_add(training: bool, fused: bool):
    if not fused:
        return bias_dropout_add_unfused(training)
    return bias_dropout_add_fused_train if training else bias_dropout_add_fused_inference


def enable_bf16_hidden_residual_stream(model: nn.Module) -> dict[str, Any]:
    """Cast decoder input once and prevent FP32 bias from promoting BDA outputs.

    Master weights and optimizer state are not modified.
    """
    decoder = model.decoder
    if getattr(decoder, WRAPPER_FLAG, False):
        return {
            "already_wrapped": True,
            "wrapped_module": "decoder",
            "target_dtype": "bfloat16",
            "bda_behavior": "x_and_bias_independently_cast_to_residual_dtype",
            "wrapped_layer_count": len(decoder.layers),
        }

    wrapped_layer_count = 0
    for layer in decoder.layers:
        setattr(layer, ORIGINAL_SELF_ATTN_BDA_ATTR, layer.self_attn_bda)
        setattr(layer, ORIGINAL_MLP_BDA_ATTR, layer.mlp_bda)
        layer.self_attn_bda = get_persistent_bf16_bias_dropout_add
        layer.mlp_bda = get_persistent_bf16_bias_dropout_add
        wrapped_layer_count += 1

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
        "bda_behavior": "x_and_bias_independently_cast_to_residual_dtype",
        "wrapped_layer_count": wrapped_layer_count,
    }


def assert_fp32_master_weights(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    ]
