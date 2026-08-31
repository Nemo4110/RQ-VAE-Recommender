"""Scoped PyTorch compatibility implementation of T5X Adafactor defaults.

This module implements the update semantics used by the public T5X Adafactor
for the scalar, vector, and matrix parameters present in this repository's
Hugging Face T5 decoder.  It intentionally does not claim full compatibility
with T5X logical-axis rules for rank-greater-than-two scanned/fused parameters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class T5XAdafactor(Optimizer):
    """Adafactor with the public T5X framework-default update semantics.

    The defaults mirror T5X's public optimizer defaults relevant to TIGER:
    factored second moments for sufficiently large matrices, parameter-RMS
    learning-rate scaling, ``t**-0.8`` second-moment decay, RMS update clipping,
    and no first-moment momentum or weight decay.

    Rank-greater-than-two parameters are rejected rather than silently using a
    heuristic that has not been numerically checked against T5X logical axes.
    The current decoder contains only scalar, vector, and matrix parameters.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        *,
        factored: bool = True,
        multiply_by_parameter_scale: bool = True,
        decay_rate: float = 0.8,
        step_offset: int = 0,
        clipping_threshold: Optional[float] = 1.0,
        weight_decay_rate: Optional[float] = None,
        min_dim_size_to_factor: int = 128,
        epsilon1: float = 1e-30,
        epsilon2: float = 1e-3,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        if clipping_threshold is not None and clipping_threshold < 1.0:
            raise ValueError("clipping_threshold must be at least 1.0")
        if weight_decay_rate is not None and not 0.0 <= weight_decay_rate < 1.0:
            raise ValueError("weight_decay_rate must be in [0, 1)")
        if min_dim_size_to_factor <= 0:
            raise ValueError("min_dim_size_to_factor must be positive")
        if epsilon1 < 0 or epsilon2 < 0:
            raise ValueError("epsilon values must be non-negative")

        defaults = {
            "lr": lr,
            "factored": factored,
            "multiply_by_parameter_scale": multiply_by_parameter_scale,
            "decay_rate": decay_rate,
            "step_offset": step_offset,
            "clipping_threshold": clipping_threshold,
            "weight_decay_rate": weight_decay_rate,
            "min_dim_size_to_factor": min_dim_size_to_factor,
            "epsilon1": epsilon1,
            "epsilon2": epsilon2,
            # T5X keeps one optimizer-global step.  This project uses one group;
            # storing it in the group makes checkpoint round-trips explicit.
            "step": 0,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _decay_rate_pow(step: int, exponent: float) -> float:
        adjusted_step = step + 1
        if adjusted_step <= 0:
            raise ValueError("step - step_offset must be at least zero")
        return 1.0 - adjusted_step ** (-exponent)

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.square().mean().sqrt()

    @staticmethod
    def _uses_factored_state(
        parameter: Tensor, *, factored: bool, min_dim_size_to_factor: int
    ) -> bool:
        return (
            factored
            and parameter.ndim == 2
            and min(parameter.shape) >= min_dim_size_to_factor
        )

    @staticmethod
    def _initialize_state(state: dict, parameter: Tensor, use_factored: bool) -> None:
        if use_factored:
            state["v_row"] = torch.zeros(
                parameter.shape[0], dtype=parameter.dtype, device=parameter.device
            )
            state["v_col"] = torch.zeros(
                parameter.shape[1], dtype=parameter.dtype, device=parameter.device
            )
        else:
            state["v"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )

    @torch.no_grad()
    def step(self, closure=None):
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            global_step = int(group["step"])
            decay_step = global_step - int(group["step_offset"])
            decay = self._decay_rate_pow(decay_step, float(group["decay_rate"]))
            mixing_rate = 1.0 - decay

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("T5XAdafactor does not support sparse gradients")
                if torch.is_complex(parameter):
                    raise RuntimeError("T5XAdafactor does not support complex parameters")
                if parameter.ndim > 2:
                    raise RuntimeError(
                        "T5XAdafactor compatibility is limited to rank <= 2; "
                        f"received shape {tuple(parameter.shape)}"
                    )

                use_factored = self._uses_factored_state(
                    parameter,
                    factored=bool(group["factored"]),
                    min_dim_size_to_factor=int(group["min_dim_size_to_factor"]),
                )
                state = self.state[parameter]
                if not state:
                    self._initialize_state(state, parameter, use_factored)
                elif use_factored != ("v_row" in state):
                    raise RuntimeError(
                        "parameter factorization changed after optimizer state initialization"
                    )

                grad_sqr = grad.square().add(float(group["epsilon1"]))
                if use_factored:
                    v_row = state["v_row"]
                    v_col = state["v_col"]
                    v_row.mul_(decay).add_(
                        grad_sqr.mean(dim=1), alpha=mixing_rate
                    )
                    v_col.mul_(decay).add_(
                        grad_sqr.mean(dim=0), alpha=mixing_rate
                    )
                    row_factor = (v_row / v_row.mean()).rsqrt().unsqueeze(1)
                    col_factor = v_col.rsqrt().unsqueeze(0)
                    update = grad * row_factor * col_factor
                else:
                    variance = state["v"]
                    variance.mul_(decay).add_(grad_sqr, alpha=mixing_rate)
                    update = grad * variance.rsqrt()

                clipping_threshold = group["clipping_threshold"]
                if clipping_threshold is not None:
                    clipping_denom = torch.clamp(
                        self._rms(update) / float(clipping_threshold), min=1.0
                    )
                    update = update / clipping_denom

                update_scale = float(group["lr"])
                if group["multiply_by_parameter_scale"]:
                    parameter_scale = torch.clamp(
                        self._rms(parameter), min=float(group["epsilon2"])
                    )
                    update_scale = update_scale * parameter_scale

                weight_decay_rate = group["weight_decay_rate"]
                if weight_decay_rate is not None:
                    parameter.mul_(1.0 - float(weight_decay_rate))
                parameter.sub_(update * update_scale)

            group["step"] = global_step + 1

        return loss
