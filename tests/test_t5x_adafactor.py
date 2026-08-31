import copy

import pytest
import torch

from modules.t5x_adafactor import T5XAdafactor


def _run_step(optimizer: T5XAdafactor, parameter: torch.Tensor, grad) -> None:
    parameter.grad = torch.tensor(grad, dtype=parameter.dtype)
    optimizer.step()
    optimizer.zero_grad()


def test_unfactored_two_step_update_matches_t5x_reference() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = T5XAdafactor([parameter], lr=0.01)

    _run_step(optimizer, parameter, [0.5, -0.25])
    _run_step(optimizer, parameter, [0.25, 0.75])

    # Fixed reference values from the public T5X equations at global steps 0/1:
    # decay = 1 - (step + 1)^-0.8, parameter RMS scaling, and RMS clipping.
    torch.testing.assert_close(
        parameter,
        torch.tensor([0.9739315165639421, -2.003819076961261], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    state = optimizer.state[parameter]
    torch.testing.assert_close(
        state["v"],
        torch.tensor([0.1423095292190280, 0.3496745887492587], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    assert optimizer.param_groups[0]["step"] == 2


def test_factored_two_step_update_and_state_shapes_match_t5x_reference() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    )
    optimizer = T5XAdafactor(
        [parameter], lr=0.01, min_dim_size_to_factor=2
    )

    _run_step(optimizer, parameter, [[1.0, 2.0], [3.0, 4.0]])
    _run_step(optimizer, parameter, [[4.0, 3.0], [2.0, 1.0]])

    torch.testing.assert_close(
        parameter,
        torch.tensor(
            [
                [0.9418848555420720, 1.9409159809710588],
                [2.9511590559697427, 3.9624587095690726],
            ],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-12,
    )
    state = optimizer.state[parameter]
    assert state["v_row"].shape == (2,)
    assert state["v_col"].shape == (2,)
    torch.testing.assert_close(
        state["v_row"],
        torch.tensor([8.243491774985173, 6.756508225014826], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        state["v_col"],
        torch.tensor([7.871745887492587, 7.128254112507413], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_factor_threshold_and_checkpoint_round_trip() -> None:
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
    optimizer = T5XAdafactor([parameter], lr=0.01)
    _run_step(optimizer, parameter, [[1.0, 2.0], [3.0, 4.0]])
    assert set(optimizer.state[parameter]) == {"v"}

    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(saved_parameter.clone())
    restored_optimizer = T5XAdafactor([restored_parameter], lr=0.01)
    restored_optimizer.load_state_dict(saved_optimizer)

    next_grad = [[0.25, 0.5], [0.75, 1.0]]
    _run_step(optimizer, parameter, next_grad)
    _run_step(restored_optimizer, restored_parameter, next_grad)
    torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_optimizer.state[restored_parameter]["v"],
        optimizer.state[parameter]["v"],
        rtol=0,
        atol=0,
    )
    assert restored_optimizer.param_groups[0]["step"] == 2


def test_rank_greater_than_two_is_rejected_in_scoped_compatibility_mode() -> None:
    parameter = torch.nn.Parameter(torch.ones((2, 2, 2)))
    optimizer = T5XAdafactor([parameter], lr=0.01, min_dim_size_to_factor=2)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="rank <= 2"):
        optimizer.step()
