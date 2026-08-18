from __future__ import annotations

import torch

from optimizers.matrix_functions import matrix_inverse_root


def test_scalar_matrix_inverse_root_has_uniform_four_tuple_api() -> None:
    value = torch.tensor([[4.0]])
    inverse_root, used_epsilon, damped_eigenvalues, eigenvectors = matrix_inverse_root(
        value, root=2, epsilon=1.0e-3
    )
    expected = (value + 1.0e-3).pow(-0.5)
    torch.testing.assert_close(inverse_root, expected)
    assert used_epsilon == 1.0e-3
    torch.testing.assert_close(damped_eigenvalues, torch.tensor([4.001]))
    torch.testing.assert_close(eigenvectors, torch.ones(1, 1))
