import numpy as np

from graph_kernel_svm.utils import diagnose_kernel_matrix


def test_kernel_diagnostics_on_valid_kernel() -> None:
    kernel = np.array([[2.0, 1.0], [1.0, 2.0]])

    diagnostics = diagnose_kernel_matrix(kernel)

    assert diagnostics.shape == (2, 2)
    assert diagnostics.symmetry_error == 0.0
    assert diagnostics.min_diagonal == 2.0
    assert diagnostics.max_diagonal == 2.0
    assert diagnostics.approximate_min_eigenvalue is not None
    assert diagnostics.approximate_min_eigenvalue >= 0.0
    assert diagnostics.condition_warning is None
