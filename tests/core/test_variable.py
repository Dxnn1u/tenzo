import numpy as np
import pytest

from tenzo import Variable


def test_variable_stores_data() -> None:
    data = np.array([1.0, 2.0])
    x = Variable(data)
    assert x.data is data
    assert x.grad is None
    assert x.creator is None


def test_variable_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError):
        Variable(1.0)  # type: ignore[arg-type]


def test_backward_without_creator_is_noop() -> None:
    x = Variable(np.array([1.0]))
    x.backward()
    assert np.allclose(x.grad, [1.0])
