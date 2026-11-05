import numpy as np

from tenzo import Variable
from tenzo.ops.basic import Exp, Square, exp, square


def numerical_diff(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_square_forward() -> None:
    y = Square()(Variable(np.array([2.0, 3.0])))
    assert np.allclose(y.data, [4.0, 9.0])


def test_square_backward() -> None:
    x = Variable(np.array([2.0, 3.0]))
    square(x).backward()
    assert np.allclose(x.grad, [4.0, 6.0])


def test_exp_forward() -> None:
    y = Exp()(Variable(np.array([0.0, 1.0])))
    assert np.allclose(y.data, [1.0, np.e])


def test_exp_backward() -> None:
    x = Variable(np.array([0.0, 1.0]))
    exp(x).backward()
    assert np.allclose(x.grad, [1.0, np.e])


def test_zero_dim_input_stays_ndarray() -> None:
    y = square(Variable(np.array(2.0)))
    assert isinstance(y.data, np.ndarray)
    assert y.data == 4.0


def test_chain_backward_known_value() -> None:
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    assert np.allclose(y.data, np.exp(0.5))
    assert np.allclose(x.grad, 2 * np.exp(0.5))


def test_chain_backward_matches_numerical_diff() -> None:
    def f(v: np.ndarray) -> np.ndarray:
        return np.exp(v**2) ** 2

    x = Variable(np.array([0.3, 0.5, 0.7]))
    square(exp(square(x))).backward()
    assert np.allclose(x.grad, numerical_diff(f, x.data), rtol=1e-4)
