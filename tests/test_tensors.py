import numpy as np
import sympy

import pytest

from devito import VectorFunction, TensorFunction, VectorTimeFunction, TensorTimeFunction
from devito import Grid, Function, TimeFunction, Dimension, Eq
from devito.types import NODE


def dimify(dimensions):
    assert isinstance(dimensions, str)
    return tuple(Dimension(name=i) for i in dimensions.split())


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3)])
def test_tensor_staggering(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)

    dims = grid.dimensions
    assert np.all(f[i, i].staggered == NODE for i in range(ndim))
    assert np.all(f[i, j].staggered == (dims[i], dims[j])
                  for i in range(ndim) for j in range(ndim) if i != j)


@pytest.mark.parametrize('func_type, ndim, sym', [
    (TensorFunction, 2, 'noop'), (TensorFunction, 3, 'noop'),
    (TensorFunction, 2, 'diag'), (TensorFunction, 3, 'diag'),
    (TensorFunction, 2, 'symm'), (TensorFunction, 3, 'symm'),
    (TensorTimeFunction, 2, 'noop'), (TensorTimeFunction, 3, 'noop'),
    (TensorTimeFunction, 2, 'diag'), (TensorTimeFunction, 3, 'diag'),
    (TensorTimeFunction, 2, 'symm'), (TensorTimeFunction, 3, 'symm')])
def test_tensor_symmetries(func_type, ndim, sym):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid, symmetric=(sym == 'symm'),
                  diagonal=(sym == 'diag'))
    if sym == 'symm':
        assert np.all(f[i, j] == f[j, i] for i in range(ndim) for j in range(ndim))
    if sym == 'diag':
        assert np.all(f[i, j] == 0 for i in range(ndim) for j in range(ndim) if i != j)
    if sym == 'noop':
        assert np.all(f[i, j] != f[j, i] for i in range(ndim) for j in range(ndim))
        assert np.all(f[i, j] != 0 for i in range(ndim) for j in range(ndim) if i != j)


@pytest.mark.parametrize('func_type, ndim', [
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_vector_staggering(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)
    dims = grid.dimensions
    assert np.all(f[i].staggered == dims[i] for i in range(ndim))


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3),
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_tensor_grid(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)
    assert np.all(ff.grid == grid for ff in f)


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3),
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_tensor_space_order(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid, space_order=10)
    assert np.all(ff.space_order == 10 for ff in f)


@pytest.mark.parametrize('func1, func2, out_type', [
    (Function, VectorFunction, VectorFunction),
    (Function, VectorTimeFunction, VectorTimeFunction),
    (TimeFunction, VectorTimeFunction, VectorTimeFunction),
    (Function, TensorFunction, TensorFunction),
    (Function, TensorTimeFunction, TensorTimeFunction),
    (TimeFunction, TensorTimeFunction, TensorTimeFunction),
    (TensorFunction, VectorFunction, VectorFunction),
    (TensorFunction, VectorTimeFunction, VectorTimeFunction),
    (TensorTimeFunction, VectorFunction, VectorTimeFunction),
    (TensorTimeFunction, VectorTimeFunction, VectorTimeFunction),
    (TensorTimeFunction, TensorFunction, TensorTimeFunction),
    (TensorTimeFunction, VectorTimeFunction, TensorTimeFunction)])
def test_tensor_matmul(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1*f2, out_type)


@pytest.mark.parametrize('func1, func2, out_type', [
    (VectorFunction, TensorFunction, VectorFunction),
    (VectorTimeFunction, TensorFunction, VectorTimeFunction),
    (VectorFunction, TensorTimeFunction, VectorTimeFunction),
    (VectorTimeFunction, TensorTimeFunction, VectorTimeFunction)])
def test_tensor_matmul_T(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1.T*f2, out_type)


@pytest.mark.parametrize('func1, func2, out_type', [
    (VectorFunction, VectorFunction, TensorFunction),
    (VectorTimeFunction, VectorTimeFunction, TensorTimeFunction),
    (VectorFunction, VectorTimeFunction, TensorTimeFunction)])
def test_tensor_outer(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1*f2.T, out_type)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_tensor_custom_dims(func1):
    dimensions = dimify('i j k')
    ndim = 3
    f = func1(name="f", dimensions=dimensions, shape=(2, 3, 4))
    assert np.all(f[i, i].staggered == NODE for i in range(ndim))
    assert np.all(f[i, j].staggered == (dimensions[i], dimensions[j])
                  for i in range(3) for j in range(3) if i != j)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_tensor_transpose(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid, symmetric=False)
    f2 = f1.T
    assert np.all([f1[i, j] == f2[j, i] for i in range(3) for j in range(3)])


@pytest.mark.parametrize('func1', [VectorFunction, VectorTimeFunction])
def test_vector_transpose(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.T
    assert f2.shape == f1.shape[::-1]
    assert np.all([f1[i] == f2[i] for i in range(3)])


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_tensor_fd(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    assert np.all([f.dx == f2 for f, f2 in zip(f1, f1.dx)])


@pytest.mark.parametrize('func1, symm, diag, expected',
                         [(TensorFunction, False, False, 9),
                          (TensorFunction, True, False, 6),
                          (TensorFunction, False, True, 3),
                          (TensorTimeFunction, False, False, 9),
                          (TensorTimeFunction, True, False, 6),
                          (TensorTimeFunction, False, True, 3)])
def test_tensor_eq(func1, symm, diag, expected):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid, symmetric=symm, diagonal=diag)
    for attr in f1[0]._fd:
        eq = Eq(f1, getattr(f1, attr))
        assert len(eq.evaluate._flatten) == expected


@pytest.mark.parametrize('func1', [VectorTimeFunction, TensorTimeFunction])
def test_save(func1):
    grid = Grid(tuple([5]*3))
    time = grid.time_dim
    f1 = func1(name="f1", grid=grid, save=10, time_order=1)
    assert all(ff.indices[0] == time for ff in f1)
    assert all(ff.indices[0] == time + time.spacing for ff in f1.forward)
    assert all(ff.indices[0] == time + 2*time.spacing for ff in f1.forward.forward)
    assert all(ff.indices[0] == time - time.spacing for ff in f1.backward)
    assert all(ff.shape[0] == 10 for ff in f1)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_sympy_matrix(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)

    sympy_f1 = f1.as_mutable()
    vec = sympy.Matrix(3, 1, np.random.rand(3))
    mat = sympy.Matrix(3, 3, np.random.rand(3, 3).ravel())
    assert all(sp - dp == 0 for sp, dp in zip(mat * f1, mat * sympy_f1))
    assert all(sp - dp == 0 for sp, dp in zip(f1 * vec, sympy_f1 * vec))


@pytest.mark.parametrize('func1', [VectorFunction, VectorTimeFunction])
def test_sympy_vector(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)

    sympy_f1 = f1.as_mutable()
    mat = sympy.Matrix(3, 3, np.random.rand(3, 3).ravel())

    assert all(sp - dp == 0 for sp, dp in zip(mat * f1, mat * sympy_f1))
