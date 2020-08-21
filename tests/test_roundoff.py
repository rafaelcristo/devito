import pytest
import numpy as np

from devito import Grid, Constant, TimeFunction, Eq, Operator


class TestRoundoff(object):
    """
    Class for testing SubDomains
    """
    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_forward(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = np.float32(dat)
        s = np.float32(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = np.float32(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float32(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-5)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_backward(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = np.float32(dat)
        s = np.float32(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward))

        initial_condition = np.float32(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float32(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-5)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_fb(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r')
        r.data = np.float32(dat)
        s = np.float32(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1))
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2)

        initial_condition = np.float32(0.7235)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward+(1.0/s)*dt*f1.forward-f1.forward))

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float32(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        #assert np.isclose(np.linalg.norm(f0.data[np.mod(iterations+1,3)]-f1.data[iterations+1]), 0.0, atol=1.e-5)
        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-5)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_forward_double(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r', dtype=np.float64)
        r.data = np.float64(dat)
        s = np.float64(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=np.float64)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=np.float64)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2, dtype=np.float64)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = np.float64(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float64(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-12)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_backward_double(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r', dtype=np.float64)
        r.data = np.float64(dat)
        s = np.float64(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=np.float64)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=np.float64)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2, dtype=np.float64)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward))

        initial_condition = np.float64(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float64(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-12)

    @pytest.mark.parametrize('dat', [0.5, 0.624, 1.0, 1.5, 2.0, 3.0, 3.6767, 4.0])
    def test_lm_fb_double(self, dat):
        """
        ...
        """
        iterations = 10000
        r= Constant(name='r', dtype=np.float64)
        r.data = np.float64(dat)
        s = np.float64(0.1)

        grid = Grid(shape=(2, 2), extent=(1, 1), dtype=np.float64)
        dt = grid.stepping_dim.spacing

        f0 = TimeFunction(name='f0', grid=grid, time_order=2, dtype=np.float64)
        f1 = TimeFunction(name='f1', grid=grid, time_order=2, save=iterations+2, dtype=np.float64)

        lmap0 = Eq(f0.forward, r*f0*(1.0-f0+(1.0/s)*dt*f0.backward-f0.backward+(1.0/s)*dt*f0.forward-f0.forward))
        lmap1 = Eq(f1.forward, r*f1*(1.0-f1+(1.0/s)*dt*f1.backward-f1.backward+(1.0/s)*dt*f1.forward-f1.forward))

        initial_condition = np.float64(0.7235)

        f0.data[1, :, :] = initial_condition
        f1.data[1, :, :] = initial_condition

        op0 = Operator([Eq(f0.forward, np.float64(0.0)), lmap0])
        op1 = Operator(lmap1)

        op0(time_m=1, time_M=iterations, dt=s)
        op1(time_m=1, time_M=iterations, dt=s)

        assert np.allclose(f0.data[np.mod(iterations+1,3)], f1.data[iterations+1], rtol=1.e-12)
