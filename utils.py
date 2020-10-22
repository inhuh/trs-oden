import numpy as np


def leapfrog_solver(ts, x0, func, dim):
    dts = np.diff(ts, axis=1)
    x = x0
    ls_x = []
    for i in range(dts.shape[1]):
        dt = dts[:, i, :]
        q, p = x[:, :dim], x[:, dim:]
        p = p + 0.5 * func.predict(x)[:, dim:] * dt
        q = q + func.predict(np.concatenate([q, p], axis=-1))[:, :dim] * dt
        p = p + 0.5 * func.predict(np.concatenate([q, p], axis=-1))[:, dim:] * dt
        x = np.concatenate([q, p], axis=-1)
        ls_x.append(x)
    x = np.stack(ls_x, axis=1)
    return x


def runge_kutta_solver(ts, x0, func, time_augment=True):
    dts = np.diff(ts, axis=1)
    x = x0
    ls_x = []
    for i in range(dts.shape[1]):
        t = ts[:, i, :]
        dt = dts[:, i, :]
        if time_augment:
            dx1 = func.predict([t, x]) * dt
            dx2 = func.predict([t + 0.5 * dt, x + 0.5 * dx1]) * dt
            dx3 = func.predict([t + 0.5 * dt, x + 0.5 * dx2]) * dt
            dx4 = func.predict([t + dt, x + dx3]) * dt
        else:
            dx1 = func.predict(x) * dt
            dx2 = func.predict(x + 0.5 * dx1) * dt
            dx3 = func.predict(x + 0.5 * dx2) * dt
            dx4 = func.predict(x + dx3) * dt
        dx = (1/6) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
        x = x + dx
        ls_x.append(x)
    x = np.stack(ls_x, axis=1)
    return x


def reshape_data(ts, xs, substep):
    ts_res, x0_res, xs_res = [], [], []
    for i in range(xs.shape[0]):
        for j in range(int(xs.shape[1] / substep)):
            x0_sub = xs[i, j * substep, :]
            xs_sub = xs[i, (j * substep + 1):((j + 1) * substep + 1), :]
            ts_sub = ts[i, (j * substep):((j + 1) * substep + 1), :]
            ts_res.append(ts_sub), x0_res.append(x0_sub), xs_res.append(xs_sub)
    return np.stack(ts_res, axis=0), np.stack(x0_res, axis=0), np.stack(xs_res, axis=0)
