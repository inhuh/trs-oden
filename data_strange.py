import numpy as np
from scipy.integrate import solve_ivp


def dynamics(t, y):
    x1, x2, x3 = np.split(y, 3)
    dx1 = 1 + x2 * x3
    dx2 = -x1 * x3
    dx3 = x2 ** 2 + 2 * x2 * x3
    return np.concatenate([dx1, dx2, dx3], axis=-1)


def get_trajectory(t_span, timestep, seed, noise_level, y0=None):
    np.random.seed(seed)
    t_eval = np.linspace(t_span[0], t_span[1], timestep)
    if y0 is None:
        y0 = np.array([0, 0, np.random.uniform(1, 3, 1)])
    else:
        y0 = y0
    spring_ivp = solve_ivp(fun=lambda t, y: dynamics(t, y), t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    x1, x2, x3 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2]
    x1 += noise_level * np.random.randn(*x1.shape)
    x2 += noise_level * np.random.randn(*x2.shape)
    x3 += noise_level * np.random.randn(*x3.shape)
    return t_eval, np.stack([x1, x2, x3]).T


def get_dataset(nb_samples, nb_timestep, t_span, seed, noise_level):
    ts, xs = [], []
    for i in range(nb_samples):
        t, x = get_trajectory(t_span=t_span, timestep=(nb_timestep + 1), seed=(seed + i), noise_level=noise_level)
        ts.append(t), xs.append(x)
    ts, xs = np.expand_dims(np.stack(ts, axis=0), axis=-1), np.stack(xs, axis=0)
    return ts, xs
