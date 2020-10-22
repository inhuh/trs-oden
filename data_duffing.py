"""
We refer a code https://github.com/greydanus/hamiltonian-nn/blob/master/experiment-spring/data.py
from the following publication:
@inproceedings{greydanus2019hamiltonian,
  title={Hamiltonian neural networks},
  author={Greydanus, Samuel and Dzamba, Misko and Yosinski, Jason},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15353--15363},
  year={2019}
}
"""

import numpy as np
from scipy.integrate import solve_ivp


def dynamics(t, y, alpha, beta, gamma, delta):
    q, p = np.split(y, 2)
    dq = p
    dp = - alpha * q - beta * q ** 3 - gamma * p + delta * np.cos(t)
    return np.concatenate([dq, dp], axis=-1)


def get_trajectory(t_span, timestep, noise_level, seed, params):
    alpha, beta, gamma, delta = params[0], params[1], params[2], params[3]
    np.random.seed(seed)
    t_eval = np.linspace(t_span[0], t_span[1], timestep)
    y0 = np.random.uniform(low=-1, high=1, size=2)
    y0 = y0 / np.sqrt(np.sum(y0 ** 2))
    r = np.random.uniform(low=0.2, high=1.0, size=1)
    y0 = r * y0
    spring_ivp = solve_ivp(fun=lambda t, y: dynamics(t, y, alpha=alpha, beta=beta, gamma=gamma, delta=delta),
                           t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    q += noise_level * np.random.randn(*q.shape)
    p += noise_level * np.random.randn(*p.shape)
    return t_eval, np.stack([q, p]).T


def get_dataset(nb_samples, nb_timestep, t_span, seed, noise_level, params):
    ts, xs = [], []
    for i in range(nb_samples):
        t, x = get_trajectory(t_span=t_span, timestep=(nb_timestep + 1), seed=(seed + i), noise_level=noise_level, params=params)
        ts.append(t), xs.append(x)
    ts, xs = np.expand_dims(np.stack(ts, axis=0), axis=-1), np.stack(xs, axis=0)
    return ts, xs
