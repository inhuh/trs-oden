"""
Raw data (real_double_linear_h_1.txt) from:
@article{schmidt2009distilling,
  title={Distilling free-form natural laws from experimental data},
  author={Schmidt, Michael and Lipson, Hod},
  journal={science},
  volume={324},
  number={5923},
  pages={81--85},
  year={2009},
  publisher={American Association for the Advancement of Science}
}
"""

import os
import numpy as np


def get_dataset(test_ratio):
    this_dir = os.path.abspath(os.getcwd())
    raw_data = np.loadtxt(os.path.join(this_dir, 'real_double_linear_h_1.txt'))
    ts = raw_data[:, 1]
    xs = raw_data[:, 2:] / 100
    split_ix = int(len(xs) * (1 - test_ratio))
    ts_train, xs_train = ts[:split_ix].reshape((1, -1, 1)), xs[:split_ix].reshape((1, -1, 4))
    ts_test, xs_test = ts[split_ix:].reshape((1, -1, 1)), xs[split_ix:].reshape((1, -1, 4))
    return ts_train, xs_train, ts_test, xs_test
