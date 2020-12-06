import os
import argparse
import numpy as np
import pickle as pkl
from model import ODENetwork
from data_duffing import get_dataset
from utils import leapfrog_solver, runge_kutta_solver, reshape_data


this_dir = os.path.abspath(os.getcwd())


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_folder', default='experiment-duffing-test', type=str)

    # Data variables
    parser.add_argument('--osc_params', default=(1, 0, 0, 0), type=tuple)
    parser.add_argument('--train_nb_sample', default=50, type=int)
    parser.add_argument('--train_nb_timestep', default=30, type=int)
    parser.add_argument('--train_t_span', default=(0, 3), type=tuple)
    parser.add_argument('--train_seed', default=0, type=int)
    parser.add_argument('--train_noise_level', default=0.1, type=float)
    parser.add_argument('--test_nb_sample', default=50, type=int)
    parser.add_argument('--test_nb_timestep', default=200, type=int)
    parser.add_argument('--test_t_span', default=(0, 20), type=tuple)
    parser.add_argument('--test_seed', default=999, type=int)
    parser.add_argument('--test_noise_level', default=0.0, type=float)

    # Model variables
    parser.add_argument('--nb_object', default=1, type=int)
    parser.add_argument('--nb_coords', default=1, type=int)
    parser.add_argument('--function_type', default='ode', type=str)
    parser.add_argument('--time_horizon', default=10, type=int)
    parser.add_argument('--time_augment', action='store_true')
    parser.add_argument('--nb_units', default=1000, type=int)
    parser.add_argument('--nb_layers', default=1, type=int)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--lambda_trs', default=10.0, type=float)
    parser.add_argument('--use_time_dep_lambda', action='store_true')
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--verbose', default=2, type=int)

    return parser.parse_args()


def train(args):
    save_dir = os.path.join(this_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)
    print('save directory:', save_dir)

    split_name = str(args.function_type) + '_' + str(args.lambda_trs) + '_' + str(args.use_time_dep_lambda)
    print('experimental condition:', split_name)

    ts, xs = get_dataset(nb_samples=args.train_nb_sample, nb_timestep=args.train_nb_timestep,
                         t_span=args.train_t_span, seed=args.train_seed, noise_level=args.train_noise_level,
                         params=args.osc_params)
    t_train, x_train, y_train = reshape_data(ts=ts, xs=xs, substep=args.time_horizon)

    oden = ODENetwork(nb_object=args.nb_object, nb_coords=args.nb_coords, function_type=args.function_type,
                      time_horizon=args.time_horizon, time_augment=args.time_augment,
                      nb_units=args.nb_units, nb_layers=args.nb_layers,
                      activation=args.activation, lambda_trs=args.lambda_trs,
                      use_time_dep_lambda=args.use_time_dep_lambda, learning_rate=args.learning_rate)

    oden.solver().fit(x=[t_train, x_train], y=y_train, epochs=args.epochs, batch_size=len(x_train), verbose=args.verbose)
    oden.func.save_weights(os.path.join(save_dir, split_name + '_' + 'weight.h5'))

    ts, xs = get_dataset(nb_samples=args.test_nb_sample, nb_timestep=args.test_nb_timestep,
                         t_span=args.test_t_span, seed=args.test_seed, noise_level=args.test_noise_level,
                         params=args.osc_params)
    t_test, x_test, y_test = ts, xs[:, 0, :], xs[:, 1:, :]

    if args.time_augment:
        y_pred = runge_kutta_solver(ts=t_test, x0=x_test, func=oden.func)
    else:
        y_pred = leapfrog_solver(ts=t_test, x0=x_test, func=oden.func, dim=int(args.nb_object * args.nb_coords))

    q_true, p_true, q_pred, p_pred = y_test[:, :, 0], y_test[:, :, 1], y_pred[:, :, 0], y_pred[:, :, 1]
    e_true = args.osc_params[0] * np.square(q_true) + (args.osc_params[1] * np.square(np.square(q_true)) / 2) + np.square(p_true)
    e_pred = args.osc_params[0] * np.square(q_pred) + (args.osc_params[1] * np.square(np.square(q_pred)) / 2) + np.square(p_pred)

    mse_t_per_sample = np.mean(np.mean(np.square(y_test - y_pred), axis=-1), axis=-1)
    mse_t, std_t = 1e2 * np.mean(mse_t_per_sample), 1e2 * np.std(mse_t_per_sample)

    mse_e_per_sample = np.mean(np.square(e_true - e_pred), axis=-1)
    mse_e, std_e = 1e2 * np.mean(mse_e_per_sample), 1e2 * np.std(mse_e_per_sample)
    print('trajectory MSE: {:.2f} pm {:.2f}'.format(mse_t, std_t))
    print('energy MSE: {:.2f} pm {:.2f}'.format(mse_e, std_e))

    result = {'ground_truth': y_test, 'predicted': y_pred,
              'mse_t': mse_t, 'std_t': std_t, 'mse_e': mse_e, std_t: std_e}

    with open(os.path.join(save_dir, split_name + '_' + 'result.pkl'), 'wb') as f:
        pkl.dump(result, f)


if __name__ == '__main__':
    train(get_args())
