import os
import argparse
import numpy as np
import pickle as pkl
from model import ODENetwork
from data_real import get_dataset
from utils import leapfrog_solver, runge_kutta_solver, reshape_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


this_dir = os.path.abspath(os.getcwd())


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_folder', default='experiment-real-5', type=str)

    # Data variables
    parser.add_argument('--test_ratio', default=0.4, type=float)

    # Model variables
    parser.add_argument('--nb_object', default=2, type=int)
    parser.add_argument('--nb_coords', default=1, type=int)
    parser.add_argument('--time_horizon', default=10, type=int)
    parser.add_argument('--time_augment', default=False, type=bool)
    parser.add_argument('--nb_units', default=1000, type=int)
    parser.add_argument('--nb_layers', default=1, type=int)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--use_time_dep_lambda', default=False, type=bool)
    parser.add_argument('--ls_cond', default=[['ode', 0.0], ['hamiltonian', 0.0], ['ode', 0.5]])
    parser.add_argument('--ls_color', default=['red', 'dodgerblue', 'seagreen'])
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--verbose', default=0, type=int)

    return parser.parse_args()


def run(args):
    plt.rc('font', size=10)
    plt.rc('axes', labelsize=12)
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['figure.figsize'] = 4.5, 4

    save_dir = os.path.join(this_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)
    print('save directory:', save_dir)

    for i in range(len(args.ls_cond)):
        cond, col = args.ls_cond[i], args.ls_color[i]
        function_type, lambda_trs = cond[0], cond[1]
        split_name = str(function_type) + '_' + str(lambda_trs) + '_' + str(args.use_time_dep_lambda)
        print('experimental condition:', split_name)

        ts_train, xs_train, ts_test, xs_test = get_dataset(test_ratio=args.test_ratio)
        t_train, x_train, y_train = reshape_data(ts=ts_train, xs=xs_train, substep=args.time_horizon)
        t_test, x_test, y_test = ts_test, xs_test[:, 0, :], xs_test[:, 1:, :]

        oden = ODENetwork(nb_object=args.nb_object, nb_coords=args.nb_coords, function_type=function_type,
                          time_horizon=args.time_horizon, time_augment=args.time_augment,
                          nb_units=args.nb_units, nb_layers=args.nb_layers,
                          activation=args.activation, lambda_trs=lambda_trs,
                          use_time_dep_lambda=args.use_time_dep_lambda, learning_rate=args.learning_rate)

        oden.solver().fit(x=[t_train, x_train], y=y_train, epochs=args.epochs, batch_size=len(x_train),
                          verbose=args.verbose)
        oden.func.save_weights(os.path.join(save_dir, split_name + '_' + 'weight.h5'))

        if args.time_augment:
            y_pred = runge_kutta_solver(ts=t_test, x0=x_test, func=oden.func)
        else:
            y_pred = leapfrog_solver(ts=t_test, x0=x_test, func=oden.func, dim=int(args.nb_object * args.nb_coords))

        q1_true, q2_true, p1_true, p2_true = y_test[:, :, 0], y_test[:, :, 1], y_test[:, :, 2], y_test[:, :, 3]
        q1_pred, q2_pred, p1_pred, p2_pred = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2], y_pred[:, :, 3]

        mse1 = 1e2 * np.mean(np.square(np.stack([q1_pred, p1_pred], axis=2) - np.stack([q1_true, p1_true], axis=2)))
        mse2 = 1e2 * np.mean(np.square(np.stack([q2_pred, p2_pred], axis=2) - np.stack([q2_true, p2_true], axis=2)))

        print('mass 1 trajectory MSE: {:.2f}'.format(mse1))
        print('mass 2 trajectory MSE: {:.2f}'.format(mse2))

        result = {'ground_truth': y_test, 'predicted': y_pred, 'mse1': mse1, 'mse2': mse2}

        with open(os.path.join(save_dir, split_name + '_' + 'result.pkl'), 'wb') as f:
            pkl.dump(result, f)

        plt.plot(q1_true[0], p1_true[0], c='k', label='Ground truth mass 1')
        plt.plot(q2_true[0], p2_true[0], 'k--', label='Ground truth mass 2')
        plt.plot(q1_pred[0], p1_pred[0], c=col, label=split_name + ' mass 1')
        plt.plot(q2_pred[0], p2_pred[0], c=col, linestyle='dashed', label=split_name + ' mass 2')
        plt.grid()
        plt.legend([Line2D([0], [0], color='k', lw=3), Line2D([0], [0], color=col, lw=3)], ['Ground truth', split_name])
        plt.xlabel('Position q'), plt.ylabel('Momentum p')
        plt.xlim([-2, 2]), plt.ylim([-3, 2.5])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, split_name + '_' + 'trajectory.png'))
        plt.close()


if __name__ == '__main__':
    run(get_args())
