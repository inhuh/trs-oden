import os
import argparse
import numpy as np
import pickle as pkl
from model_strange import ODENetwork
from data_strange import get_dataset, get_trajectory
from utils import runge_kutta_solver, reshape_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


this_dir = os.path.abspath(os.getcwd())


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--save_folder', default='experiment-strange-6', type=str)

    # Data variables
    parser.add_argument('--train_nb_sample', default=1000, type=int)
    parser.add_argument('--train_nb_timestep', default=400, type=int)
    parser.add_argument('--train_t_span', default=(0, 20), type=tuple)
    parser.add_argument('--train_seed', default=0, type=int)
    parser.add_argument('--train_noise_level', default=0.05, type=float)
    parser.add_argument('--test_nb_sample', default=50, type=int)
    parser.add_argument('--test_nb_timestep', default=400, type=int)
    parser.add_argument('--test_t_span', default=(0, 20), type=tuple)
    parser.add_argument('--test_seed', default=2000, type=int)
    parser.add_argument('--test_noise_level', default=0.0, type=float)

    # Model variables
    parser.add_argument('--time_horizon', default=10, type=int)
    parser.add_argument('--nb_units', default=200, type=int)
    parser.add_argument('--nb_layers', default=2, type=int)
    parser.add_argument('--activation', default='tanh', type=str)
    parser.add_argument('--ls_cond', default=[0.0, 10.0])
    parser.add_argument('--ls_color', default=['red', 'seagreen'])
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--verbose', default=0, type=int)

    return parser.parse_args()


def run(args):
    plt.rc('font', size=10)
    plt.rc('axes', labelsize=12)
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['figure.figsize'] = 4.5, 4

    save_dir = os.path.join(this_dir, args.save_folder)
    os.makedirs(save_dir, exist_ok=True)
    print('save directory:', save_dir)

    for i in range(len(args.ls_cond)):
        lambda_trs, col = args.ls_cond[i], args.ls_color[i]
        split_name = 'ode_' + str(lambda_trs)
        print('experimental condition:', split_name)

        ts, xs = get_dataset(nb_samples=args.train_nb_sample, nb_timestep=args.train_nb_timestep,
                             t_span=args.train_t_span, seed=args.train_seed, noise_level=args.train_noise_level)
        t_train, x_train, y_train = reshape_data(ts=ts, xs=xs, substep=args.time_horizon)

        oden = ODENetwork(time_horizon=args.time_horizon, nb_units=args.nb_units, nb_layers=args.nb_layers,
                          activation=args.activation, lambda_trs=lambda_trs, learning_rate=args.learning_rate)

        oden.solver().fit(x=[t_train, x_train], y=y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)
        oden.func.save_weights(os.path.join(save_dir, split_name + '_' + 'weight.h5'))

        ts, xs = get_dataset(nb_samples=args.test_nb_sample, nb_timestep=args.test_nb_timestep,
                             t_span=args.test_t_span, seed=args.test_seed, noise_level=args.test_noise_level)
        t_test, x_test, y_test = ts, xs[:, 0, :], xs[:, 1:, :]

        y_pred = runge_kutta_solver(ts=t_test, x0=x_test, func=oden.func, time_augment=False)

        x1_true, x2_true, x3_true = y_test[:, :, 0], y_test[:, :, 1], y_test[:, :, 2]
        x1_pred, x2_pred, x3_pred = y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2]

        mse_t_per_sample = np.mean(np.mean(np.square(y_test - y_pred), axis=-1), axis=-1)
        mse_t, std_t = 1e2 * np.mean(mse_t_per_sample), 1e2 * np.std(mse_t_per_sample)

        print('trajectory MSE: {:.2f} pm {:.2f}'.format(mse_t, std_t))

        result = {'ground_truth': y_test, 'predicted': y_pred, 'mse_t': mse_t, 'std_t': std_t}

        with open(os.path.join(save_dir, split_name + '_' + 'result.pkl'), 'wb') as f:
            pkl.dump(result, f)

        plt.plot(x1_true[:20, :].T, x2_true[:20, :].T, 'k', label='Ground truth')
        plt.plot(x1_pred[:20, :].T, x2_pred[:20, :].T, col, label=split_name)
        plt.grid()
        plt.legend([Line2D([0], [0], color='k', lw=3), Line2D([0], [0], color=col, lw=3)], ['Ground truth', split_name])
        plt.xlabel('x'), plt.ylabel('y')
        plt.xlim([-3, 3]), plt.ylim([-4, 3])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, split_name + '_' + 'xy.png'))
        plt.close()

        plt.plot(x1_true[:20, :].T, x3_true[:20, :].T, 'k', label='Ground truth')
        plt.plot(x1_pred[:20, :].T, x3_pred[:20, :].T, col, label=split_name)
        plt.grid()
        plt.legend([Line2D([0], [0], color='k', lw=3), Line2D([0], [0], color=col, lw=3)], ['Ground truth', split_name])
        plt.xlabel('x'), plt.ylabel('z')
        plt.xlim([-3, 3]), plt.ylim([0, 10])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, split_name + '_' + 'xz.png'))
        plt.close()

        plt.plot(x2_true[:20, :].T, x3_true[:20, :].T, 'k', label='Ground truth')
        plt.plot(x2_pred[:20, :].T, x3_pred[:20, :].T, col, label=split_name)
        plt.grid()
        plt.legend([Line2D([0], [0], color='k', lw=3), Line2D([0], [0], color=col, lw=3)], ['Ground truth', split_name])
        plt.xlabel('y'), plt.ylabel('z')
        plt.xlim([-4, 3]), plt.ylim([0, 10])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, split_name + '_' + 'yz.png'))
        plt.close()

        ls_y_test_eps = []
        dist_init = 1e-5

        for i in range(len(x_test)):
            _, y = get_trajectory(t_span=args.test_t_span, timestep=(1 + args.test_nb_timestep), seed=0,
                                  noise_level=args.test_noise_level, y0=x_test[i] + np.array([0, 0, dist_init]))
            ls_y_test_eps.append(y[1:, :])
        y_test_eps = np.stack(ls_y_test_eps, axis=0)
        dist_true = np.sqrt(np.sum(np.square(y_test - y_test_eps), axis=-1))
        le_true = np.mean(np.log(dist_true / dist_init) / t_test[:, 1:, 0], axis=0)

        y_pred_eps = runge_kutta_solver(ts=t_test, x0=x_test + np.repeat(np.array([[0, 0, dist_init]]), repeats=x_test.shape[0], axis=0),
                                        func=oden.func, time_augment=False)
        dist_pred = np.sqrt(np.sum(np.square(y_pred - y_pred_eps), axis=-1))
        le_pred = np.mean(np.log(dist_pred / dist_init) / t_test[:, 1:, 0], axis=0)

        plt.plot(t_test[0, 1:].squeeze(), le_true, 'k', label='Ground truth', linewidth=3)
        plt.plot(t_test[0, 1:].squeeze(), le_pred, col, linestyle='dashed', label=split_name, linewidth=3)
        plt.grid(), plt.legend()
        plt.xlabel('Time'), plt.ylabel('Lyapunov exponent')
        plt.xlim([0, 20]), plt.ylim([-0.65, 0.15])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, split_name + '_' + 'lyapunov.png'))
        plt.close()


if __name__ == '__main__':
    run(get_args())
