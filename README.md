
# Time-Reversal Symmetric ODE Network (TRS-ODEN)

In Huh, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin | NeurIPS 2020

## Basic usage

To reproduce the experiments in the paper:
- Experiment I (simple oscillators): `python3 run_exp1.py`
- Experiment II (non-linear oscillators): `python3 run_exp2.py`
- Experiment III (forced oscillators): `python3 run_exp3.py`
- Experiment IV (damped oscillators): `python3 run_exp4.py`
- Experiment V (real-world double oscillators): `python3 run_exp5.py`
- Experiment VI (strange attractors): `python3 run_exp6.py`

To train the models with arbitrary settings:
- with synthetic Duffing oscillator dataset: `python3 train_duffing.py`
- with real-world oscillator dataset: `python3 train_real.py`
- with strange attractor dataset: `python3 train_strange.py`

## Dependencies

- Python 3.6
- Numpy 1.13.3
- Scipy 1.0.0
- Tensorflow 1.12.0
- Keras 2.2.4
