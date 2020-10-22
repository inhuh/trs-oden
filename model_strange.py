import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense
from keras.layers import Add
from keras.optimizers import Adam


def time_difference(x):
    return x[:, 1:] - x[:, :-1]


def reversing_operator(x, with_batch):
    if with_batch:
        x1, x2, x3 = -x[:, :, 0:1], -x[:, :, 1:2], -x[:, :, 2:3]
        return K.concatenate(tensors=[x1, x2, x3], axis=-1)
    else:
        x1, x2, x3 = -x[:, 0:1], -x[:, 1:2], -x[:, 2:3]
        return K.concatenate(tensors=[x1, x2, x3], axis=-1)


class ODENetwork:
    def __init__(self, time_horizon=10, nb_units=100, nb_layers=2, activation='relu', lambda_trs=0.0, learning_rate=2e-4):
        self.T = time_horizon
        self.units, self.layers = nb_units, nb_layers
        self.act = activation
        self.lambda_trs = lambda_trs
        self.lr = learning_rate
        self.func = self.ode_function()

    def ode_function(self):
        x = Input(shape=(3,))
        h = Dense(units=self.units, activation=self.act)(x)
        for _ in range((self.layers - 1)):
            h = Dense(units=self.units, activation=self.act)(h)
        y = Dense(units=3, use_bias=False)(h)
        return Model(inputs=x, outputs=y)

    def solver(self):
        def l_ode(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))

        def l_trs(y_true, y_pred):
            return K.mean(K.square(reversing_operator(x=X, with_batch=True) - Xr))

        def l_trsoden(y_true, y_pred):
            return l_ode(y_true, y_pred) + self.lambda_trs * l_trs(y_true, y_pred)

        ts = Input(shape=(self.T + 1, 1))
        dts = Lambda(function=time_difference)(ts)

        x0 = Input(shape=(3,))
        x = x0
        xr = Lambda(function=reversing_operator, arguments={'with_batch': False})(x)

        ls_x, ls_xr = [], []

        for i in range(self.T):
            dt = Lambda(lambda f: f[:, i, :])(dts)
            # We used Runge-Kutta 4th solver for the strange attractor experiment.
            # Forward time evolution
            dx1 = Lambda(lambda f: f * dt)(self.func(x))
            dx2 = Lambda(lambda f: f * dt)(self.func(Add()([x, Lambda(lambda f: 0.5 * f)(dx1)])))
            dx3 = Lambda(lambda f: f * dt)(self.func(Add()([x, Lambda(lambda f: 0.5 * f)(dx2)])))
            dx4 = Lambda(lambda f: f * dt)(self.func(Add()([x, dx3])))
            dx = Lambda(lambda f: (1 / 6) * f)(Add()([dx1, Lambda(lambda f: 2 * f)(dx2), Lambda(lambda f: 2 * f)(dx3), dx4]))
            x = Add()([x, dx])
            ls_x.append(x)

            # Backward time evolution
            dxr1 = Lambda(lambda f: -f * dt)(self.func(xr))
            dxr2 = Lambda(lambda f: -f * dt)(self.func(Add()([xr, Lambda(lambda f: 0.5 * f)(dxr1)])))
            dxr3 = Lambda(lambda f: -f * dt)(self.func(Add()([xr, Lambda(lambda f: 0.5 * f)(dxr2)])))
            dxr4 = Lambda(lambda f: -f * dt)(self.func(Add()([xr, dxr3])))
            dxr = Lambda(lambda f: (1 / 6) * f)(Add()([dxr1, Lambda(lambda f: 2 * f)(dxr2), Lambda(lambda f: 2 * f)(dxr3), dxr4]))
            xr = Add()([xr, dxr])
            ls_xr.append(xr)

        X = Lambda(lambda f: K.stack(f, axis=1))(ls_x)
        Xr = Lambda(lambda f: K.stack(f, axis=1))(ls_xr)

        model = Model(inputs=[ts, x0], outputs=X)
        model.compile(optimizer=Adam(lr=self.lr), loss=l_trsoden, metrics=[l_ode, l_trs])
        return model
