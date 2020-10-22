import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Layer
from keras.layers import Dense
from keras.layers import Add, Concatenate
from keras.optimizers import Adam


class Gradient(Layer):
    def call(self, inputs, **kwargs):
        x, y = inputs
        return K.gradients(loss=y, variables=x)[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def time_difference(x):
    return x[:, 1:] - x[:, :-1]


def reversing_operator(x, dim, with_batch):
    if with_batch:
        q = x[:, :, :dim]
        p = -x[:, :, dim:]
        return K.concatenate(tensors=[q, p], axis=-1)
    else:
        q = x[:, :dim]
        p = -x[:, dim:]
        return K.concatenate(tensors=[q, p], axis=-1)


class ODENetwork:
    def __init__(self, nb_object=1, nb_coords=1, function_type='ode', time_horizon=10, time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh',
                 lambda_trs=0.0, use_time_dep_lambda=False, learning_rate=2e-4):
        self.dim = int(nb_object * nb_coords)
        self.T = time_horizon
        self.augment = time_augment
        self.units, self.layers = nb_units, nb_layers
        self.act = activation
        self.lambda_trs = lambda_trs
        self.t_dep = use_time_dep_lambda
        self.lr = learning_rate
        if function_type == 'ode':
            self.func = self.ode_function()
        elif function_type == 'hamiltonian':
            self.func = self.hamilton_equation()
        else:
            raise NotImplementedError

    def ode_function(self):
        x = Input(shape=(int(2 * self.dim),))
        if self.augment:
            t = Input(shape=(1,))
            h = Dense(units=self.units, activation=self.act)(Concatenate(axis=-1)([x, t]))
        else:
            h = Dense(units=self.units, activation=self.act)(x)
        for _ in range((self.layers - 1)):
            h = Dense(units=self.units, activation=self.act)(h)
        y = Dense(units=int(2 * self.dim), use_bias=False)(h)
        if self.augment:
            return Model(inputs=[t, x], outputs=y)
        else:
            return Model(inputs=x, outputs=y)

    def hamilton_equation(self):
        x = Input(shape=(int(2 * self.dim),))
        q, p = Lambda(lambda f: f[:, :self.dim])(x), Lambda(lambda f: f[:, self.dim:])(x)
        if self.augment:
            t = Input(shape=(1,))
            v = Dense(units=self.units, activation=self.act)(Concatenate(axis=-1)([q, t]))
            k = Dense(units=self.units, activation=self.act)(Concatenate(axis=-1)([p, t]))
        else:
            v = Dense(units=self.units, activation=self.act)(q)
            k = Dense(units=self.units, activation=self.act)(p)
        for _ in range((self.layers - 1)):
            v = Dense(units=self.units, activation=self.act)(v)
            k = Dense(units=self.units, activation=self.act)(k)
        v = Dense(units=1, use_bias=False)(v)
        k = Dense(units=1, use_bias=False)(k)
        dq = Gradient()([p, k])
        dp = Lambda(lambda f: -1 * f)(Gradient()([q, v]))
        if self.augment:
            return Model(inputs=[t, x], outputs=Concatenate(axis=-1)([dq, dp]))
        else:
            return Model(inputs=x, outputs=Concatenate(axis=-1)([dq, dp]))

    def solver(self):
        def l_ode(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))

        def l_trs(y_true, y_pred):
            if self.t_dep:
                norm_ts = ts / K.max(ts)
                return K.mean(norm_ts[:, 1:, 0] * K.mean(K.square(reversing_operator(x=X, dim=self.dim, with_batch=True) - Xr), axis=-1))
            else:
                return K.mean(K.square(reversing_operator(x=X, dim=self.dim, with_batch=True) - Xr))

        def l_trsoden(y_true, y_pred):
            return l_ode(y_true, y_pred) + self.lambda_trs * l_trs(y_true, y_pred)

        ts = Input(shape=(self.T + 1, 1))
        dts = Lambda(function=time_difference)(ts)

        x0 = Input(shape=(int(2 * self.dim),))
        x = x0
        xr = Lambda(function=reversing_operator, arguments={'dim': self.dim, 'with_batch': False})(x)

        ls_x, ls_xr = [], []

        for i in range(self.T):
            t = Lambda(lambda f: f[:, i, :])(ts)
            tr = Lambda(lambda f: -f)(t)
            dt = Lambda(lambda f: f[:, i, :])(dts)

            if self.augment:  # For time-augmented (non-autonomous) cases, we used Runge-Kutta 4th solver.
                # Forward time evolution
                dx1 = Lambda(lambda f: f*dt)(self.func([t, x]))
                dx2 = Lambda(lambda f: f*dt)(self.func([Add()([t, Lambda(lambda f: 0.5*f)(dt)]), Add()([x, Lambda(lambda f: 0.5*f)(dx1)])]))
                dx3 = Lambda(lambda f: f*dt)(self.func([Add()([t, Lambda(lambda f: 0.5*f)(dt)]), Add()([x, Lambda(lambda f: 0.5*f)(dx2)])]))
                dx4 = Lambda(lambda f: f*dt)(self.func([Add()([t, dt]), Add()([x, dx3])]))
                dx = Lambda(lambda f: (1/6)*f)(Add()([dx1, Lambda(lambda f: 2*f)(dx2), Lambda(lambda f: 2*f)(dx3), dx4]))
                x = Add()([x, dx])
                ls_x.append(x)

                # Backward time evolution
                dxr1 = Lambda(lambda f: -f*dt)(self.func([tr, xr]))
                dxr2 = Lambda(lambda f: -f*dt)(self.func([Add()([tr, Lambda(lambda f: -0.5*f)(dt)]), Add()([xr, Lambda(lambda f: 0.5*f)(dxr1)])]))
                dxr3 = Lambda(lambda f: -f*dt)(self.func([Add()([tr, Lambda(lambda f: -0.5*f)(dt)]), Add()([xr, Lambda(lambda f: 0.5*f)(dxr2)])]))
                dxr4 = Lambda(lambda f: -f*dt)(self.func([Add()([tr, Lambda(lambda f: -f)(dt)]), Add()([xr, dxr3])]))
                dxr = Lambda(lambda f: (1/6)*f)(Add()([dxr1, Lambda(lambda f: 2*f)(dxr2), Lambda(lambda f: 2*f)(dxr3), dxr4]))
                xr = Add()([xr, dxr])
                ls_xr.append(xr)

            else:  # Leapfrog solver for autonomous systems
                # Forward time evolution
                q, p = Lambda(lambda f: f[:, :self.dim])(x), Lambda(lambda f: f[:, self.dim:])(x)
                p = Add()([p, Lambda(lambda f: (0.5*f*dt)[:, self.dim:])(self.func(x))])
                q = Add()([q, Lambda(lambda f: (1.0*f*dt)[:, :self.dim])(self.func(Concatenate(axis=-1)([q, p])))])
                p = Add()([p, Lambda(lambda f: (0.5*f*dt)[:, self.dim:])(self.func(Concatenate(axis=-1)([q, p])))])
                x = Concatenate(axis=-1)([q, p])
                ls_x.append(x)

                # Backward time evolution
                qr, pr = Lambda(lambda f: f[:, :self.dim])(xr), Lambda(lambda f: f[:, self.dim:])(xr)
                pr = Add()([pr, Lambda(lambda f: (-0.5*f*dt)[:, self.dim:])(self.func(xr))])
                qr = Add()([qr, Lambda(lambda f: (-1.0*f*dt)[:, :self.dim])(self.func(Concatenate(axis=-1)([qr, pr])))])
                pr = Add()([pr, Lambda(lambda f: (-0.5*f*dt)[:, self.dim:])(self.func(Concatenate(axis=-1)([qr, pr])))])
                xr = Concatenate(axis=-1)([qr, pr])
                ls_xr.append(xr)

        X = Lambda(lambda f: K.stack(f, axis=1))(ls_x)
        Xr = Lambda(lambda f: K.stack(f, axis=1))(ls_xr)

        model = Model(inputs=[ts, x0], outputs=X)
        model.compile(optimizer=Adam(lr=self.lr), loss=l_trsoden, metrics=[l_ode, l_trs])
        return model
