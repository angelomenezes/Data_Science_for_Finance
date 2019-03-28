import numpy as np
import pandas as pd
from stacked_single_target import StackedSingleTarget
from sklearn.preprocessing import StandardScaler


class MTRTimeSeries:
    def __init__(self, time_serie, window_size=20, horizon_size=5,
                 init_buffer_size=200, max_buffer_size=None,
                 use_exp_smoothing=False, gamma=0.1, sst_method='predictions'):

        self.init_buffer_size = init_buffer_size
        if max_buffer_size is None:
            self.max_buffer_size = self.init_buffer_size
        else:
            self.max_buffer_size = max_buffer_size

        self._time_serie = time_serie
        self.window_size = window_size
        self.horizon_size = horizon_size
        self.use_exp_smoothing = use_exp_smoothing
        self.gamma = gamma
        self.sst_method = sst_method

    def set_regressor(self, regressor, regressor_params):
        self._regressor = regressor
        self._regressor_params = regressor_params

    def iterate_fit_predict(self):
        s_idx = 0
        ml_observations = {}
        ema = self._time_serie[s_idx]
        while s_idx < self.init_buffer_size:
            incr = (s_idx + self.window_size + self.horizon_size)
            if self.use_exp_smoothing:
                aux_smooth = []
                for s in self._time_serie[s_idx:incr]:
                    ema = self.gamma * s + (1 - self.gamma) * ema
                    aux_smooth.append(ema)
                ml_observations[s_idx] = aux_smooth
            else:
                ml_observations[s_idx] = self._time_serie[s_idx:incr]
            s_idx += 1

        i = self.init_buffer_size + self.horizon_size
        stop_point = len(self._time_serie) - \
            (self.window_size + self.horizon_size)

        predictions = []
        while i < stop_point:
            data = pd.DataFrame.from_dict(
                ml_observations, orient='index'
            )
            sst = StackedSingleTarget(
                n_targets=self.horizon_size,
                default_regressor=self._regressor,
                default_regressor_params=self._regressor_params,
                method=self.sst_method
            )
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_x.fit_transform(
                data.iloc[:, :-self.horizon_size].values
            )
            Y = scaler_y.fit_transform(
                data.iloc[:, -self.horizon_size:].values
            )
            sst.fit(X, Y)
            if self.use_exp_smoothing:
                aux_ema = ema
                aux_smooth = []
                for s in self._time_serie[i:(i+self.window_size)]:
                    aux_ema = self.gamma * s + (1 - self.gamma) * aux_ema
                    aux_smooth.append(ema)
                new_x = np.array(aux_smooth)
            else:
                new_x = np.array(self._time_serie[i:(i+self.window_size)])
            new_x = new_x[None, :]
            new_x = scaler_x.transform(new_x)
            predictions.extend(
                scaler_y.inverse_transform(sst.predict(new_x)).tolist()[0]
            )

            old_idx = s_idx
            if s_idx >= self.max_buffer_size:
                s_idx = 0
                old_idx -= 1
            else:
                s_idx += 1

            if self.use_exp_smoothing:
                aux_smooth = ml_observations[old_idx][self.horizon_size:]
                for s in self._time_serie[
                    (i + self.window_size):
                    (i + self.window_size + self.horizon_size)
                ]:
                    ema = self.gamma * s + (1 - self.gamma) * ema
                    aux_smooth.append(ema)
                ml_observations[s_idx] = aux_smooth
            else:
                ml_observations[s_idx] = self._time_serie[
                    i:(i + self.window_size + self.horizon_size)
                ]
            i += self.horizon_size
        return predictions

    def init_prediction_index(self):
        return self.init_buffer_size + self.window_size
