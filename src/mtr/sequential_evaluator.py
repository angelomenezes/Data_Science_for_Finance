import numpy as np
from mtr.stacked_single_target import StackedSingleTarget
from mtr.file_stream import FileStream


class SequentialEvaluator:
    class PortifolioStreamFormatter:
        def __init__(self, file_stream, n_actives, init_buffer_size,
                     max_buffer_size, batch_size, month_info='exact'):
            self.file_stream = file_stream
            self.n_actives = n_actives
            self.init_buffer_size = init_buffer_size
            self.max_buffer_size = max_buffer_size
            self.batch_size = batch_size
            self.buffer = self.file_stream.get_batch(self.init_buffer_size)
            self.first_predicted_date = self.init_buffer_size + 1
            self.buffer_matrix = None
            self.month_info = month_info

        def update_buffer(self):
            if not self.file_stream.has_next():
                return None
            sample = self.file_stream.next()
            self.buffer.extend(sample)
            if self.max_buffer_size is not None and \
               len(self.buffer) > self.max_buffer_size:
                while len(self.buffer) > self.max_buffer_size:
                    self.buffer.pop(0)

            return sample

        def get_training_space(self):
            self.buffer_matrix = np.row_stack(self.buffer)
            training_data = np.zeros(
                (len(self.buffer_matrix) - 241, 32 * self.n_actives)
            )
            for i in range(len(self.buffer_matrix) - 241):
                training_data[i, :(20 * self.n_actives)] = \
                    self.buffer_matrix[(i+220):(i+240), :].flatten()
                if self.month_info == 'exact':
                    for j in range(11):
                        training_data[
                          i, ((20+j)*self.n_actives):((20+j+1)*self.n_actives)
                        ] = self.buffer_matrix[i + 20 * j, :]
                else:
                    # Aggregate month days using mean
                    for j in range(11):
                        training_data[
                          i, ((20+j)*self.n_actives):((21+j)*self.n_actives)
                        ] = np.mean(self.buffer_matrix[
                            (i+20*j):(i+20*(j+1)), :
                        ], axis=0)
                training_data[i, -self.n_actives:] = \
                    self.buffer_matrix[i + 240, :]
            return training_data

        def transform_sample(self, sample):
            temp_buffer = []
            temp_buffer.extend(self.buffer)
            temp_buffer.extend(sample)
            temp_buffer = np.row_stack(temp_buffer)
            test_sample = np.zeros((self.batch_size, 31 * self.n_actives))
            k = 0
            for i in range(
                len(temp_buffer) - 240 * self.batch_size,
                len(temp_buffer) - 240
            ):
                test_sample[k, :(20 * self.n_actives)] = \
                    temp_buffer[(i+220):(i+240), :].flatten()
                if self.month_info == 'exact':
                    for j in range(11):
                        test_sample[
                          k, ((20+j)*self.n_actives):((20+j+1)*self.n_actives)
                        ] = temp_buffer[i + 20 * j, :]
                else:
                    # Aggregate month days using mean
                    for j in range(11):
                        test_sample[
                          k, ((20+j)*self.n_actives):((21+j)*self.n_actives)
                        ] = np.mean(temp_buffer[
                            (i+20*j):(i+20*(j+1)), :
                        ], axis=0)
            return test_sample

        def get_current_date(self):
            return self.file_stream.current_date()

        def predicted_dates(self):
            return self.file_stream.get_all_dates()[self.first_predicted_date:]

        def actives_names(self):
            return self.file_stream.get_column_names()

    def __init__(self, file_stream, n_actives, regressor, regressor_params,
                 init_buffer_size=300, max_buffer_size=400, batch_size=1,
                 month_info='exact'):
        self.n_actives = n_actives
        self._regressor = regressor
        self._regressor_params = regressor_params
        self._psf = SequentialEvaluator.PortifolioStreamFormatter(
            file_stream, n_actives, init_buffer_size, max_buffer_size,
            batch_size
        )
        self._sst = StackedSingleTarget(
            n_targets=self.n_actives,
            default_regressor=self._regressor,
            default_regressor_params=self._regressor_params,
        )
        self._predictions = []

    def fit_predict(self):
        while True:
            training_data = self._psf.get_training_space()
            # Train the MTR model
            self._sst.fit(
                training_data[:, :-self.n_actives],
                training_data[:, -self.n_actives:]
            )
            new_sample = self._psf.update_buffer()
            if new_sample is None:
                break
            new_sample = self._psf.transform_sample(new_sample)
            self._predictions.append(self._sst.predict(new_sample))
        final_predictions = np.row_stack(self._predictions)
        self._predictions = []
        return final_predictions

    def get_predicted_dates(self):
        dates = self._psf.predicted_dates()
        if isinstance(dates, list):
            return dates
        else:
            return dates.tolist()

    def get_actives_names(self):
        return self._psf.actives_names()
