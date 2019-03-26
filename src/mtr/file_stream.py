import pandas as pd


class FileStream:
    def __init__(self, data_path=None, batch_size=1, remove_first_column=False, data=None):
        if data_path is None:
            self._data = data
        else:
            self._data = pd.read_csv(data_path)
        self._date = None
        if remove_first_column:
            self._date = self._data.iloc[:, 0]
            self._data = self._data.iloc[:, 1:]
        if self._date is None:
            self._date = [i for i in range(len(self._data))]
        self._columns = list(self._data)
        self._data = self._data.values
        self.batch_size = batch_size
        self._current = 0

    def has_next(self):
        return self._current + self.batch_size < len(self._data)

    def next(self):
        if self._current + self.batch_size >= len(self._data):
            return None
        sample = []
        for i in range(self._current, self._current + self.batch_size):
            sample.append(self._data[i, :])
        self._current += self.batch_size
        return sample

    def get_batch(self, batch_size=240):
        if self._current + batch_size >= len(self._data):
            return None
        sample = []
        for i in range(self._current, self._current + batch_size):
            sample.append(self._data[i, :])
        self._current += batch_size
        return sample

    def current_date(self):
        if self._date is not None:
            return self._date[self._current]
        return None

    def get_all_dates(self):
        return self._date

    def get_column_names(self):
        return self._columns
