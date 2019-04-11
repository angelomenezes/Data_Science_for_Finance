import numpy as np
import math


class OnlinePerceptron:
    def __init__(self, learning_rate=0.01, perceptron_weight=None,
                 learning_rate_const=True, learning_rate_decay=0.001,
                 random_state=None):
        """LearningNodePerceptron class constructor

        Parameters
        ----------
        initial_class_observations
        perceptron_weight
        """
        self.learning_rate = learning_rate
        self.perceptron_weight = perceptron_weight
        self.learning_rate_const = learning_rate_const
        self.learning_rate_decay = learning_rate_decay
        self.random_state = random_state

        self.examples_seen = 0
        self.sum_of_attribute_values = 0.0
        self.sum_of_attribute_squares = 0.0

        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_dimensions(self, X):
        """ Return the dimensions from a numpy.array, numpy.ndarray or list.
        Parameters
        ----------
        X: numpy.array, numpy.ndarray, list, list of lists.
        Returns
        -------
        tuple
            A tuple representing the X structure's dimensions.
        """
        r, c = 1, 1
        if isinstance(X, type(np.array([0]))):
            if X.ndim > 1:
                r, c = X.shape
            else:
                r, c = 1, X.size

        elif isinstance(X, type([])):
            if isinstance(X[0], type([])):
                r, c = len(X), len(X[0])
            else:
                c = len(X)

        return r, c

    def _update_weights(self, X, y, learning_rate):
        """Update the perceptron weights

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
                Targets values.
        """
        normalized_sample = self._normalize_sample(X)
        normalized_pred = self._inner_predict(normalized_sample)

        normalized_target_value = self._normalize_target_value(y)

        self.perceptron_weight += learning_rate * \
            np.matmul(
                (normalized_target_value - normalized_pred)[:, None],
                normalized_sample[None, :]
            )

        self._normalize_perceptron_weights()

    def _normalize_perceptron_weights(self):
        # Normalize perceptron weights
        sum_w = np.sum(np.abs(self.perceptron_weight))
        self.perceptron_weight /= sum_w

    # Predicts new income instances as a multiplication of the neurons
    # weights with the inputs augmented with a bias value
    def _inner_predict(self, X):
        return np.matmul(self.perceptron_weight, X)

    def _normalize_sample(self, X):
        """Normalize the features in order to have the same influence during the
        process of training.
        Parameters
        ----------
        X: np.array
            features.
        Returns
        -------
        np.array:
            normalized samples
        """
        if self.examples_seen <= 1:
            aux = X.tolist()
            aux.append(1.0)
            return np.array(aux)

        mean = self.sum_of_attribute_values / self.examples_seen
        variance = (self.sum_of_attribute_squares -
                    (self.sum_of_attribute_values ** 2) /
                    self.examples_seen) / (self.examples_seen - 1)

        sd = np.sqrt(variance, out=np.zeros_like(variance),
                     where=variance >= 0.0)

        normalized_sample = np.zeros(X.shape[0] + 1, dtype=np.float64)
        np.divide(X - mean, sd, where=sd != 0, out=normalized_sample[:-1])
        # Augments sample with the bias input signal (or y intercept for
        # each target)
        normalized_sample[-1] = 1.0

        return normalized_sample

    def _normalize_target_value(self, y):
        """Normalize the targets in order to have the same influence during the
        process of training.
        Parameters
        ----------
        y: np.array
            targets.
        Returns
        -------
        np.array:
            normalized targets values
        """
        if self.examples_seen <= 1:
            return y

        mean = self.sum_of_values / self.examples_seen
        variance = (self.sum_of_squares -
                    (self.sum_of_values ** 2) /
                    self.examples_seen) / (self.examples_seen - 1)

        sd = math.sqrt(variance) if variance > 0 else 1

        normalized_target = (y - mean)/sd if variance > 0 else 0

        return normalized_target

    def _denormalize_target(self, y):
        if self.examples_seen <= 1:
            return y

        mean = self.sum_of_values / self.examples_seen
        variance = (self.sum_of_squares -
                    (self.sum_of_values ** 2) /
                    self.examples_seen) / (self.examples_seen - 1)

        sd = math.sqrt(variance) if variance > 0 else 1

        normalized_target = y * sd + mean if variance > 0 else y

        return normalized_target

    def partial_fit(self, X, y, weight=1.0):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
                Instance targets.
        weight: float
                Instance weight.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.examples_seen += weight
        self.sum_of_attribute_values += weight * X
        self.sum_of_attribute_squares += weight * X ** 2

        self.sum_of_values += weight * y
        self.sum_of_squares += weight * y ** 2
        if self.perceptron_weight is None:
            # Creates matrix of perceptron random weights
            _, cols = self._get_dimensions(X)

            self.perceptron_weight = np.random.uniform(
                -1.0, 1.0, (1, cols + 1)
            )
            self._normalize_perceptron_weights()

        if self.learning_rate_const:
            learning_rate = self.learning_rate
        else:
            learning_rate = self.learning_rate / \
                (1 + self.examples_seen * self.learning_rate_decay)

        for i in range(int(weight)):
            self._update_weights(X, y, learning_rate)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_new = self._normalize_sample(X)
        n_pred = np.matmul(self.perceptron_weight, X_new)
        return self._denormalize_target(n_pred)
