import numpy as np
from sklearn.model_selection import cross_val_predict
import pandas as pd


class StackedSingleTarget:
    """ Class that implements the Stacked Single-target_[1] MTR method.

    [1] Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., & Vlahavas, I.
    (2016). Multi-target regression via input space expansion: treating
    targets as inputs. Machine Learning, 104(1), 55-98.
    """
    INTERNAL_CV = 'internal_cv'
    PREDICTIONS = 'predictions'
    TARGETS_VALUES = 'targets_values'

    def __init__(self, n_targets, default_regressor=None,
                 default_regressor_params=None, method='predictions',
                 n_part=10):
        """ Constructs a SST model with the default configurations.

            Parameters
            ----------
            n_targets: int
                Task's number of targets.
            default_regressor: sklearn regressor function
                A model that will act as default configuration to SST, being
                employed in all targets for both base and meta models (unless)
                stated otherwise in the 'set_regressor' method.
            default_regressor_params: dict
                Parameters for the 'default_regressor'.
            method: str (Default: predictions)
                String defining the strategy to generate the augmented
                features for the meta models. Possible values are:
                - 'internal_cv': use a cross-validation approach with 'n_part'
                folds, to determine the base predictions (The 'n_part'
                parameter must be subjected in case of selecting this option).
                - 'predictions': use the predictions from the base models in
                in the training set to compose the augmented features.
                - 'targets_values': use the targets values in the training set
                as augmented features.
            n_part: int (Optional)
                Determines the number of internal cross-validation folds
                will be performed to generate the augmented features in case
                'method' is set to 'internal_cv'.
        """
        self.n_targets = n_targets
        self._regressors = {}
        self._init_regressors(default_regressor, default_regressor_params)

        if method == self.INTERNAL_CV:
            if n_part is None or not isinstance(n_part, int) or n_part <= 2:
                raise ValueError(
                    '"n_part" must be a interger greater than 2'
                )
            self._cv = n_part
        self.method = method

        if method not in [self.INTERNAL_CV, self.PREDICTIONS,
                          self.TARGETS_VALUES]:
            raise ValueError('Invalid "method" value (Options are: \
                             "internal_cv", "predictions", "targets_values").')

    def _init_regressors(self, default_regressor, default_regressor_params):
        for layer in range(2):
            for t in range(self.n_targets):
                self._regressors[(t, layer)] = {
                    'model': default_regressor,
                    'params': default_regressor_params
                }

    def set_regressor(self, target_index, regressor, regressor_params,
                      meta_model=False):
        """ Set the regressor for a given target and base/meta layer.

            Parameters
            ----------
                target_index: int
                    An integer denoting the index of the target.
                regressor: sklearn regressor model
                    The regressor function to be employed at the specific
                    target/layer combination. In this case, the layer can be
                    either base or meta.
                regressor_params: dict
                    A dictionary with configurations for the 'regressor'.
                meta_model: Boolean (Default: False)
                    Determines whether the passed regressor refers to the base
                    or meta model layers.
        """
        self._regressors[(target_index, int(meta_model))] = {
            'model': regressor,
            'params': regressor_params
        }

    def fit(self, X, Y):
        """ Trains the SST model for the given input and output examples.

            Parameters
            ----------
                X: pandas.DataFrame or numpy.ndarray
                    The input values.
                Y: pandas.DataFrame or numpy.ndarray
                    The targets values.
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise ValueError('"X" must be a numpy matrix or pandas \
                             DataFrame')
        if isinstance(Y, pd.core.frame.DataFrame):
            Y = Y.values
        elif not isinstance(Y, np.ndarray):
            raise ValueError('"Y" must be a numpy matrix or pandas \
                             DataFrame')

        self._base_models = {}
        self._meta_models = {}

        # Training the base models
        for t in range(self.n_targets):
            self._base_models[t] = self._regressors[(t, 0)]['model'](
                **self._regressors[(t, 0)]['params']
            )
            self._base_models[t].fit(X, Y[:, t])

        base_predictions = np.zeros_like(Y)

        # Getting the base predictions
        if self.method == self.INTERNAL_CV:
            for t in range(self.n_targets):
                base_predictions[:, t] = cross_val_predict(
                    self._regressors[(t, 0)]['model'](
                        **self._regressors[(t, 0)]['params']
                    ),
                    X,
                    Y[:, t],
                    cv=self._cv
                )
        elif self.method == self.PREDICTIONS:
            for t in range(self.n_targets):
                base_predictions[:, t] = self._base_models[t].predict(X)
        elif self.method == self.TARGETS_VALUES:
            base_predictions = Y.copy()

        # Augmented training set
        X_aug = np.column_stack((X, base_predictions))

        # Training the meta_models
        for t in range(self.n_targets):
            self._meta_models[t] = self._regressors[(t, 1)]['model'](
                **self._regressors[(t, 1)]['params']
            )
            self._meta_models[t].fit(X_aug, Y[:, t])

    def predict(self, X):
        """ Predicts new instances using the trained SST model.

            Parameters
            ----------
                X: pandas.DataFrame or numpy.ndarray
                    The new input values.
            Returns
            -------
                predictions: numpy.ndarray
                    A matrix with the same number of lines of 'X', where each
                    column represents a target.
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise ValueError('"X" must be a numpy matrix or pandas \
                             DataFrame')

        n_rows = X.shape[0]
        base_predictions = np.zeros((n_rows, self.n_targets))
        meta_predictions = np.zeros((n_rows, self.n_targets))

        for t in range(self.n_targets):
            base_predictions[:, t] = self._base_models[t].predict(X)

        X_aug = np.column_stack((X, base_predictions))

        for t in range(self.n_targets):
            meta_predictions[:, t] = self._meta_models[t].predict(X_aug)

        return meta_predictions
