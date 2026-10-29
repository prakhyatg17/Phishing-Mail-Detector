"""
ensemble.py – SoftVotingEnsemble that averages predict_proba of
already-fitted models.  Kept in its own module so it's picklable.
"""

import numpy as np


class SoftVotingEnsemble:
    """
    Simple soft-voting ensemble: averages the predict_proba outputs of
    a list of pre-fitted classifiers with equal weights.
    Threshold = 0.5 for binary classification.
    """

    def __init__(self, estimators: list | None = None):
        """
        Parameters
        ----------
        estimators : list of (name, fitted_model) tuples
        """
        self.estimators = estimators or []

    def predict_proba(self, X):
        """Average the predict_proba matrices of all estimators."""
        probas = np.array([est.predict_proba(X) for _, est in self.estimators])
        return probas.mean(axis=0)

    def predict(self, X):
        avg = self.predict_proba(X)
        return (avg[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"estimators": self.estimators}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

