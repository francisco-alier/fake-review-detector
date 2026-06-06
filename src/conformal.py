import numpy as np


class ConformalClassifier:
    """
    Inductive (split) Conformal Prediction wrapper for binary classifiers.
    Guarantees that the true label is in the prediction set with probability 1 - alpha.
    """

    def __init__(self, estimator, alpha: float = 0.05):
        self.estimator = estimator
        self.alpha = alpha
        self.q_hat = None

    def fit_calibration(self, X_calib, y_calib):
        """
        Calibrates the conformal threshold using a calibration dataset.
        y_calib must contain ground truth labels (0 or 1).
        """
        # Get class probabilities: shape (n_samples, 2)
        probs = self.estimator.predict_proba(X_calib)
        n = len(y_calib)

        # Non-conformity score: s_i = 1 - P(y_i | x_i)
        # We extract the probability corresponding to the true label for each row
        true_probs = probs[np.arange(n), y_calib]
        scores = 1.0 - true_probs

        # Calculate the conformal quantile value (1 - alpha) * (1 + 1/n)
        quantile_target = (1.0 - self.alpha) * (1.0 + 1.0 / n)
        # Ensure target is bounded by [0, 1]
        quantile_target = min(max(quantile_target, 0.0), 1.0)

        # q_hat is the (1-alpha)(1 + 1/n) empirical quantile of the scores
        self.q_hat = np.quantile(scores, quantile_target, method="higher")
        return self

    def predict_set(self, X_new):
        """
        Generates prediction sets and returns them alongside the raw probabilities.
        Prediction sets contain elements from {0, 1}.
        """
        if self.q_hat is None:
            raise ValueError("ConformalClassifier has not been calibrated. Run fit_calibration first.")

        # Get probabilities: shape (m_samples, 2)
        probs = self.estimator.predict_proba(X_new)

        # Condition for class membership: 1 - P(c | x) <= q_hat  ==>  P(c | x) >= 1 - q_hat
        threshold = 1.0 - self.q_hat

        prediction_sets = []
        for p in probs:
            p_set = []
            if p[0] >= threshold:
                p_set.append(0)
            if p[1] >= threshold:
                p_set.append(1)

            # Safeguard: if prediction set is empty, return the argmax (highest probability class)
            if len(p_set) == 0:
                p_set = [int(np.argmax(p))]
            prediction_sets.append(p_set)

        return prediction_sets, probs
