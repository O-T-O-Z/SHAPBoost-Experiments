import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import Booster, XGBRegressor


class XGBSurvivalRegressor(XGBRegressor):
    """XGBoost Survival Regressor."""

    def __init__(
        self,
        objective: str = "survival:aft",
        eval_metric: str = "aft-nloglik",
        **kwargs
    ) -> None:
        """
        Create a new XGBSurvivalRegressor.

        :param objective: for the XGBRegressor, accelerated failure time (AFT) model.
        :param eval_metric: evaluation metric for the XGBRegressor.
        """
        super().__init__(objective=objective, eval_metric=eval_metric, **kwargs)
        self.__kwargs = kwargs
        self.__kwargs["objective"] = objective
        self.__kwargs["eval_metric"] = eval_metric

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        **kwargs
    ) -> Booster:
        """
        Fit the XGBSurvivalRegressor.

        :param X: training data.
        :param y: training labels.
        :param sample_weight: weight per sample.
        :return: trained model.
        """
        censored = y[:, 1] == np.inf
        if sample_weight is not None:
            (
                X_train,
                X_val,
                sample_weight_train,
                sample_weight_val,
                y_train,
                y_val,
            ) = train_test_split(
                X, sample_weight, y, test_size=0.2, random_state=42, stratify=censored
            )
            dmatrix = xgb.DMatrix(X_train, weight=sample_weight_train)
            dmatrix_val = xgb.DMatrix(X_val, weight=sample_weight_val)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=censored
            )
            dmatrix = xgb.DMatrix(X_train)
            dmatrix_val = xgb.DMatrix(X_val)
        if self.__kwargs["objective"] == "survival:cox":
            y_cox_train = np.array(
                [
                    y_lower if y_lower == y_train[:, 1][i] else -y_lower
                    for i, y_lower in enumerate(y_train[:, 0])
                ]
            )
            y_cox_val = np.array(
                [
                    y_lower if y_lower == y_val[:, 1][i] else -y_lower
                    for i, y_lower in enumerate(y_val[:, 0])
                ]
            )
            dmatrix = xgb.DMatrix(X_train, label=y_cox_train)
            dmatrix_val = xgb.DMatrix(X_val, label=y_cox_val)
        else:
            dmatrix.set_float_info("label_lower_bound", y_train[:, 0])
            dmatrix.set_float_info("label_upper_bound", y_train[:, 1])

            dmatrix_val.set_float_info("label_lower_bound", y_val[:, 0])
            dmatrix_val.set_float_info("label_upper_bound", y_val[:, 1])

        self._model = xgb.train(
            self.__kwargs,
            dmatrix,
            num_boost_round=1000,
            early_stopping_rounds=10,
            evals=[(dmatrix, "train"), (dmatrix_val, "val")],
            verbose_eval=False,
            **kwargs,
        )
        return self._model

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict the labels for the data.

        :param X: data to predict.
        :return: predicted labels.
        """
        return self._model.predict(xgb.DMatrix(X), **kwargs)

    def get_booster(self) -> Booster:
        """
        Get the underlying booster of the model.

        :return: Booster.
        """
        return self._model
