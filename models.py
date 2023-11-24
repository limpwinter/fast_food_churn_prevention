from abc import ABC
from abc import abstractmethod
import numpy.typing as npt
import typing as tp
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


class BaseModel(ClassifierMixin):
    def __init__(self, *params: tp.Any):
        self.last_predicted_proba: tp.Optional[npt.NDArray[float]] = None
        self.last_class_predictions: tp.Optional[npt.NDArray[int]] = None

        self.last_float_predictions = None

        self.last_binary_labels: tp.Optional[npt.NDArray[int]] = None
        self.last_float_labels: tp.Optional[npt.NDArray[float]] = None
    def fit(self):

    def f_score(self) -> float:
        if self.last_binary_labels and self.last_class_predictions:
            print(
                f"calculate prediction for f1 score first.\n"
                f"{'self.last_binary_labels is None' if self.last_binary_labels is None else ''}"
                f"{'self.last_predicted_classes is None' if self.last_class_predictions is None else ''}"
            )
            return -1.0
        return f1_score(self.last_binary_labels, self.last_class_predictions)

    def rmse(self) -> float:
        return np.sqrt(
            mean_squared_error(
                self.last_float_labels,
                self.last_float_predictions
            )
        )

    def roc_auc(self) -> float:
        return roc_auc_score(
            self.last_binary_labels,
            self.last_predicted_proba[:, 1]
        )


class CatBoostModel(BaseModel):
    def __init__(self,
                 model: tp.Union[CatBoostRegressor, CatBoostClassifier],
                 *args: tp.Any
                 ):
        super().__init__(*args)
        self.model = model

    def predict_proba(self, data: npt.NDArray) -> npt.NDArray:
        return self.model.predict_proba(data)

    def fit(self, data: npt.NDArray, labels: npt.NDArray) -> "CatBoostModel":
        self.model.fit(data, labels)
        return self

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        return self.model.predict(data)


class RandomForestModel(BaseModel):
    def __init__(self,
                 model: tp.Union[RandomForestRegressor, RandomForestClassifier],
                 *args: tp.Any
                 ):
        super().__init__(*args)
        self.model = model

    def predict_proba(self, data: npt.NDArray) -> npt.NDArray:
        return self.model.predict_proba(data)

    def fit(self, data: npt.NDArray, labels: npt.NDArray) -> "RandomForestModel":
        self.model.fit(data, labels)
        return self

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        return self.model.predict(data)

class TwoLevelModel:
    def __init__(self, classifier, regressor):
        self.classifier:BaseModel = classifier
        self.regressor:BaseModel = regressor
    def fit(self, train_data, train_labels)->'TwoLevelModel':
        self.classifier.fit(X=train_data,y=)
        return self
    def predict(self):
