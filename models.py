from abc import ABC
from abc import abstractmethod
import numpy.typing as npt
import typing as tp
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class BaseModel(ABC):
    def __init__(self, *params: tp.Any):
        self.last_predicted_proba: tp.Optional[npt.NDArray[float]] = None
        self.last_binary_predictions: tp.Optional[npt.NDArray[int]] = None
        self.last_binary_labels: tp.Optional[npt.NDArray[int]] = None
        self.last_float_labels: tp.Optional[npt.NDArray[float]] = None
        self.last_float_predictions: tp.Optional[npt.NDArray[float]] = None

    @abstractmethod
    def fit(self, data: npt.NDArray, labels: npt.NDArray) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, data: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def predict_proba(self, data: npt.NDArray) -> npt.NDArray:
        pass

    def f_score(self) -> float:
        if self.last_binary_labels and self.last_binary_predictions:
            print(
                f"calculate prediction for f1 score first.\n"
                f"{'self.last_binary_labels is None' if self.last_binary_labels is None else ''}"
                f"{'self.last_predicted_classes is None' if self.last_binary_predictions is None else ''}"
            )
            return -1.0
        return f1_score(self.last_binary_labels, self.last_binary_predictions)

    def rmse(self) -> float:
        return np.sqrt(
            mean_squared_error(
                self.last_float_labels,
                self.last_float_predictions
            )
        )

    def roc_auc(self) -> float:
        raise NotImplementedError


class CatBoostModel(BaseModel):
    def __init__(self,
                 model: tp.Union[CatBoostRegressor, CatBoostClassifier],
                 *args: tp.Any
                 ):
        super().__init__(*args)
        self.model = model

    def predict_proba(self, data: npt.NDArray) -> npt.NDArray:
        self.last_predicted_proba = self.model.predict_proba(data)[:, 1]
        return self.last_predicted_proba

    def fit(self, data: npt.NDArray, labels: npt.NDArray) -> "CatBoostModel":
        self.model.fit(data, labels)
        return self

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        self.last_binary_predictions = self.model.predict(data)
        return self.last_binary_predictions


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
