import pandas as pd
import numpy as np
import typing as tp
from numpy.typing import NDArray
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score


class MetaInfo:
    # TODO rename
    def __init__(self, dataframe: pd.DataFrame, fill_churn: tp.Optional[int] = None):
        # if fill_churn is not None:
        #     dataframe['date_diff_post'] = dataframe['date_diff_post'].fillna(fill_churn)
        self.fill_churn = fill_churn if fill_churn else 61
        self.df = dataframe
        self.customers: NDArray[np.int64] = self.df['customer_id'].unique()
        self.grouped_by_customer = self.df.groupby('customer_id')

    def num_unique_places(self, col_name: str) -> pd.Series:
        return self.grouped_by_customer['ownareaall_sqm'].nunique().rename(col_name)

    def total_spend_money(self, col_name: str) -> pd.Series:
        return self.grouped_by_customer['revenue'].sum().rename(col_name)

    def avg_spend_money_per_order(self, col_name: str) -> pd.Series:
        return (
                self.grouped_by_customer['revenue'].sum() /
                self.grouped_by_customer['startdatetime'].nunique()
        ).rename(col_name)

    def num_orders(self, col_name: str) -> pd.Series:
        return self.grouped_by_customer['startdatetime'].nunique().rename(col_name)

    # def avg_discount_per_order(self, col_name: str) -> pd.Series:
    #     prices: pd.Series = self.df.groupby('dish_name')['revenue'].median()
    #     self.df

    def unique_days(self, col_name: str) -> pd.Series:
        self.df['startdatetime_day'] = self.df['startdatetime'].dt.floor('D')
        self.grouped_by_customer = self.df.groupby('customer_id')
        # del self.df['startdatetime_day']
        return self.df.groupby('customer_id')['startdatetime_day'].nunique().rename(col_name)

    def generate_meta_features(self) -> pd.DataFrame:
        meta_features = pd.DataFrame()
        meta_features['customer_id'] = self.customers
        meta_features = pd.merge(
            meta_features,
            self.num_unique_places("unique_places"), on='customer_id', how='left'
        )
        meta_features = pd.merge(
            meta_features,
            self.num_orders("orders_count"), on='customer_id', how='left'
        )
        meta_features = pd.merge(
            meta_features, self.total_spend_money("total_spend"), on='customer_id', how='left'
        )
        meta_features = pd.merge(
            meta_features,
            self.avg_spend_money_per_order("avg_bill"), on='customer_id', how='left'
        )
        meta_features = pd.merge(
            meta_features,
            self.unique_days("unique_days"), on='customer_id', how='left'
        )

        return meta_features \
            .set_index('customer_id') \
            .sort_index()

    def get_labels(self) -> pd.DataFrame:
        return self.df[['customer_id', 'date_diff_post', 'buy_post']] \
            .fillna(self.fill_churn) \
            .drop_duplicates() \
            .set_index('customer_id') \
            .sort_index()

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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


class Wrapper:
    def __init__(self, model, X_train, y_train, X_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.predictions = None
        self.metrics: list[str] = ["f1_macro", "f1", "roc_auc"]

    def crossval(self, cv=5):
        results = {}
        for metric in self.metrics:
            scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring=metric)
            mean_metric = np.mean(scores)
            results[metric] = mean_metric
        return results

    def fit(self):
        """Fit the model on the training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test data."""
        self.predictions = self.model.predict(self.X_test)
        return self.predictions
