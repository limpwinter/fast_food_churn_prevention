import pandas as pd
import numpy as np
import typing as tp
from numpy.typing import NDArray


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
