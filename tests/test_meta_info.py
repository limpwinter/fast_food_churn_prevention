import pandas as pd
import pytest
from src import MetaInfo

CHURN_FILL = 198


@pytest.fixture(scope="session")
def meta_info(train_dataframe):
    return MetaInfo(train_dataframe, fill_churn=CHURN_FILL)


@pytest.fixture(scope="session")
def fitted_train_data(meta_info):
    return meta_info.generate_meta_features()


@pytest.fixture(scope="session")
def fitted_train_labels(meta_info):
    return meta_info.get_labels()


@pytest.fixture(scope="session")
def train_dataframe():
    # Create a sample DataFrame
    data = pd.read_parquet("../data/train.parquet")
    df = pd.DataFrame(data)
    return df


def test_is_dataframe(fitted_train_data):
    assert isinstance(fitted_train_data, pd.DataFrame)


def test_fitted_meta_info_column_names(fitted_train_data):
    assert sorted(fitted_train_data.columns.tolist()) == sorted(
        [
            'unique_places',
            'orders_count',
            'total_spend',
            'avg_bill',
            'unique_days'
        ])


def test_zero_nan_unique_places(fitted_train_data):
    assert fitted_train_data["unique_places"].isna().sum() == 0


def test_zero_nan_num_orders(fitted_train_data):
    assert fitted_train_data["orders_count"].isna().sum() == 0


def test_zero_nan_total_spend_money(fitted_train_data):
    assert fitted_train_data["total_spend"].isna().sum() == 0


def test_zero_nan_total_spend_money_per_order(fitted_train_data):
    assert fitted_train_data["avg_bill"].isna().sum() == 0


def test_zero_nan_values_in_labels(fitted_train_labels):
    assert fitted_train_labels.isna().sum().all() == 0


def test_equal_x_y_sizes(fitted_train_data, fitted_train_labels):
    assert fitted_train_data.shape[0] == fitted_train_labels.shape[0]


def test_date_diff_post_is_churn_fill_value(train_dataframe, fitted_train_labels):
    # TODO FIX IT to no
    # churn_customers_indexes = train_dataframe[train_dataframe['date_diff_post'].isna()].index
    # assert fitted_train_labels.loc[churn_customers_indexes]['date_diff_post'].all() == CHURN_FILL
    assert (fitted_train_labels['date_diff_post'] == CHURN_FILL).any()
