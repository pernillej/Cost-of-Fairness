from sklearn.metrics import roc_auc_score
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from src.data import dataframe_to_dataset, get_privileged_and_unprivileged_groups


def auc(y_true, y_pred):
    y_true = preprocess_y_true(y_true)
    y_pred = preprocess_y_pred(y_pred)
    return roc_auc_score(y_true, y_pred)


def statistical_parity(X_test, y_pred, data_attributes):
    """
    Calculate statistical parity score: 1 - |Statistical Parity Difference|

    :param X_test: Dataset used to generate predictions
    :param y_pred:  Predictions
    :param data_attributes: Attributes describing the dataset
    :return: Statistical parity score
    """
    df = X_test.assign(credit=y_pred)
    data = dataframe_to_dataset(df, data_attributes)
    if type(data) != BinaryLabelDataset:
        raise Exception('Not valid dataset. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes)
    data_metric = BinaryLabelDatasetMetric(data, unprivileged_groups=unprivileged, privileged_groups=privileged)
    # Must have abs() because metric can be both + and -
    return 1 - abs(data_metric.statistical_parity_difference())


def preprocess_y_true(y_true):
    """
    Preprocess german dataset labels changing 1->0, and 2->1 for easier calculation of auc scores etc.

    :param y_true: Labels to preprocess
    :return: Processed labels
    """
    new_y_true = []
    for y in y_true:
        new_y_true.append(y - 1)
    return new_y_true


def preprocess_y_pred(y_pred):
    """
    Preprocess german dataset probability predictions to only contain one column (the column predicting label 2).

    :param y_pred: Predictions to preprocess
    :return: Processed predictions
    """
    return y_pred[:, 1]
