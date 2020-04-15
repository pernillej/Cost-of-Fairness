from sklearn.metrics import roc_auc_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from src.data import dataframe_to_dataset, get_privileged_and_unprivileged_groups


def auc(y_true, y_pred):
    y_true = preprocess_y_true(y_true)
    y_pred = preprocess_y_pred(y_pred)
    return roc_auc_score(y_true, y_pred)


def statistical_parity(X_test_orig, y_pred, data_attributes):
    """
    Calculate statistical parity score: 1 - |Statistical Parity Difference|

    :param X_test_orig: Dataset used to generate predictions, including the label
    :param y_pred:  Predictions
    :param data_attributes: Attributes describing the dataset
    :return: Statistical parity score
    """
    pred_df = X_test_orig.assign(credit=y_pred)
    pred_data = dataframe_to_dataset(pred_df, data_attributes)
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes)
    # data_metric = BinaryLabelDatasetMetric(pred_data, unprivileged_groups=unprivileged, privileged_groups=privileged)
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    # Must have abs() because metric can be both + and -
    return 1 - abs(data_metric.statistical_parity_difference())


def theil_index(X_test_orig, y_pred, data_attributes):
    pred_df = X_test_orig.assign(credit=y_pred)
    pred_data = dataframe_to_dataset(pred_df, data_attributes)
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes)
    # data_metric = BinaryLabelDatasetMetric(pred_data, unprivileged_groups=unprivileged, privileged_groups=privileged)
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    # Must have abs() because metric can be both + and - ??
    return 1 - data_metric.theil_index()


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
