from sklearn.metrics import roc_auc_score
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from src.data import dataframe_to_dataset, get_privileged_and_unprivileged_groups


def auc(y_true, y_pred):
    """
    Get the AUC score for the predictions

    :param y_true: The ground truth labels
    :param y_pred: The predicted labels
    :return: The AUC score
    """
    y_true = preprocess_y_true(y_true)
    y_pred = preprocess_y_pred(y_pred)
    return roc_auc_score(y_true, y_pred)


def statistical_parity(X_test_orig, y_pred, data_attributes, sample_weights_column_name=None):
    """
    Calculate statistical parity score: 1 - |Statistical Parity Difference|

    :param X_test_orig: Dataset used to generate predictions, including the original label
    :param y_pred: Predictions
    :param data_attributes: Attributes describing the dataset
    :param sample_weights_column_name: Column name in df corresponding to instance/sample weights
    :return: Statistical parity score
    """
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes,
                                     sample_weights_column_name=sample_weights_column_name)
    # Replace labels with predicted labels
    pred_df = X_test_orig.drop(data_attributes["label_names"][0], axis=1)
    pred_df[data_attributes["label_names"][0]] = y_pred
    pred_data = dataframe_to_dataset(pred_df, data_attributes, sample_weights_column_name=sample_weights_column_name)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes["protected_attribute_names"],
                                                                      data_attributes[
                                                                          "unprivileged_protected_attributes"],
                                                                      data_attributes[
                                                                          "privileged_protected_attributes"])
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    # Must have abs() because metric can be both + and -
    return 1 - abs(data_metric.statistical_parity_difference())


def theil_index(X_test_orig, y_pred, data_attributes, sample_weights_column_name=None):
    """
    Calculate Theil Index score: 1 - |Theil Index|

    :param X_test_orig: Dataset used to generate predictions, including the original label
    :param y_pred: Predictions
    :param data_attributes: Attributes describing the dataset
    :param sample_weights_column_name: Column name in df corresponding to instance/sample weights
    :return:
    """
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes,
                                     sample_weights_column_name=sample_weights_column_name)

    # Replace labels with predicted labels
    pred_df = X_test_orig.drop(data_attributes["label_names"][0], axis=1)
    pred_df[data_attributes["label_names"][0]] = y_pred
    pred_data = dataframe_to_dataset(pred_df, data_attributes, sample_weights_column_name=sample_weights_column_name)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes["protected_attribute_names"],
                                                                      data_attributes[
                                                                          "unprivileged_protected_attributes"],
                                                                      data_attributes[
                                                                          "privileged_protected_attributes"])
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    return 1 - data_metric.theil_index()


def equal_opportunity(X_test_orig, y_pred, data_attributes, sample_weights_column_name=None):
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes,
                                     sample_weights_column_name=sample_weights_column_name)

    # Replace labels with predicted labels
    pred_df = X_test_orig.drop(data_attributes["label_names"][0], axis=1)
    pred_df[data_attributes["label_names"][0]] = y_pred
    pred_data = dataframe_to_dataset(pred_df, data_attributes, sample_weights_column_name=sample_weights_column_name)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes["protected_attribute_names"],
                                                                      data_attributes[
                                                                          "unprivileged_protected_attributes"],
                                                                      data_attributes[
                                                                          "privileged_protected_attributes"])
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    # Must have abs() because metric can be both + and -
    return 1 - abs(data_metric.equal_opportunity_difference())


def disparate_impact(X_test_orig, y_pred, data_attributes, sample_weights_column_name=None):
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes,
                                     sample_weights_column_name=sample_weights_column_name)

    # Replace labels with predicted labels
    pred_df = X_test_orig.drop(data_attributes["label_names"][0], axis=1)
    pred_df[data_attributes["label_names"][0]] = y_pred
    pred_data = dataframe_to_dataset(pred_df, data_attributes, sample_weights_column_name=sample_weights_column_name)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes["protected_attribute_names"],
                                                                      data_attributes[
                                                                          "unprivileged_protected_attributes"],
                                                                      data_attributes[
                                                                          "privileged_protected_attributes"])
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    return data_metric.disparate_impact()


def average_odds(X_test_orig, y_pred, data_attributes, sample_weights_column_name=None):
    orig_data = dataframe_to_dataset(X_test_orig, data_attributes,
                                     sample_weights_column_name=sample_weights_column_name)

    # Replace labels with predicted labels
    pred_df = X_test_orig.drop(data_attributes["label_names"][0], axis=1)
    pred_df[data_attributes["label_names"][0]] = y_pred
    pred_data = dataframe_to_dataset(pred_df, data_attributes, sample_weights_column_name=sample_weights_column_name)
    if type(pred_data) != BinaryLabelDataset and type(orig_data) != BinaryLabelDataset:
        raise Exception('Invalid datasets. Must be BinaryLabelDataset')
    unprivileged, privileged = get_privileged_and_unprivileged_groups(data_attributes["protected_attribute_names"],
                                                                      data_attributes[
                                                                          "unprivileged_protected_attributes"],
                                                                      data_attributes[
                                                                          "privileged_protected_attributes"])
    data_metric = ClassificationMetric(orig_data, pred_data,
                                       unprivileged_groups=unprivileged,
                                       privileged_groups=privileged)
    # Must have abs() because metric can be both + and -
    return 1 - abs(data_metric.average_odds_difference())


"""
Utils
"""


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
    if len(y_pred[0]) > 1:
        return y_pred[:, 1]

    return y_pred


def function_name_to_string(func):
    """
    Produces a readable string from a metric function. Update when adding more metrics.

    :param func: The metric function to get a readable string from
    :return: String with name of metric
    """
    if func == auc:
        return "AUC"
    if func == statistical_parity:
        return "Statistical Parity Difference"
    if func == theil_index:
        return "Theil Index"
    if func == equal_opportunity:
        return "Equal Opportunity Difference"
    if func == disparate_impact:
        return "Disparate Impact"
    if func == average_odds:
        return "Average Odds Difference"
