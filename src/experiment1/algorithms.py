from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.metrics import ClassificationMetric
import numpy as np


def svm(dataset, fairness_metric, accuracy_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
    """
    Run SVM(SVC) classifier on specified data set, with provided parameters, and calculate fitness scores.

    :param dataset: The data set to run the classifier on
    :param fairness_metric: The fairness metric to calculate
    :param accuracy_metric: The accuracy metric to calculate
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :return:
    """
    # Split dataset
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale.transform(dataset_orig_test_pred.features)
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]
        X_test = X_test[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Test
    pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]  # positive class index
    dataset_orig_test_pred.labels = clf.predict(X_test)
    dataset_orig_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    # Calculate metrics
    cm = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = accuracy_metric(cm)
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_reweighing(dataset, fairness_metric, accuracy_metric, C, gamma, keep_features, privileged_groups,
                   unprivileged_groups):
    """
    Run SVM classifier with Reweighing preprocessing on specified data set,
    with provided parameters, and calculate fitness scores.

    :param dataset: The data set to run the classifier on
    :param fairness_metric: The fairness metric to calculate
    :param accuracy_metric: The accuracy metric to calculate
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :return:
    """
    # Split dataset
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Run Reweighing
    rw = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    dataset_transf_train = rw.fit_transform(dataset_orig_train)

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    w_train = dataset_transf_train.instance_weights
    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale.fit_transform(dataset_transf_test_pred.features)
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]
        X_test = X_test[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Test
    pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]  # positive class index
    dataset_transf_test_pred.labels = clf.predict(X_test)
    dataset_transf_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    # Calculate metrics
    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy_score = accuracy_metric(cm)
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_dir(dataset, fairness_metric, accuracy_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
    """
    Run SVM classifier with Disparate Impact Remover preprocessing on specified data set,
    with provided parameters, and calculate fitness scores.

    :param dataset: The data set to run the classifier on
    :param fairness_metric: The fairness metric to calculate
    :param accuracy_metric: The accuracy metric to calculate
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :return:
    """
    # Split dataset
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Run Disparate Impact Remover
    di = DisparateImpactRemover(repair_level=0.8, sensitive_attribute='age')
    dataset_transf_train = di.fit_transform(dataset_orig_train)

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    w_train = dataset_transf_train.instance_weights
    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale.fit_transform(dataset_transf_test_pred.features)
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]
        X_test = X_test[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Test
    pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]  # positive class index
    dataset_transf_test_pred.labels = clf.predict(X_test)
    dataset_transf_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    # Calculate metrics
    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy_score = accuracy_metric(cm)
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_optimpreproc(dataset, fairness_metric, accuracy_metric, C, gamma, keep_features, privileged_groups,
                     unprivileged_groups):
    """
    Run SVM classifier with Optimized Preprocessing method on specified data set,
    with provided parameters, and calculate fitness scores.

    :param dataset: The data set to run the classifier on
    :param fairness_metric: The fairness metric to calculate
    :param accuracy_metric: The accuracy metric to calculate
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :return:
    """
    return svm(dataset=dataset, fairness_metric=fairness_metric, accuracy_metric=accuracy_metric, C=C, gamma=gamma,
               keep_features=keep_features, privileged_groups=privileged_groups,
               unprivileged_groups=unprivileged_groups)



