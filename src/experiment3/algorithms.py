from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.metrics import ClassificationMetric
import numpy as np
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def test_classifier(classifier, scale, test_data, fairness_metric, accuracy_metric, keep_features, privileged_threshold,
                    unprivileged_threshold, privileged_groups, unprivileged_groups):
    """
    Test the provided classifier on specified data set, and calculate fitness scores.

    :param classifier: The classifier to test
    :param scale: Scaler to transform the test set
    :param test_data: The test data set to test the classifier on
    :param fairness_metric: The fairness metric to calculate
    :param accuracy_metric: The accuracy metric to calculate
    :param keep_features: The features to keep for SVC
    :param privileged_threshold: The classification threshold to be used for the privileged group
    :param unprivileged_threshold: The classification threshold to be used for the unprivileged group
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :return:
    """
    dataset_orig_test = test_data

    # Prepare data
    dataset_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale.transform(dataset_test_pred.features)
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_test = X_test[:, keep_features]

    # Test
    pos_ind = np.where(classifier.classes_ == dataset_orig_test.favorable_label)[0][0]  # positive class index
    dataset_test_pred.scores = classifier.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    # Assign labels using the classification thresholds
    for i in range(len(dataset_test_pred.labels)):
        # 4 = index of the sensitive attr, 1 = privileged value
        if dataset_test_pred.features[i][4] == 1.:  # Privileged,
            if dataset_test_pred.scores[i] > privileged_threshold:  # Above threshold
                dataset_test_pred.labels[i] = dataset_test_pred.favorable_label
            else:
                dataset_test_pred.labels[i] = dataset_test_pred.unfavorable_label
        else:  # Unprivileged
            if dataset_test_pred.scores[i] > unprivileged_threshold:  # Above threshold
                dataset_test_pred.labels[i] = dataset_test_pred.favorable_label
            else:
                dataset_test_pred.labels[i] = dataset_test_pred.unfavorable_label

    # Calculate metrics
    cm = ClassificationMetric(dataset_orig_test, dataset_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy_score = accuracy_metric(cm)
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


@ignore_warnings(category=ConvergenceWarning)
def train_svm(training_data, C, gamma, keep_features, max_iter, svm_seed):
    """
    Train SVM(SVC) classifier on specified data set, with provided parameters, and calculate fitness scores.

    :param training_data: The training data set to run the classifier on
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param max_iter: Max iterations for SVM
    :param svm_seed: Seed used for RNG in SVM
    :return: The trained classifier, and the scaler
    """
    dataset_orig_train = training_data

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, max_iter=max_iter, random_state=svm_seed)
    clf.fit(X_train, y_train, sample_weight=w_train)

    return clf, scale


@ignore_warnings(category=ConvergenceWarning)
def train_svm_reweighing(training_data, C, gamma, keep_features, privileged_groups, unprivileged_groups, max_iter,
                         svm_seed):
    """
    Train the SVM classifier with Reweighing preprocessing on specified data set,
    with provided parameters, and calculate fitness scores.

    :param training_data: The training data set to run the classifier on
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param privileged_groups: The privileged group in the data set
    :param unprivileged_groups: The unprivileged group in the data set
    :param max_iter: Max iterations for SVM
    :param svm_seed: Seed used for RNG in SVM
    :return: The trained classifier and the scaler
    """
    dataset_orig_train = training_data

    # Run Reweighing
    rw = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    dataset_transf_train = rw.fit_transform(dataset_orig_train)

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    w_train = dataset_transf_train.instance_weights
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, max_iter=max_iter, random_state=svm_seed)
    clf.fit(X_train, y_train, sample_weight=w_train)

    return clf, scale


@ignore_warnings(category=ConvergenceWarning)
def train_svm_dir(training_data, C, gamma, keep_features, sensitive_attribute, max_iter, svm_seed):
    """
    Train the SVM classifier with Disparate Impact Remover preprocessing on specified data set,
    with provided parameters, and calculate fitness scores.

    :param training_data: The training data set to run the classifier on
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param sensitive_attribute: The sensitive attribute in the dataset
    :param max_iter: Max iterations for SVM
    :param svm_seed: Seed used for RNG in SVM
    :return: The trained classifier and the scaler
    """
    dataset_orig_train = training_data

    # Run Disparate Impact Remover
    di = DisparateImpactRemover(repair_level=0.8, sensitive_attribute=sensitive_attribute)
    dataset_transf_train = di.fit_transform(dataset_orig_train)

    # Prepare data
    scale = StandardScaler()
    X_train = scale.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()
    w_train = dataset_transf_train.instance_weights
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True, max_iter=max_iter, random_state=svm_seed)
    clf.fit(X_train, y_train, sample_weight=w_train)

    return clf, scale


@ignore_warnings(category=ConvergenceWarning)
def train_svm_optimpreproc(training_data, C, gamma, keep_features, max_iter, svm_seed):
    """
    Train SVM classifier with Optimized Preprocessing method on specified data set,
    with provided parameters, and calculate fitness scores.

    :param training_data: The training data set to run the classifier on
    :param C: The C parameter for SVC
    :param gamma: The gamma parameter for SVC
    :param keep_features: The features to keep for SVC
    :param max_iter: Max iterations for SVM
    :param svm_seed: Seed used for RNG in SVM
    :return: The trained classifier
    """
    return train_svm(training_data=training_data, C=C, gamma=gamma, keep_features=keep_features, max_iter=max_iter,
                     svm_seed=svm_seed)




