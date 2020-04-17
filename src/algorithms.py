from sklearn.svm import SVC
from sklearn.model_selection import KFold
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.postprocessing import RejectOptionClassification
from src.data import get_privileged_and_unprivileged_groups, dataframe_to_dataset, to_dataframe


def baseline_svm(data, metrics, data_attributes, C, gamma, drop_features, label_name):
    """
    Baseline SVM with no mitigation added.

    :param data: The dataset to run svm on
    :param metrics: The metrics to calculate
    :param data_attributes: Attributes describing the dataset
    :param C: The C value to be used in SVM
    :param gamma: The gamma value to be used in SVM
    :param drop_features: The features to drop in SVM
    :return: Tuple with the scores for the metrics: (accuracy_score, fairness_score)
    """
    return kfold_svm(data, metrics, data_attributes, C, gamma, drop_features, label_name)


def kfold_svm(data, metrics, data_attributes, C, gamma, drop_features, label_name, k=5):
    """
    Central SVM algorithm with kfold cross validation.

    :param data: The dataset to run kfold svm on
    :param metrics: The metrics to calculate
    :param data_attributes: Attributes describing the dataset
    :param C: The C value to be used in SVM
    :param gamma: The gamma value to be used in SVM
    :param drop_features: The features to drop in SVM
    :param k: Amount of folds. (Default=5)
    :return: Tuple with the scores for the metrics: (accuracy_score, fairness_score)
    """
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]
        X_orig = train.drop(label_name, axis=1)
        X = X_orig.drop(drop_features, axis=1)
        y = train[label_name]
        X_test_orig = test.drop(label_name, axis=1)
        X_test = X_test_orig.drop(drop_features, axis=1)
        y_test = test[label_name]

        clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        clf.fit(X, y)
        y_prob_pred = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        # score = clf.score(X_test, y_test)  # Basic mean average score included in svc class
        accuracy_scores.append(metrics['accuracy'](y_test, y_prob_pred))
        fairness_scores.append(metrics['fairness'](test, y_pred, data_attributes))

    avg_accuracy = sum(accuracy_scores)/len(accuracy_scores)
    avg_fairness = sum(fairness_scores)/len(fairness_scores)
    return avg_accuracy, avg_fairness


"""
Add more algorithms with mitigation methods here:
"""


def perform_reweighing(dataframe, df_attributes):
    dataset = dataframe_to_dataset(dataframe, df_attributes)
    unprivileged, privileged = get_privileged_and_unprivileged_groups(dataset.protected_attribute_names,
                                                                      dataset.unprivileged_protected_attributes,
                                                                      dataset.privileged_protected_attributes)
    rw = Reweighing(unprivileged_groups=unprivileged, privileged_groups=privileged)
    rw_dataset = rw.fit_transform(dataset)
    return to_dataframe(rw_dataset, df_attributes["favorable_label"], df_attributes["unfavorable_label"])


def svm_reweighing(data, metrics, data_attributes, C, gamma, drop_features, k=5):
    """
    SVM with Reweighing preprocessing

    :param data: The dataset to run on
    :param metrics: The metrics to calculate
    :param data_attributes: Attributes describing the dataset
    :param C: The C value to be used in SVM
    :param gamma: The gamma value to be used in SVM
    :param drop_features: The features to drop in SVM
    :param k: Amount of folds. (Default=5)
    :return: Tuple with the scores for the metrics: (accuracy_score, fairness_score)
    """
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]

        # Reweighing
        train, train_attributes = perform_reweighing(train, data_attributes)
        train = train.assign(sample_weights=train_attributes['instance_weights'])

        X_orig = train.drop(['credit', 'sample_weights'], axis=1)
        X = X_orig.drop(drop_features, axis=1)
        y = train['credit']
        sw = train['sample_weights']
        X_test_orig = test.drop('credit', axis=1)
        X_test = X_test_orig.drop(drop_features, axis=1)
        y_test = test['credit']

        clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        clf.fit(X, y, sample_weight=sw)
        y_prob_pred = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        # score = clf.score(X_test, y_test)  # Basic mean average score included in svc class
        accuracy_scores.append(metrics['accuracy'](y_test, y_prob_pred))
        fairness_scores.append(metrics['fairness'](test, y_pred, data_attributes))

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    return avg_accuracy, avg_fairness


def perform_dir(repair_level, dataframe, df_attributes):
    dataset = dataframe_to_dataset(dataframe, df_attributes)
    di = DisparateImpactRemover(repair_level=repair_level)
    di_dataset = di.fit_transform(dataset)
    return to_dataframe(di_dataset, df_attributes["favorable_label"], df_attributes["unfavorable_label"])


def svm_dir(data, metrics, data_attributes, C, gamma, repair_level, drop_features, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]

        # Disparate Impact Remover
        train, train_attributes = perform_dir(repair_level, train, data_attributes)
        test, test_attributes = perform_dir(repair_level, test, data_attributes)

        X_orig = train.drop('credit', axis=1)
        X = X_orig.drop(drop_features, axis=1)
        y = train['credit']
        X_test_orig = test.drop('credit', axis=1)
        X_test = X_test_orig.drop(drop_features, axis=1)
        y_test = test['credit']

        clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        clf.fit(X, y)
        y_prob_pred = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        # score = clf.score(X_test, y_test)  # Basic mean average score included in svc class
        accuracy_scores.append(metrics['accuracy'](y_test, y_prob_pred))
        fairness_scores.append(metrics['fairness'](test, y_pred, data_attributes))

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    return avg_accuracy, avg_fairness


"""
Not working...
"""


def fit_ROC(orig_dataframe, pred_dataframe, df_attributes):
    orig_dataset = dataframe_to_dataset(orig_dataframe, df_attributes)
    pred_dataset = dataframe_to_dataset(pred_dataframe, df_attributes)
    unprivileged, privileged = get_privileged_and_unprivileged_groups(orig_dataset.protected_attribute_names,
                                                                      orig_dataset.unprivileged_protected_attributes,
                                                                      orig_dataset.privileged_protected_attributes)
    roc = RejectOptionClassification(unprivileged_groups=unprivileged, privileged_groups=privileged)
    roc = roc.fit(orig_dataset, pred_dataset)
    return roc


def svm_ROC(data, metrics, data_attributes, C, gamma, drop_features, k=5):
    # NOT FINISHED, something wrong with predictions, all scores either 0 or 1
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]
        X_orig = train.drop('credit', axis=1)
        X = X_orig.drop(drop_features, axis=1)
        y = train['credit']
        X_test_orig = test.drop('credit', axis=1)
        X_test = X_test_orig.drop(drop_features, axis=1)
        y_test = test['credit']

        clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        clf.fit(X, y)
        # y_prob_pred = clf.predict_proba(X_test)
        y_train_pred = clf.predict(X)
        # score = clf.score(X_test, y_test)  # Basic mean average score included in svc class
        # accuracy_scores.append(metrics['accuracy'](y_test, y_prob_pred))
        # fairness_scores.append(metrics['fairness'](test, y_pred, data_attributes))

        # ROC
        roc = fit_ROC(train, train.assign(credit=y_train_pred), data_attributes)
        test_dataset = dataframe_to_dataset(test, data_attributes)
        test_pred_set = roc.predict(test_dataset)
        print(test_pred_set.accuracy)
        accuracy_scores.append(metrics['accuracy'](y_test, test_pred_set.scores))
        fairness_scores.append(metrics['fairness'](test, test_pred_set.labels, data_attributes))

    print(accuracy_scores)
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    return avg_accuracy, avg_fairness


def meta_fair(data, metrics, data_attributes, tau, drop_features, k=5):
    # NOT FINISHED, runs slow, gets zero division error for gamma
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]
        X = train.drop(drop_features, axis=1)
        X_test = test.drop(drop_features, axis=1)

        # To aif360 dataset
        X = dataframe_to_dataset(X, data_attributes)
        X_test = dataframe_to_dataset(X_test, data_attributes)

        # Metafair
        mf = MetaFairClassifier(tau=tau)
        mf.fit(X)

        pred_dataset = mf.predict(X_test)
        print(pred_dataset.scores)

        accuracy_scores.append(metrics['accuracy'](pred_dataset.labels, pred_dataset.scores))
        #fairness_scores.append(metrics['fairness'](test, y_pred, data_attributes))


    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_fairness = sum(fairness_scores) / len(fairness_scores)
    return avg_accuracy, avg_fairness
