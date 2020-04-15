from sklearn.svm import SVC
from sklearn.model_selection import KFold


def baseline_svm(data, metrics, data_attributes, C, gamma, drop_features):
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
    return kfold_svm(data, metrics, data_attributes, C, gamma, drop_features)


def kfold_svm(data, metrics, data_attributes, C, gamma, drop_features, k=5):
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

    avg_accuracy = sum(accuracy_scores)/len(accuracy_scores)
    avg_fairness = sum(fairness_scores)/len(fairness_scores)
    return avg_accuracy, avg_fairness


"""
Add more algorithms with mitigation methods here:
"""