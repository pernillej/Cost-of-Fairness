from sklearn.svm import SVC
from sklearn.model_selection import KFold


def baseline_svm(data, metrics, data_attributes):
    return kfold_svm(data, metrics, data_attributes)


def kfold_svm(data, metrics, data_attributes, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    accuracy_scores = []
    fairness_scores = []

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]
        X = train.drop('credit', axis=1)
        y = train['credit']
        X_test = test.drop('credit', axis=1)
        y_test = test['credit']

        clf = SVC(C=0.1, gamma='scale', kernel='rbf', probability=True)
        clf.fit(X, y)
        y_prob_pred = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        # score = clf.score(X_test, y_test)  # Basic mean average score included in svc class
        accuracy_scores.append(metrics['accuracy'](y_test, y_prob_pred))
        fairness_scores.append(metrics['fairness'](X_test, y_pred, data_attributes))

    avg_accuracy = sum(accuracy_scores)/len(accuracy_scores)
    avg_fairness = sum(fairness_scores)/len(fairness_scores)
    return avg_accuracy, avg_fairness
