from sklearn.svm import SVC
from src.data import get_X_and_y
from sklearn.model_selection import KFold


def baseline_svm(data, data_attributes):
    kf = KFold(n_splits=5, shuffle=True)

    for train_indexes, test_indexes in kf.split(data):
        train = data.iloc[train_indexes]
        test = data.iloc[test_indexes]
        X = train.drop('credit', axis=1)
        y = train['credit']
        X_test = test.drop('credit', axis=1)
        y_test = test['credit']

        clf = SVC(C=0.1, gamma='scale', kernel='rbf')
        clf.fit(X, y)
        # TODO: update scoring to include the two metrics
        score = clf.score(X_test, y_test)
        print(score)

