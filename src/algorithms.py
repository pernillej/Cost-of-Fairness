from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import RejectOptionClassification, CalibratedEqOddsPostprocessing
from aif360.algorithms.inprocessing import MetaFairClassifier, PrejudiceRemover
from aif360.metrics import ClassificationMetric
import numpy as np


def svm(dataset, fairness_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
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

    cm = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_reweighing(dataset, fairness_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
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

    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_dir(dataset, fairness_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
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

    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_optimpreproc(dataset, fairness_metric, C, gamma, privileged_groups, unprivileged_groups):
    return svm(dataset=dataset, fairness_metric=fairness_metric, C=C, gamma=gamma, keep_features=[],
               privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)


def svm_roc(dataset, fairness_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
    # Split data
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
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = clf.predict(X_train)
    dataset_orig_train_pred.scores = clf.predict_proba(X_train)[:, pos_ind].reshape(-1, 1)
    dataset_orig_test_pred.labels = clf.predict(X_test)
    dataset_orig_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    pp = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    pp = pp.fit(dataset_orig_train, dataset_orig_train_pred)

    dataset_transf_test_pred = pp.predict(dataset_orig_test_pred)

    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def svm_ceq(dataset, fairness_metric, C, gamma, keep_features, privileged_groups, unprivileged_groups):
    # Split data
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Prepare data
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    if len(keep_features) > 0:  # If keep_features empty, use all features
        X_train = X_train[:, keep_features]
        X_test = X_test[:, keep_features]

    # Train
    clf = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Test
    pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]  # positive class index
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = clf.predict(X_train)
    dataset_orig_train_pred.scores = clf.predict_proba(X_train)[:, pos_ind].reshape(-1, 1)
    dataset_orig_test_pred.labels = clf.predict(X_test)
    dataset_orig_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    pp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    pp = pp.fit(dataset_orig_train, dataset_orig_train_pred)

    dataset_transf_test_pred = pp.predict(dataset_orig_test_pred)

    cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def meta_fair(dataset, fairness_metric, tau, sensitive_attr, privileged_groups, unprivileged_groups):
    # Split dataset
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Prepare data
    scale = StandardScaler()
    dataset_train = dataset_orig_train.copy()
    dataset_train.features = scale.fit_transform(dataset_train.features)
    dataset_test = dataset_orig_test.copy()
    dataset_test.features = scale.fit_transform(dataset_test.features)

    # Train
    model = MetaFairClassifier(tau=tau, sensitive_attr=sensitive_attr)
    model.fit(dataset_orig_train)

    # Test
    dataset_debiasing_test = model.predict(dataset_test)

    cm = ClassificationMetric(dataset_test, dataset_debiasing_test,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score


def prejudice_remover(dataset, fairness_metric, eta, sensitive_attr, privileged_groups, unprivileged_groups):
    # Split dataset
    dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

    # Prepare data
    scale = StandardScaler()
    dataset_train = dataset_orig_train.copy()
    dataset_train.features = scale.fit_transform(dataset_train.features)
    dataset_test = dataset_orig_test.copy()
    dataset_test.features = scale.fit_transform(dataset_test.features)

    # Train
    model = PrejudiceRemover(sensitive_attr=sensitive_attr, eta=eta)
    model.fit(dataset_orig_train)

    # Test
    dataset_debiasing_test = model.predict(dataset_test)

    cm = ClassificationMetric(dataset_test, dataset_debiasing_test,
                              unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy_score = cm.accuracy()
    fairness_score = fairness_metric(cm)
    return accuracy_score, fairness_score




