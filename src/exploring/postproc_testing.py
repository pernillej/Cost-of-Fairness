from src.data import load_german_dataset
from aif360.algorithms.postprocessing import RejectOptionClassification, CalibratedEqOddsPostprocessing, \
    EqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

dataset = load_german_dataset()
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

scale_orig = StandardScaler()

# Train
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()
w_train = dataset_orig_train.instance_weights

clf = SVC(C=0.1, gamma='scale', kernel='rbf', probability=True)
clf.fit(X_train, y_train, sample_weight=w_train)

# Test
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_orig.transform(dataset_orig_test_pred.features)
y_test = dataset_orig_test_pred.labels
# positive class index
pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_train_pred.labels = clf.predict(X_train)
dataset_orig_train_pred.scores = clf.predict_proba(X_train)[:, pos_ind].reshape(-1, 1)
dataset_orig_test_pred.labels = clf.predict(X_test)
dataset_orig_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
"""
# ROC
pp = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
pp = pp.fit(dataset_orig_train, dataset_orig_train_pred)
"""
# Calibrated Equalized Odds
pp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
pp = pp.fit(dataset_orig_train, dataset_orig_train_pred)
"""
# Equalized Odds
pp = EqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
pp = pp.fit(dataset_orig_train, dataset_orig_train_pred)
"""
dataset_transf_test_pred = pp.predict(dataset_orig_test_pred)

cm = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                          unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

print(cm.accuracy())
print(1-abs(cm.statistical_parity_difference()))
