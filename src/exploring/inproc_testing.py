from src.data import load_german_dataset
from src.metrics import auc
from aif360.algorithms.inprocessing import AdversarialDebiasing, MetaFairClassifier, PrejudiceRemover
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf

dataset = load_german_dataset()
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)

scale_orig = StandardScaler()

# Train
dataset_train = dataset_orig_train.copy()
dataset_train.features = scale_orig.fit_transform(dataset_train.features)

# Adverserial Debiasing
sess = tf.Session()
model = AdversarialDebiasing(privileged_groups=privileged_groups,
                             unprivileged_groups=unprivileged_groups,
                             scope_name='debiased_classifier',
                             debias=True,
                             sess=sess)
"""

# Prejudice Remover
model = PrejudiceRemover(sensitive_attr="age", eta=25.0)
"""

model.fit(dataset_train)

dataset_test = dataset_orig_test.copy()
dataset_test.features = scale_orig.fit_transform(dataset_test.features)
dataset_debiasing_test = model.predict(dataset_test)

cm = ClassificationMetric(dataset_test, dataset_debiasing_test,
                          unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

print(cm.accuracy())
print(1-abs(cm.theil_index()))

"""
# Test
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_orig.transform(dataset_orig_test_pred.features)
y_test = dataset_orig_test_pred.labels
# positive class index
pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_orig_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
"""