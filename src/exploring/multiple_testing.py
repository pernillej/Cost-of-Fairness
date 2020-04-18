"""
from src.data import load_german_dataset
from src.metrics import auc
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf

dataset = load_german_dataset()
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

dataset_orig_train, dataset_orig_test = dataset.split([0.8], shuffle=True)


#########
# Pre-Processing
#########

# With Reweighing
rw = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
dataset_transf_train = rw.fit_transform(dataset_orig_train)

# With Disparate Impact Remover
di = DisparateImpactRemover(repair_level=0.8, sensitive_attribute='age')
dataset_transf_train = di.fit_transform(dataset_orig_train)

#########
# In-Processing
#########

# SVM
# Train
scale_transf = StandardScaler()
X_train = scale_transf.fit_transform(dataset_transf_train.features)
y_train = dataset_transf_train.labels.ravel()
w_train = dataset_transf_train.instance_weights

clf = SVC(C=1., gamma='scale', kernel='rbf', probability=True)
clf.fit(X_train, y_train, sample_weight=w_train)

dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
y_test = dataset_transf_test_pred.labels
# positive class index
pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_transf_test_pred.labels = clf.predict(X_test)
dataset_transf_test_pred.scores = clf.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

# Adverserial Debiasing
sess = tf.Session()
model = AdversarialDebiasing(privileged_groups=privileged_groups,
                             unprivileged_groups=unprivileged_groups,
                             scope_name='debiased_classifier',
                             debias=True,
                             sess=sess)

# Prejudice Remover
model = PrejudiceRemover(sensitive_attr="age", eta=25.0)


model.fit(dataset_transf_test_pred)

dataset_test = dataset_orig_test.copy()
dataset_test.features = scale_orig.fit_transform(dataset_test.features)
dataset_debiasing_test = model.predict(dataset_test)

#########
# Post-Processing
#########

...

#########
# Metrics
#########

cm = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                          unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

print(cm.accuracy())
print(1-abs(cm.statistical_parity_difference()))
"""