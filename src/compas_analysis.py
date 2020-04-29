from src.data import load_compas_dataset
from aif360.metrics import BinaryLabelDatasetMetric
import collections
import matplotlib.pyplot as plt

"""
File used to provide data and plots for analysis of the Compas Data Set
"""

train, test = load_compas_dataset()
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
train_metric = BinaryLabelDatasetMetric(dataset=train, privileged_groups=privileged_groups,
                                        unprivileged_groups=unprivileged_groups)
test_metric = BinaryLabelDatasetMetric(dataset=test, privileged_groups=privileged_groups,
                                       unprivileged_groups=unprivileged_groups)


def plot_label_distribution(dataset_metric, y_ticks, title='Distribution of labels'):
    neg = dataset_metric.num_negatives()
    pos = dataset_metric.num_positives()
    # Label 0 is the favorable label
    labels = ['No recid.', 'Did recid.']
    count = [pos, neg]
    plt.bar(labels, count)
    x_ticks = [0, 1]
    plt.gca().set_xticks([0, 1])
    plt.gca().set_yticks(y_ticks)
    plt.title(title)
    for i in range(len(count)):
        plt.text(x=x_ticks[i] - 0.1, y=count[i] + y_ticks[-1]*0.01, s=str(int(count[i])), size=13)
    plt.show()


def plot_priv_label_distribution(dataset_metric, y_ticks, title='Distribution of labels'):
    neg = dataset_metric.num_negatives(True)
    pos = dataset_metric.num_positives(True)
    labels = ['No recid.', 'Did recid.']
    count = [pos, neg]
    plt.bar(labels, count, color=(1.0, 0.85, 0.0))
    x_ticks = [0, 1]
    plt.gca().set_xticks([0, 1])
    plt.gca().set_yticks(y_ticks)
    plt.title(title)
    for i in range(len(count)):
        plt.text(x=x_ticks[i] - 0.1, y=count[i] + y_ticks[-1]*0.01, s=str(int(count[i])), size=13)
    plt.show()


def plot_unpriv_label_distribution(dataset_metric, y_ticks, title='Distribution of labels'):
    neg = dataset_metric.num_negatives(False)
    pos = dataset_metric.num_positives(False)
    labels = ['No recid.', 'Did recid.']
    count = [pos, neg]
    plt.bar(labels, count, color=(0.0, 0.4, 0.0))
    x_ticks = [0, 1]
    plt.gca().set_xticks([0, 1])
    plt.gca().set_yticks(y_ticks)
    plt.title(title)
    for i in range(len(count)):
        plt.text(x=x_ticks[i] - 0.1, y=count[i] + y_ticks[-1]*0.01, s=str(int(count[i])), size=13)
    plt.show()


def plot_race_distribution(dataset, y_ticks, title='Distribution of race attribute'):
    race_count = collections.Counter(dataset.features[:, 2])
    labels = ['Not Caucasian', 'Caucasian']
    count = [race_count[0.0], race_count[1.0]]  # 1 Caucasian, 0 not caucasian
    plt.bar(labels, count, color=(1.0, 0.4, 0.0))
    x_ticks = [0, 1]
    plt.gca().set_xticks(x_ticks)
    plt.gca().set_yticks(y_ticks)
    plt.title(title)
    for i in range(len(count)):
        plt.text(x=x_ticks[i] - 0.1, y=count[i] + y_ticks[-1]*0.01, s=str(int(count[i])), size=13)
    plt.show()


def plot_label_distributions():
    plot_label_distribution(train_metric, y_ticks=range(0, 3100, 500),
                            title='Distribution of labels for the training set')
    plot_label_distribution(test_metric, y_ticks=range(0, 900, 100),
                            title='Distribution of labels for the test set')


def plot_priv_label_distributions():
    plot_priv_label_distribution(train_metric, y_ticks=range(0, 1210, 200),
                                 title='Distribution of labels for Caucasians in the training set')
    plot_priv_label_distribution(test_metric, y_ticks=range(0, 310, 50),
                                 title='Distribution of labels for Caucasians in the test set')


def plot_unpriv_label_distributions():
    plot_unpriv_label_distribution(train_metric, y_ticks=range(0, 2100, 500),
                                   title='Distribution of labels for Non Caucasians in the training set')
    plot_unpriv_label_distribution(test_metric, y_ticks=range(0, 510, 100),
                                   title='Distribution of labels for Non Caucasians in the test set')


def plot_race_distributions():
    plot_race_distribution(train, y_ticks=range(0, 3600, 500),
                           title='Distribution of race attribute in the training set')
    plot_race_distribution(test, y_ticks=range(0, 1000, 100),
                           title='Distribution of race attribute in the test set')


def get_disparate_impact(dataset_metric):
    return dataset_metric.disparate_impact()


def get_statistical_parity_difference(dataset_metric):
    return dataset_metric.statistical_parity_difference()


def get_num_samples(dataset):
    return len(dataset.features)


def get_num_features(dataset):
    return len(dataset.feature_names)
