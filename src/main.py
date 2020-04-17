from src.experiments.baseline import baseline_experiment
from src.experiments.reweighing import reweighing_experiment
from src.experiments.metafair import metafair_experiment
from src.experiments.roc import roc_experiment
from src.experiments.dir import dir_experiment
from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file
from src.data import load_compas_dataframe

"""
Done: 
1. Collect the dataset(s)
2. Setup SVM with 5-fold testing.
3. Add basic metrics
4. Setup NSGA2
5. Add saving and plotting of results

TODO:
5. Add/switch out mitigation methods to svm
6. Customize...
"""


def run_experiment():
    result = baseline_experiment()
    # result = reweighing_experiment()
    # result = metafair_experiment() NOT WORKING
    # result = roc_experiment() NOT WORKING
    # result = dir_experiment()
    print('Results: ' + str(result))


def plot():
    data1 = read_result_from_file('svm_reweighing_17-04-2020_10-42.txt')  # Update to match desired file
    data2 = read_result_from_file('baseline_svm_17-04-2020_10-35.txt')  # Update to match desired file
    plot_results([data1, data2])


run_experiment()
# plot()
