from src.experiments.baseline import baseline_experiment
from src.util.plotter import plot_results
from src.util.filehandler import write_result_to_file, read_result_from_file

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
    baseline_result = baseline_experiment()
    print('Results: ' + str(baseline_result))


def plot():
    data1 = read_result_from_file('baseline_svm_16-04-2020_11-34.txt')  # Update to match desired file
    data2 = read_result_from_file('baseline_svm_16-04-2020_11-55.txt')  # Update to match desired file
    plot_results([data1, data2])


plot()
