from src.experiments.reweighing import svm_reweighing_experiment
from src.experiments.roc import svm_roc_experiment
from src.experiments.baseline import svm_experiment
from src.experiments.calibrated_eq_odds import svm_caleqodds_experiment
from src.experiments.disparate_impact_remover import svm_dir_experiment
from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file

"""
Done: 
1. Collect the dataset(s)
2. Setup SVM with 5-fold testing.
3. Add basic metrics
4. Setup NSGA2
5. Add saving and plotting of results
6. Add out mitigation methods to svm

TODO:
Customize...
"""


def run_experiment():
    print("Running SVM")
    result = svm_experiment()
    print('Results: ' + str(result))
    print("Running SVM with Reweighing")
    result = svm_reweighing_experiment()
    print('Results: ' + str(result))
    print("Running SVM with DisparateImpactRemover")
    result = svm_dir_experiment()
    print('Results: ' + str(result))
    print("Running SVM with Calibrated Eq Odds")
    result = svm_caleqodds_experiment()
    print('Results: ' + str(result))
    print("Running SVM with ROC")
    result = svm_roc_experiment()
    print('Results: ' + str(result))


def plot():
    svm_results = read_result_from_file('svm_18-04-2020_15-07.txt')
    reweighing_results = read_result_from_file('svm_reweighing_18-04-2020_15-07.txt')
    dir_results = read_result_from_file('svm_dir_18-04-2020_15-07.txt')
    caleqodds_results = read_result_from_file('svm_caleqodds_18-04-2020_15-07.txt')

    plot_results([svm_results, reweighing_results, dir_results, caleqodds_results])


run_experiment()
# plot()
