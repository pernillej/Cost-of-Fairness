from src.experiments.baseline import baseline_experiment
from src.util.plotter import plot

"""
Done: 
1. Collect the dataset(s)
2. Setup SVM with 5-fold testing.
3. Add basic metrics
4. Setup GA chromosome (aka. SVM parameters, with mutation and crossover) to fit DEAP

TODO:
5. Add/switch out mitigation methods to svm
6. Customize...
"""

baseline_result = baseline_experiment()
print('Results: ' + str(baseline_result))
plot(baseline_result)

