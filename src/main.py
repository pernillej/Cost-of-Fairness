from src.data import load_german_dataset, to_dataframe
from src.algorithms import baseline_svm

"""
TODO:
1. Understand how the packages interconnect
2. Collect the dataset(s)
3. Setup SVM with 5-fold testing.
4. Setup GA chromosome (aka. SVM parameters, with mutation and crossover) to fit DEAP
5. Add/switch out mitigation methods to svm
6. Find metrics
etc...

Setup:
1. Collect data
2. Run NSGA-II with custom chromosome of SVM parameters, 
with the different SVM setups for fitness calculation using chosen metrics
"""

train, test = load_german_dataset()

df, df_attributes = to_dataframe(train)

baseline_svm(df, df_attributes)
