import aif360
import sklearn
import numpy as np
from deap import algorithms

"""
TODO:
1. Understand how the packages interconnect
2. Collect the dataset(s)
3. Setup GA chromosome (aka. SVM parameters, with mutation and crossover) to fit DEAP
4. Setup SVM with 5-fold testing.
5. Add/switch out mitigation methods to svm
6. Find metrics
etc...

Setup:
1. Collect data
2. Run NSGA-II with custom chromosome of SVM parameters, 
with the different SVM setups for fitness calculation using chosen metrics
"""

print('Hello world')
