# Investigating the Cost of Fairness in Automated Decision Making Systems
My master thesis project. Uses multi-objective evolutionary algorithms and Pareto fronts to investigate the cost of fairness in automated decision making systems. Inspired by:
- Haas, C. (2019).  The Price of Fairness - A Framework to Explore Trade-Offs in Algorithmic Fairness.  In 40th  International  Conference  on  Information  Systems, ICIS 2019, pages 1–17

## Requirements
This code is verified to run using:

- Python 3.7.3
- Modules:

|Module      |Version|Notes|
|------------|-------|-----|
|numpy       |1.16.4 |     |
|scikit-learn|0.21.2 |     |
|aif360      |0.2.3  |     |
|pandas      |0.24.2 |At the time of writing the code appears incompatible with the newest versions of pandas. We believe this is because of the pandas version used by aif360.|
|matplotlib  |3.1.0  |     |

## User guide
The project consists of code for running 3 different experiments. The code for each experiment is contained in 3 separate folders. In addition, some code is used across experiments and exist in their own folders.
In order to run the code the requirements, as described above, have to be installed. 

#### Content description
- `src/nsga2` - This folder contains the code used to perform NSGA-II operations used to produce Pareto Frontiers.
- `src/util` - This folder contains utility code used to read and write result files, plot results, and convert from binary to decimal values.
- `src/compas_analysis.py` - This file contains the code used to analyse the COMPAS data set that is used in all 3 experiments.
- `src/data.py` - This file contains the code used to gather the COMPAS data set from `aif360`.
- `src/metrics.py` - This file defines the possible fairness and accuracy metrics that can be used in each experiment.
- `src/experiment1`, `src/experiment2`, `src/experiment3` - These folders contain the code used to run each respective experiment. Each folder contains:
    - `/results` - This folder is used to store the results as .txt files containing json dictionaries describing the results and configurations from each run of the experiment.
    - `algorithms.py` - This file defines the 4 algorithms: `svm`, `svm_reweighing`, `svm_dir`, `svm_optimpreproc`. See thesis for the purpose of these algorithms. 
    - `config.py` - This file defines the run configurations for the experiment.
    - `main.py` - This file is the file used to define and run the experiment.
    - `plot_results.py` - This file is used to plot the results from the experiment. 
    - `baseline.py`, `disparate_impact_remover.py`, `optimpreproc.py`, `reweighing.py` - These files initiate each algorithm into the NSGA-II optimization approach, by running NSGA-II using the proper parameters, and defining the evaluation function to be used by NSGA-II.

#### Running an experiment
Each experiment folder contains its own `main.py` file. Running this file will run the entire experiment as it was performed in the thesis.

##### Run configurations
Each experiment folder also contains its own `config.py` file that can be used to update NSGA-II parameters like number of generations, population size, mutation and crossover rate. 
The max iterations and seed used for the SVM classifiers can also be changed. 
Finally, it is also possible to update which fairness and accuracy metrics are used.

Possible accuracy metrics: `auc` and `binary_accuracy`.
Possible fairness metrics: `statistical_parity_difference`, `theil_index`, `equal_opportunity_difference`, `average_odds_difference`, and `disparate_impact`.
