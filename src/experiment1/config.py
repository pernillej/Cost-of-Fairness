from src.metrics import statistical_parity_difference, theil_index, binary_accuracy

theil_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 10000,
  "svm_seed": 0,
  "fairness_metric": theil_index,
  "accuracy_metric": binary_accuracy
}

statistical_parity_config = {
  "num_generations": 100,
  "pop_size": 50,
  "mutation_rate": 0.05,
  "crossover_rate": 0.7,
  "svm_max_iter": 10000,
  "svm_seed": 0,
  "fairness_metric": statistical_parity_difference,
  "accuracy_metric": binary_accuracy
}
