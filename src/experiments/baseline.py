from src.metrics import auc, statistical_parity
from src.data import load_german_dataframe
from src.algorithms import baseline_svm

metrics = {
        'accuracy': auc,
        'fairness': statistical_parity
    }


def baseline_experiment():
    df, df_attributes = load_german_dataframe()

    accuracy_score, fairness_score = baseline_svm(df, metrics, df_attributes)
    print(accuracy_score)
    print(fairness_score)
