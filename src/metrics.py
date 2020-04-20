def statistical_parity_difference(classification_metric):
    """
    Calculate statistical parity score: 1 - |Statistical Parity Difference|

    :param classification_metric: aif369 ClassificationMetric type containing the true, and the predicted data sets
    :return: Statistical parity score
    """
    return 1 - abs(classification_metric.statistical_parity_difference())


def theil_index(classification_metric):
    """
    Calculate Theil Index score: 1 - |Theil Index|

    :param classification_metric: aif369 ClassificationMetric type containing the true, and the predicted data sets
    :return: Theil Index score
    """
    return 1 - abs(classification_metric.theil_index())


def equal_opportunity_difference(classification_metric):
    """
    Calculate Equal Opportunity Difference score: 1 - |Equal Opportunity Difference|

    :param classification_metric: aif369 ClassificationMetric type containing the true, and the predicted data sets
    :return: Equal Opportunity Difference score
    """
    return 1 - abs(classification_metric.equal_opportunity_difference())


def average_odds_difference(classification_metric):
    """
    Calculate Average Odds Difference score: 1 - |Average Odds Difference|

    :param classification_metric: aif369 ClassificationMetric type containing the true, and the predicted data sets
    :return: Equal Opportunity Difference score
    """
    return 1 - abs(classification_metric.average_odds_difference())


def disparate_impact(classification_metric):
    """
    Calculate Disparate Impact score: 1 - |1 - Average Odds Difference|
    Disparate Impact = 𝑃𝑟(𝑌̂ =1|𝐷=unprivileged)/𝑃𝑟(𝑌̂ =1|𝐷=privileged),
    which means that 1 is perfect fairness, but the score can also be above 1
    depending on which group is getting the best outcome,
    meaning it has to be scaled down to fit in range [0,1] where 1 is best.

    :param classification_metric: aif369 ClassificationMetric type containing the true, and the predicted data sets
    :return: Equal Opportunity Difference score
    """
    return 1 - abs(1 - classification_metric.disparate_impact())


def function_name_to_string(func):
    """
    Produces a readable string from a metric function. Update when adding more metrics.

    :param func: The metric function to get a readable string from
    :return: String with name of metric
    """
    if func == statistical_parity_difference:
        return "Statistical Parity Difference"
    if func == theil_index:
        return "Theil Index"
    if func == equal_opportunity_difference:
        return "Equal Opportunity Difference"
    if func == disparate_impact:
        return "Disparate Impact"
    if func == average_odds_difference:
        return "Average Odds Difference"
