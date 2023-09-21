from experiment.model_training import fit_model
from experiment.model_evaluation import evaluate_model

from sklearn.linear_model import LogisticRegression
from hiclass import LocalClassifierPerNode

import logging
logging.getLogger()

def unify_metric_dict(metrics):
    metrics_unified = {}
    for subset_name, metrics in metrics.items():
        for metric_name, metric_value in metrics.items():
            metrics_unified[subset_name+'__'+metric_name] = metric_value 

    return metrics_unified

def run_experiment(x_train, x_test, y_train, y_test, base_model=LogisticRegression, hierarchical_classifier=LocalClassifierPerNode, binary_policy=None):
    model = fit_model(
        x_train=x_train,
        y_train=y_train,
        base_model=base_model, 
        hierarchical_classifier=hierarchical_classifier, 
        binary_policy=binary_policy
    )

    metrics_per_subset = {}
    metrics_per_subset['train'] = evaluate_model(model, x_train, y_train)
    metrics_per_subset['test'] = evaluate_model(model, x_test, y_test)

    metrics = unify_metric_dict(metrics_per_subset)
    return model, metrics

if __name__ == '__main__':
    run_experiment()