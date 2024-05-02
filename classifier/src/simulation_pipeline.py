from experiment.data_preparation import get_data
from sklearn.model_selection import train_test_split
from experiment.experiment_pipeline import run_experiment

from sklearn.linear_model import LogisticRegression
from hiclass import LocalClassifierPerNode, LocalClassifierPerLevel
from xgboost import XGBClassifier

import itertools as it
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.NOTSET,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


base_models = {
    'LogReg': LogisticRegression(),
    'XGBoost': XGBClassifier()
}

hierarchical_classifiers = {
    'LCPN': LocalClassifierPerNode,
}

binary_policy = {
    'siblings':'siblings',
    'exclusive':'exclusive'
}


def get_experiment_names(*experiment_params):
    return list(it.product(*[param.keys() for param in experiment_params]))

def get_experiment_parameters(*experiment_params):
    return list(it.product(*[param.values() for param in experiment_params]))

def run_simulation(test_size=0.25, val_size=0.25):
    logging.info('Initializing simulation.')
    X, y = get_data()
    
    logging.info('Data loaded.')
    x_train_test, x_val, y_train_test, y_val = train_test_split(
        X, 
        y,
        test_size=val_size,
        random_state=2023
    )
    logging.info('Train and test divided.')
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_train_test,
    #     y_train_test,
    #     test_size=test_size, 
    #     random_state=2023
    # )

    experiment_names = get_experiment_names(base_models, hierarchical_classifiers)
    experiment_parameters = get_experiment_parameters(base_models, hierarchical_classifiers)
    logging.info('Got experiment parameters.')

    n_experiments = len(experiment_names)
    logging.info(f'Initializing {n_experiments} experiments..')
    experiment_metrics = []
    for names, parameters in tqdm(zip(experiment_names, experiment_parameters)):
        experiment_name = '+'.join(names)
        logging.info(f'Running experiment - {experiment_name}')
        model, metrics = run_experiment(
            x_train=x_train_test,
            y_train=y_train_test,
            x_test=x_val,
            y_test=y_val,   
            base_model=parameters[0],
            hierarchical_classifier=parameters[1]
        )
        experiment_metrics.append(
            (experiment_name, *metrics.values())
        )
    logging.info(f'Finished all experiments.')
    df_metrics = pd.DataFrame(experiment_metrics, columns=['experiment_name', *metrics.keys()])

    return df_metrics

if __name__ == '__main__':
    run_simulation()