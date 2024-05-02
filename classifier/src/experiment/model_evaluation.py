import pandas as pd
from utils.nota_empenho import level_order, level_to_cod
from sklearn.metrics import f1_score, precision_score, recall_score
from hiclass.metrics import f1, precision, recall

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    hierarchical_metrics = get_hierarchical_metrics(y_test, y_pred)
    flat_metrics = get_flat_metrics(y_test, y_pred)
    metrics = {
        **flat_metrics,
        **hierarchical_metrics
    }
    return metrics

def get_flat_metrics(y_test_level, y_pred_level):
    flat_metrics = {}
    df_y_test = pd.DataFrame(y_test_level, columns=level_order)
    df_y_pred = pd.DataFrame(y_pred_level, columns=level_order)

    y_test = level_to_cod(df_y_test).reset_index(drop=True)
    y_pred = level_to_cod(df_y_pred).reset_index(drop=True)

    flat_metrics['f1'] = f1_score(y_test, y_pred, average='weighted'),
    flat_metrics['precision'] = precision_score(y_test, y_pred, average='weighted'),
    flat_metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
    return flat_metrics

def get_hierarchical_metrics(y_test, y_pred):
    hierarchical_metrics = {}

    hierarchical_metrics['h_f1'] = f1(y_test, y_pred)
    hierarchical_metrics['h_precision'] = precision(y_test, y_pred)
    hierarchical_metrics['h_recall'] = recall(y_test, y_pred)
    return hierarchical_metrics