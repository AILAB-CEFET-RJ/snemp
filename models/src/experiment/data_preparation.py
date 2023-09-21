import pandas as pd
import utils.nota_empenho as emp_utils
import logging

logging.getLogger()

path_tce_data = 'data/tce_ML.csv'
path_embeddings_historico = 'data/historico-word-embeddings-cls-state.parquet'
path_cd_list = 'data/cd_list.csv'

def get_empenho_data(path_tce_data):
    df_empenho = pd.read_csv(path_tce_data, sep=';')
    df_empenho[emp_utils.level_order] = df_empenho.apply(
        lambda row: emp_utils.cod_to_level(row.CG),
        axis='columns', 
        result_type='expand'
    )
    return df_empenho

def get_word_embeddings_for_historico_column_data(path_embeddings_historico):
    df_embeddings = pd.read_parquet(path_embeddings_historico)
    return df_embeddings

def get_data(min_observation_in_class=2):
    X = get_word_embeddings_for_historico_column_data(path_embeddings_historico)

    df_empenho = get_empenho_data(path_tce_data)
    y = df_empenho[emp_utils.level_order]
    y_cod = emp_utils.level_to_cod(y)

    # Filter Hierarchical with more than one observation
    qtd_cod = y_cod.value_counts()
    cod_more_than_one_observation = qtd_cod[(qtd_cod >= min_observation_in_class)].index
    filter_cod_index = y_cod.isin(cod_more_than_one_observation)

    X, y = X[filter_cod_index], y[filter_cod_index]
    return X, y

