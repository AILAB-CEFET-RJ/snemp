### Como instalar o ambiente conda?

1. A partir da raiz do projeto `snemp`. 
2. Com o conda instalado, instale as dependências:
   `conda env create --file classifier/environment.yml`
3. Ative o ambiente 
   `conda activate dl`


### O que você irá precisar?
- Ambiente conda com as dependências instaladas
- Arquivo contendo os embeddings da coluna histórico (`data/historico-word-embeddings-cls-state.parquet`)
- Arquivo contendo os códigos hierárquicos (`data/cd_list.csv`)
- Arquivo contendo os registros de nota de empenho (`data/tce_ML.csv`)  
  

*Obs:* Caso você não possua os embeddings, é possível gerá-los rodando o notebook `applying-bertimbau.ipynb`. Originalmente esse notebook foi executado em um ambiente virtual do Kaggle que possui 2 GPUs T4, porém qualquer ambiente com possua GPU CUDA deverá funcionar.

  
1. Entre na pasta com os códigos do classificador. (Você deverá estar em `snemp/classfier`)
   `cd classfifier`
2. Com o ambiente conda ativo, execute:
   `python simulation_pipeline.py`