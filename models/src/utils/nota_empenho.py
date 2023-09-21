level_order = ['categoria', 'grupo', 'modalidade', 'elemento', 'subelemento']

def cod_to_level(cod):
    cod = str(cod)
    categoria = cod[0]
    grupo = cod[1]
    modalidade = cod[2:4]
    elemento = cod[4:6]
    subelemento = cod[6:]
    
    return categoria, grupo, modalidade, elemento, subelemento

def level_to_cod(df_levels):
    return df_levels.sum(axis=1).astype(str).str.rstrip('.0')