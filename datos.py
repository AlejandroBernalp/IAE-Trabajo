# Importamos la interfaz rob_objects para código estructurado de R y el conversor nativo
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

def cargar_datos_desde_r():
    """
    Versión ETL Pura usando rpy2. 
    Se ejecuta el motor de R embebido en el espacio de memoria de Python
    para descargar el dataset desde UCI y estructurar las columnas.
    El DataFrame resultante se convierte directamente de R a Pandas sin tocar el disco.
    """
    print("[ETL] Iniciando el motor embebido de R vía rpy2...")
    
    # Definimos la lógica de descarga estructurada en código puro de R
    codigo_r = """
    function() {
        options(warn = -1, encoding = "UTF-8")
        url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        datos <- read.table(url, header = FALSE, sep = " ", stringsAsFactors = FALSE)
        
        columnas <- c("checking_status", "duration_months", "credit_history", "purpose", 
                      "credit_amount", "savings_status", "employment_since", "installment_rate", 
                      "personal_status", "other_debtors", "residence_since", "property_type", 
                      "age", "installment_plans", "housing_type", "existing_credits", 
                      "job_type", "dependents", "telephone", "foreign_worker", "class")
        names(datos) <- columnas
        return(datos)
    }
    """
    
    print("[ETL] Solicitando descarga cruda a R mediante ejecución integrada...")
    # 1. Compilamos e instanciamos la función de R en el entorno global de rpy2
    funcion_r = robjects.r(codigo_r)
    
    # 2. Ejecutamos la función de R para obtener el data.frame en formato nativo de R
    dataframe_r = funcion_r()
    
    print("[ETL] Transfiriendo DataFrame nativo de R a Pandas...")
    # 3. Usamos el localconverter oficial de rpy2 para mapear R a Pandas de forma segura
    with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
        df_crudo = robjects.conversion.rpy2py(dataframe_r)
        
    return df_crudo