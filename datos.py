import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import streamlit as st

# Aplicamos el decorador para congelar el resultado en la caché global de Streamlit
@st.cache_data(show_spinner=False)
def cargar_datos_desde_r():
    """
    Versión ETL Pura usando rpy2 optimizada con caché global.
    Evita la colisión de hilos concurrentes al almacenar el DataFrame en el búfer de Streamlit.
    """
    print("[ETL] Iniciando el motor embebido de R vía rpy2...")
    
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
    
    # 1. Compilamos e instanciamos la función de R
    funcion_r = robjects.r(codigo_r)
    
    # 2. Ejecutamos para obtener el objeto nativo
    dataframe_r = funcion_r()
    
    print("[ETL] Transfiriendo DataFrame nativo de R a Pandas...")
    from rpy2.robjects import conversion

    # Usamos la conversión estándar estructurada
    with conversion.localconverter(robjects.default_converter + pandas2ri.converter) as cv:
        df_crudo = conversion.rpy2py(dataframe_r)
        
    return df_crudo