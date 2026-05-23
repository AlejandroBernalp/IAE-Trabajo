import subprocess
import io
import shutil
import sys
import os
import pandas as pd

def cargar_datos_desde_r():
    """
    Versión portátil optimizada para solucionar el error de SSL Connect en Windows.
    """
    print("[ETL] Buscando el motor de R en el sistema...")
    r_binary = shutil.which("Rscript")
    
    if not r_binary:
        if sys.platform.startswith("win"):
            r_binary = r"C:\Program Files\R\R-4.3.1\bin\x64\Rscript.exe"
        else:
            raise FileNotFoundError("No se encontró 'Rscript'.")
            
    print(f"[ETL] Motor de R detectado en: {r_binary}")
    
    codigo_r = """
    # 1. FORZAR MÉTODO DE DESCARGA SEGURO PARA EVITAR 'SSL CONNECT ERROR' EN WINDOWS
    options(download.file.method = "wininet", keep.source = TRUE, show.error.messages = TRUE)
    
    if (!require("dplyr", quietly = TRUE)) {
        install.packages("dplyr", repos="http://cran.us.r-project.org", quiet = TRUE)
        library(dplyr)
    }
    
    # Usamos la URL oficial
    url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Forzamos la descarga usando el canal nativo de internet de Windows
    X <- tryCatch({
        read.table(url, header = FALSE, sep = " ")
    }, error = function(e) {
        # Si aun así fallara el https por políticas restrictivas, intentamos el fallback por http ordinario
        tryCatch({
            url_http <- "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            read.table(url_http, header = FALSE, sep = " ")
        }, error = function(e2) {
            stop(paste("Error de conexión en R. Revisa tu antivirus/firewall:", e2$message))
        })
    })
    
    nuevos_nombres <- c(
      "checking_status", "duration_months", "credit_history", "purpose", 
      "credit_amount", "savings_status", "employment_since", "installment_rate", 
      "personal_status", "other_debtors", "residence_since", "property_type", 
      "age", "installment_plans", "housing_type", "existing_credits", 
      "job_type", "dependents", "telephone", "foreign_worker", "class"
    )
    names(X) <- nuevos_nombres
    
    X <- X %>% mutate(
        checking_status = recode(checking_status, "A11" = "< 0 DM", "A12" = "0-200 DM", "A13" = ">= 200 DM", "A14" = "no checking"),
        credit_history = recode(credit_history, "A30" = "all paid duly", "A31" = "all paid bank", "A32" = "existing paid", "A33" = "past delay", "A34" = "critical account"),
        purpose = recode(purpose, "A40" = "car (new)", "A41" = "car (used)", "A42" = "furniture/equipment", "A43" = "radio/television", "A44" = "domestic appliances", "A45" = "repairs", "A46" = "education", "A47" = "vacation", "A48" = "retraining", "A49" = "business", "A410" = "others"),
        savings_status = recode(savings_status, "A61" = "< 100 DM", "A62" = "100-500 DM", "A63" = "500-1000 DM", "A64" = ">= 1000 DM", "A65" = "no savings"),
        employment_since = recode(employment_since, "A71" = "unemployed", "A72" = "< 1 year", "A73" = "1-4 years", "A74" = "4-7 years", "A75" = ">= 7 years"),
        personal_status = recode(personal_status, "A91" = "male: divorced/sep", "A92" = "female: div/sep/mar", "A93" = "male: single", "A94" = "male: mar/wid", "A95" = "female: single"),
        other_debtors = recode(other_debtors, "A101" = "none", "A102" = "co-applicant", "A103" = "guarantor"),
        property_type = recode(property_type, "A121" = "real estate", "A122" = "life insurance", "A123" = "car/other", "A124" = "no property"),
        installment_plans = recode(installment_plans, "A141" = "bank", "A142" = "stores", "A143" = "none"),
        housing_type = recode(housing_type, "A151" = "rent", "A152" = "own", "A153" = "for free"),
        job_type = recode(job_type, "A171" = "unskilled non-res", "A172" = "unskilled res", "A173" = "skilled official", "A174" = "mgmt/highly qualif"),
        telephone = recode(telephone, "A191" = "none", "A192" = "yes"),
        foreign_worker = recode(foreign_worker, "A201" = "yes", "A202" = "no")
    )
    write.csv(X, stdout(), row.names = FALSE)
    """
    
    temp_script = "temp_etl_script.R"
    
    try:
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(codigo_r)
            
        print("[ETL] Ejecutando motor de R mediante script puente...")
        proceso = subprocess.run(
            [r_binary, temp_script],
            capture_output=True,
            text=True
        )
        
        if proceso.returncode != 0:
            print("\n❌ [ERROR CRÍTICO DENTRO DE R] El script de R ha fallado con el siguiente mensaje:")
            print("-" * 60)
            print(proceso.stderr)
            print("-" * 60)
            sys.exit(1)
        
        print("[ETL] Transfiriendo flujo de datos a Pandas...")
        df_pandas = pd.read_csv(io.StringIO(proceso.stdout))
        
        # Corrección del Warning: Incluimos tanto 'object' como 'string' de forma explícita
        columnas_texto = df_pandas.select_dtypes(include=['object', 'string']).columns
        df_pandas[columnas_texto] = df_pandas[columnas_texto].astype('category')
        df_pandas['class'] = df_pandas['class'].astype('category')
        
        print("[ETL] Extracción y tipado completados con éxito.")
        return df_pandas

    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    print("--- INICIANDO TEST DE CONTROL ---")
    df = cargar_datos_desde_r()
    print(f"\n¡Éxito absoluto! Registros totales cargados en Python: {len(df)}")
    print(df.head(2))