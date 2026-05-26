import subprocess
import io
import shutil
import sys
import os
import pandas as pd

def cargar_datos_desde_r():
    """
    Versión ETL Pura. R solo descarga el dataset crudo desde UCI 
    y asigna los nombres de las columnas originales. 
    Todo el preprocesamiento de strings se delega en Python.
    """
    print("[ETL] Buscando el motor de R en el sistema...")
    r_binary = shutil.which("Rscript")
    
    if not r_binary and sys.platform.startswith("win"):
        base_path = r"C:\Program Files\R"
        if os.path.exists(base_path):
            versiones = [f for f in os.listdir(base_path) if f.startswith("R-")]
            if versiones:
                version_reciente = sorted(versiones)[-1]
                r_binary = os.path.join(base_path, version_reciente, "bin", "x64", "Rscript.exe")
    
    if not r_binary:
        raise FileNotFoundError("❌ No se ha podido localizar 'Rscript' automáticamente.")
            
    print(f"[ETL] Motor de R detectado en: {r_binary}")
    
    # R descarga y nombra las columnas, pero NO procesa los niveles (A11, A12...)
    codigo_r = """
    options(warn = -1, encoding = "UTF-8")
    url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    datos <- read.table(url, header = FALSE, sep = " ", stringsAsFactors = FALSE)
    
    columnas <- c("checking_status", "duration_months", "credit_history", "purpose", 
                  "credit_amount", "savings_status", "employment_since", "installment_rate", 
                  "personal_status", "other_debtors", "residence_since", "property_type", 
                  "age", "installment_plans", "housing_type", "existing_credits", 
                  "job_type", "dependents", "telephone", "foreign_worker", "class")
    names(datos) <- columnas
    
    write.csv(datos, stdout(), row.names = FALSE, quote = TRUE)
    """
    
    temp_script = "temp_etl_script.R"
    try:
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(codigo_r)
            
        print("[ETL] Solicitando descarga cruda a R...")
        proceso = subprocess.run([r_binary, temp_script], capture_output=True, text=True)
        
        if proceso.returncode != 0:
            print("\n❌ [ERROR CRÍTICO DENTRO DE R]:", proceso.stderr)
            sys.exit(1)
        
        print("[ETL] Transfiriendo DataFrame crudo a Python...")
        df_crudo = pd.read_csv(io.StringIO(proceso.stdout))
        return df_crudo

    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)