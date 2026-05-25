import subprocess
import io
import shutil
import sys
import os
import pandas as pd

def cargar_datos_desde_r():
    """
    Versión portátil optimizada para solucionar el error de SSL Connect en Windows.
    Incluye búsqueda dinámica del motor de R para entornos colaborativos y Streamlit.
    """
    print("[ETL] Buscando el motor de R en el sistema...")
    
    # 1. Intentar buscar Rscript en las variables de entorno (PATH) - Vital para Linux/Streamlit Cloud
    r_binary = shutil.which("Rscript")
    
    # 2. Si no se encuentra en el PATH y es Windows, buscar dinámicamente en Archivos de Programa
    if not r_binary and sys.platform.startswith("win"):
        base_path = r"C:\Program Files\R"
        if os.path.exists(base_path):
            # Listar subcarpetas que empiecen por "R-" (ej: R-4.3.1, R-4.4.0)
            versiones = [f for f in os.listdir(base_path) if f.startswith("R-")]
            if versiones:
                # Ordenar para seleccionar la versión más reciente instalada en el equipo
                version_reciente = sorted(versiones)[-1]
                r_binary = os.path.join(base_path, version_reciente, "bin", "x64", "Rscript.exe")
    
    # 3. Control de seguridad definitivo si no se localiza por ninguna vía
    if not r_binary or not os.path.exists(r_binary if not shutil.which("Rscript") else r_binary):
        raise FileNotFoundError(
            "❌ No se ha podido localizar 'Rscript' automáticamente en el sistema.\n"
            "Asegúrese de que R esté instalado o de añadirlo a las variables de entorno (PATH)."
        )
            
    print(f"[ETL] Motor de R detectado en: {r_binary}")
    
    codigo_r = """
    # Desactivar alertas de instalación y forzar codificación nativa UTF-8
    options(warn = -1, encoding = "UTF-8", download.file.method = "wininet")
    
    # Carga silenciosa de la librería preinstalada por el sistema operativo
    suppressPackageStartupMessages(library(dplyr))
    
    # URL oficial del dataset
    url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Extracción segura con control de errores por si falla el protocolo HTTPS
    X <- tryCatch({
        read.table(url, header = FALSE, sep = " ", stringsAsFactors = FALSE)
    }, error = function(e) {
        tryCatch({
            url_http <- "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            read.table(url_http, header = FALSE, sep = " ", stringsAsFactors = FALSE)
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
    
    # Transformaciones estructuradas con case_when para evitar mensajes de advertencia
    X <- X %>% mutate(
        checking_status = case_when(
          checking_status == "A11" ~ "< 0 DM",
          checking_status == "A12" ~ "0-200 DM",
          checking_status == "A13" ~ ">= 200 DM",
          checking_status == "A14" ~ "no checking",
          TRUE ~ checking_status
        ),
        credit_history = case_when(
          credit_history == "A30" ~ "no credits",
          credit_history == "A31" ~ "all paid duly",
          credit_history == "A32" ~ "existing paid",
          credit_history == "A33" ~ "past delay",
          credit_history == "A34" ~ "critical account",
          TRUE ~ credit_history
        ),
        purpose = case_when(
          purpose == "A40" ~ "car (new)",
          purpose == "A41" ~ "car (used)",
          purpose == "A42" ~ "furniture/equipment",
          purpose == "A43" ~ "radio/television",
          purpose == "A44" ~ "domestic appliances",
          purpose == "A45" ~ "repairs",
          purpose == "A46" ~ "education",
          purpose == "A47" ~ "vacation",
          purpose == "A48" ~ "retraining",
          purpose == "A49" ~ "business",
          purpose == "A410" ~ "others",
          TRUE ~ purpose
        ),
        savings_status = case_when(
          savings_status == "A61" ~ "< 100 DM",
          savings_status == "A62" ~ "100-500 DM",
          savings_status == "A63" ~ "500-1000 DM",
          savings_status == "A64" ~ ">= 1000 DM",
          savings_status == "A65" ~ "no savings",
          TRUE ~ savings_status
        ),
        employment_since = case_when(
          employment_since == "A71" ~ "unemployed",
          employment_since == "A72" ~ "< 1 year",
          employment_since == "A73" ~ "1-4 years",
          employment_since == "A74" ~ "4-7 years",
          employment_since == "A75" ~ ">= 7 years",
          TRUE ~ employment_since
        ),
        personal_status = case_when(
          personal_status == "A91" ~ "male: divorced/sep",
          personal_status == "A92" ~ "female: div/sep/mar",
          personal_status == "A93" ~ "male: single",
          personal_status == "A94" ~ "male: mar/wid",
          personal_status == "A95" ~ "female: single",
          TRUE ~ personal_status
        ),
        other_debtors = case_when(
          other_debtors == "A101" ~ "none",
          other_debtors == "A102" ~ "co-applicant",
          other_debtors == "A103" ~ "guarantor",
          TRUE ~ other_debtors
        ),
        property_type = case_when(
          property_type == "A121" ~ "real estate",
          property_type == "A122" ~ "life insurance",
          property_type == "A123" ~ "car/other",
          property_type == "A124" ~ "no property",
          TRUE ~ property_type
        ),
        installment_plans = case_when(
          installment_plans == "A141" ~ "bank",
          installment_plans == "A142" ~ "stores",
          installment_plans == "A143" ~ "none",
          TRUE ~ installment_plans
        ),
        housing_type = case_when(
          housing_type == "A151" ~ "rent",
          housing_type == "A152" ~ "own",
          housing_type == "A153" ~ "for free",
          TRUE ~ housing_type
        ),
        job_type = case_when(
          job_type == "A171" ~ "unemployed non-res",
          job_type == "A172" ~ "unskilled res",
          job_type == "A173" ~ "skilled official",
          job_type == "A174" ~ "mgmt/highly qualif",
          TRUE ~ job_type
        ),
        telephone = case_when(
          telephone == "A191" ~ "none",
          telephone == "A192" ~ "yes",
          TRUE ~ telephone
        ),
        foreign_worker = case_when(
          foreign_worker == "A201" ~ "yes",
          foreign_worker == "A202" ~ "no",
          TRUE ~ foreign_worker
        )
    )
    
    # Exportar los datos procesados directamente a la salida estándar en formato CSV
    write.csv(X, stdout(), row.names = FALSE, quote = TRUE)
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
        
        # Tipado de columnas para asegurar compatibilidad con el pipeline de entrenamiento
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