import streamlit as st
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from xgboost import XGBClassifier
import io

# Importación del módulo de datos local (Extractor Crudo)
from datos import cargar_datos_desde_r

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Statlog (German Credit Data)", page_icon="🏦", layout="wide")

# Diccionarios de mapeo idénticos a las reglas estructuradas del script original de R
MAPEOS_GERMAN_CREDIT = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking"},
    "credit_history": {"A30": "no credits", "A31": "all paid duly", "A32": "existing paid", "A33": "past delay", "A34": "critical account"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "others"},
    "savings_status": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "no savings"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status": {"A91": "male: divorced/sep", "A92": "female: div/sep/mar", "A93": "male: single", "A94": "male: mar/wid", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property_type": {"A121": "real estate", "A122": "life insurance", "A123": "car/other", "A124": "no property"},
    "installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing_type": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job_type": {"A171": "unskilled non-res", "A172": "unskilled res", "A173": "skilled official", "A174": "mgmt/highly qualif"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"}
}

# ==========================================
# FUNCIONES DE PREPROCESAMIENTO COGNITIVO
# ==========================================
def preprocesar_con_pandas(df_crudo):
    df = df_crudo.copy()
    for columna, mapa in MAPEOS_GERMAN_CREDIT.items():
        df[columna] = df[columna].replace(mapa)
    
    columnas_texto = df.select_dtypes(include=['object', 'string']).columns
    df[columnas_texto] = df[columnas_texto].astype('category')
    df['class'] = df['class'].astype('category')
    return df

def preprocesar_con_dask(df_crudo):
    # Convertimos a colección Dask particionando para forzar cómputo paralelo en bloques
    ddf = dd.from_pandas(df_crudo, npartitions=4)
    
    # Reemplazo perezoso estructurado (Lazy Evaluation)
    for columna, mapa in MAPEOS_GERMAN_CREDIT.items():
        ddf[columna] = ddf[columna].replace(mapa)
        
    # Cómputo usando el planificador multinúcleo por hilos (Inofensivo para los límites de Streamlit)
    df_procesado = ddf.compute(scheduler='threads')
    
    columnas_texto = df_procesado.select_dtypes(include=['object', 'string']).columns
    df_procesado[columnas_texto] = df_procesado[columnas_texto].astype('category')
    df_procesado['class'] = df_procesado['class'].astype('category')
    return df_procesado

# ==========================================
# FUNCIONES AUXILIARES Y COSTE
# ==========================================
def calcular_coste_financiero(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        fp = cm[0, 1]  # Cliente bueno clasificado como malo (Coste 1)
        fn = cm[1, 0]  # Cliente malo clasificado como bueno (Coste 5)
        return (fp * 1) + (fn * 5)
    return 9999

# ==========================================
# MOTOR CENTRAL DE CARGA E INGENIERÍA DE DATOS
# ==========================================
if 'df_data' not in st.session_state:
    with st.spinner("⏳ Extrayendo dataset crudo y evaluando motores de procesamiento (Pandas vs Dask)..."):
        try:
            # 1. Llamada a R para descargar el CSV crudo desde UCI
            df_inicial = cargar_datos_desde_r()
            st.session_state['df_crudo'] = df_inicial
            
            # 2. Benchmark de ingeniería de características: Pandas
            start_p = time.time()
            df_pandas_final = preprocesar_con_pandas(df_inicial)
            tiempo_pandas = time.time() - start_p
            
            # 3. Benchmark de ingeniería de características: Dask DataFrame
            start_d = time.time()
            df_dask_final = preprocesar_con_dask(df_inicial)
            tiempo_dask = time.time() - start_d
            
            # Almacenamiento en sesión
            st.session_state['df_data'] = df_pandas_final  # Dataset normalizado para la UI
            st.session_state['t_pandas_etl'] = tiempo_pandas
            st.session_state['t_dask_etl'] = tiempo_dask
            
        except Exception as e:
            st.error(f"Error crítico en el pipeline de datos: {e}")
            st.stop()

# ==========================================
# INTERFAZ DE USUARIO - BARRA LATERAL
# ==========================================
st.sidebar.title("Navegación")
st.sidebar.markdown("Seleccione una de las opciones para explorar los datos, entrenar el modelo o realizar predicciones.")
opcion_menu = st.sidebar.radio(
    "Menú",
    ["Análisis Exploratorio", "Entrenamiento del Modelo", "Predicción de Crédito"]
)

# Panel de Métricas ETL integrado en la barra lateral para seguimiento académico
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Rendimiento ETL (Preprocesamiento)")
st.sidebar.metric(label="Tiempo con Pandas", value=f"{st.session_state['t_pandas_etl']:.5f} s")
st.sidebar.metric(label="Tiempo con Dask DF", value=f"{st.session_state['t_dask_etl']:.5f} s")

# Cálculo de la tasa de sobrecarga estructural
overhead_etl = ((st.session_state['t_dask_etl'] - st.session_state['t_pandas_etl']) / st.session_state['t_pandas_etl']) * 100
st.sidebar.caption(f"Dask presenta un *overhead* del `{overhead_etl:.1f}%` debido al fraccionamiento de grafos en colecciones pequeñas.")

df_global = st.session_state['df_data']

# ==========================================
# PESTAÑA 1: ANÁLISIS EXPLORATORIO
# ==========================================
if opcion_menu == "Análisis Exploratorio":
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("Se presentan las distribuciones de los distintos atributos financieros y demográficos de los clientes.")
    st.success("✅ ¡Datos cargados desde R y preprocesados en Python de forma paralela!")

    col1, col2 = st.columns([1, 3])
    with col1:
        variables_disponibles = [col for col in df_global.columns if col != 'class']
        var_seleccionada = st.selectbox("Seleccione la variable a visualizar:", variables_disponibles)
    
    with col2:
        st.subheader(f"Distribución de: {var_seleccionada}")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if df_global[var_seleccionada].dtype.name == 'category' or df_global[var_seleccionada].dtype == 'object':
            sns.countplot(data=df_global, x=var_seleccionada, hue='class', palette='Set2', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Número de clientes")
        else:
            sns.histplot(data=df_global, x=var_seleccionada, hue='class', multiple="stack", palette='Set2', kde=True, ax=ax)
            plt.ylabel("Frecuencia")
            
        plt.xlabel(var_seleccionada)
        sns.despine()
        st.pyplot(fig)

# ==========================================
# PESTAÑA 2: ENTRENAMIENTO DEL MODELO
# ==========================================
elif opcion_menu == "Entrenamiento del Modelo":
    st.title("⚙️ Optimización y Entrenamiento")
    st.markdown("En esta sección se entrena la malla de modelos mediante *GridSearchCV* utilizando una función de coste financiero.")

    # Control de estado de entrenamiento en la sesión
    if 'entrenamiento_realizado' not in st.session_state:
        st.session_state['entrenamiento_realizado'] = False

    # SI NO SE HA ENTRENADO AÚN: Mostrar el botón de inicio
    if not st.session_state['entrenamiento_realizado']:
        if st.button("🚀 Iniciar Entrenamiento (GridSearch)"):
            with st.spinner("Entrenando macro-malla de algoritmos. Este proceso puede tardar unos minutos..."):
                
                nominal_cols = ['credit_history', 'purpose', 'personal_status', 'other_debtors', 'property_type', 'installment_plans', 'housing_type']
                ordinal_cols = ['checking_status', 'savings_status', 'employment_since', 'job_type']
                binary_cols = ['telephone', 'foreign_worker']

                ordinal_order = [
                    ['no checking', '< 0 DM', '0-200 DM', '>= 200 DM'], 
                    ['no savings', '< 100 DM', '100-500 DM', '500-1000 DM', '>= 1000 DM'], 
                    ['unemployed', '< 1 year', '1-4 years', '4-7 years', '>= 7 years'], 
                    ['unskilled non-res', 'unskilled res', 'skilled official', 'mgmt/highly qualif']
                ]

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('ord', OrdinalEncoder(categories=ordinal_order), ordinal_cols),
                        ('nom', OneHotEncoder(drop='first', sparse_output=False), nominal_cols),
                        ('bin', OrdinalEncoder(), binary_cols)
                    ],
                    remainder='passthrough'
                )

                X = df_global.drop(columns=['class'])
                y = df_global['class'].astype(int).replace({1: 0, 2: 1})

                X_processed = preprocessor.fit_transform(X)
                cols_names = preprocessor.get_feature_names_out()
                df_final = pd.DataFrame(X_processed, columns=cols_names).apply(pd.to_numeric)

                X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42, stratify=y)
                cost_scorer = make_scorer(calcular_coste_financiero, greater_is_better=False)

                from sklearn.pipeline import Pipeline
                
                config_modelos = {
                    "Logística": {
                        "model": LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42, max_iter=2000),
                        "params": {
                            "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 50]
                        }
                    },
                    "SVM": {
                        "model": SVC(class_weight={0: 1, 1: 5}, random_state=42),
                        "params": {
                            "clf__C": [0.01, 0.1, 1, 5, 10, 50],
                            "clf__kernel": ['linear', 'rbf'],
                            "clf__gamma": ['scale', 'auto', 0.01, 0.1]
                        }
                    },
                    "Random Forest": {
                        "model": RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
                        "params": {
                            "clf__n_estimators": [100, 200, 400],
                            "clf__max_depth": [4, 6, 8, 12, None],
                            "clf__min_samples_leaf": [5, 10, 15, 20],
                            "clf__criterion": ["gini", "entropy"]
                        }
                    },
                    "XGBoost": {
                        "model": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
                        "params": {
                            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "clf__max_depth": [3, 4, 6, 8],
                            "clf__n_estimators": [100, 200, 300],
                            "clf__subsample": [0.8, 1.0],
                            "clf__scale_pos_weight": [5]
                        }
                    }
                }

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                modelos_optimizados = {}
                resultados_test = {}
                resultados_cv = []

                for nombre, config in config_modelos.items():
                    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', config["model"])])
                    grid = GridSearchCV(estimator=pipeline, param_grid=config["params"], scoring=cost_scorer, cv=skf, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    
                    mejor_modelo = grid.best_estimator_
                    modelos_optimizados[nombre] = mejor_modelo
                    
                    y_pred = mejor_modelo.predict(X_test)
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    acc = rep['accuracy']
                    rec = rep['1']['recall']
                    f1 = rep['1']['f1-score']
                    coste = calcular_coste_financiero(y_test, y_pred)
                    
                    resultados_test[nombre] = coste
                    mejores_params_limpios = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}
                    
                    resultados_cv.append({
                        "Algoritmo": nombre, 
                        "Parámetros Óptimos": str(mejores_params_limpios),
                        "Accuracy": acc,
                        "Recall (Malos)": rec,
                        "F1-Score": f1,
                        "Coste Financiero": coste
                    })

                df_resultados = pd.DataFrame(resultados_cv)
                ganador = min(resultados_test, key=resultados_test.get)
                
                # Extracción de importancia de variables del ganador
                final_clf = modelos_optimizados[ganador].named_steps['clf']
                if hasattr(final_clf, 'coef_'):
                    importancia = final_clf.coef_.flatten()
                elif hasattr(final_clf, 'feature_importances_'):
                    importancia = final_clf.feature_importances_
                else:
                    importancia = None

                # Guardar absolutamente todo el estado del entrenamiento en st.session_state
                st.session_state['preprocessor'] = preprocessor
                st.session_state['cols_names'] = cols_names
                st.session_state['best_model'] = modelos_optimizados[ganador]
                st.session_state['best_model_name'] = ganador
                st.session_state['df_resultados'] = df_resultados
                st.session_state['resultados_test'] = resultados_test
                st.session_state['importancia_ganador'] = importancia
                st.session_state['df_final_cols'] = df_final.columns
                
                # Marcar como completado para congelar la pestaña
                st.session_state['entrenamiento_realizado'] = True
                st.rerun()

    # SI YA HA SIDO ENTRENADO ANTES: Recuperar los objetos y renderizarlos directamente
    else:
        ganador = st.session_state['best_model_name']
        coste_ganador = st.session_state['resultados_test'][ganador]
        df_resultados = st.session_state['df_resultados']
        importancia = st.session_state['importancia_ganador']
        df_final_cols = st.session_state['df_final_cols']

        # Cabecera de éxito persistente con botón de reinicio al lado
        c_info, c_btn = st.columns([4, 1])
        with c_info:
            st.success(f"🎉 El entrenamiento está activo. El mejor modelo recomendado es: **{ganador}** (Coste: {coste_ganador})")
        with c_btn:
            if st.button("🔄 Volver a entrenar", use_container_width=True):
                st.session_state['entrenamiento_realizado'] = False
                # Limpieza opcional para evitar residuos en memoria
                del st.session_state['best_model']
                del st.session_state['best_model_name']
                st.rerun()

        # Mostrar tabla de resultados estática
        st.dataframe(df_resultados.style.highlight_min(subset=['Coste Financiero'], color='lightgreen'))

        # Renderizar Gráfico de Métricas de Test
        st.subheader("Comparativa de Rendimiento en Test")
        df_plot = df_resultados.melt(id_vars=["Algoritmo"], value_vars=["Accuracy", "Recall (Malos)", "F1-Score"], var_name="Métrica", value_name="Valor")
        fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_plot, x='Algoritmo', y='Valor', hue='Métrica', palette={"Accuracy": "#7FB3D5", "Recall (Malos)": "#E74C3C", "F1-Score": "#F39C12"}, ax=ax_perf)
        plt.ylim(0, 1.1)
        sns.despine()
        st.pyplot(fig_perf)

        # Renderizar Gráfico de Importancia de Variables
        st.subheader(f"Importancia de Variables ({ganador})")
        if importancia is not None:
            importance_df = pd.DataFrame({'Feature': df_final_cols, 'Importance': importancia})
            importance_df['Abs_Importance'] = importance_df['Importance'].abs()
            importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False).head(10)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='crest', legend=False, ax=ax_imp)
            plt.axvline(0, color='black', lw=1)
            sns.despine()
            st.pyplot(fig_imp)
        else:
            st.info("Este algoritmo no expone métricas nativas de importancia de variables.")

# ==========================================
# PESTAÑA 3: PREDICCIÓN DE CRÉDITO
# ==========================================
elif opcion_menu == "Predicción de Crédito":
    st.title("🔮 Predicción de Riesgo Crediticio")
    
    if 'best_model' not in st.session_state:
        st.warning("⚠️ El modelo no ha sido entrenado. Por favor, acuda a la pestaña 'Entrenamiento del Modelo' y ejecute la optimización primero.")
        st.stop()

    st.info(f"Modelo activo en producción: **{st.session_state['best_model_name']}**")
    st.markdown("Se debe introducir la información del cliente para calcular si se aprueba o deniega la solicitud de crédito.")

    with st.form("form_prediccion"):
        st.subheader("Datos de la Solicitud y Financieros")
        c1, c2, c3 = st.columns(3)
        checking_status = c1.selectbox("Estado cuenta corriente (checking_status)", ["no checking", "< 0 DM", "0-200 DM", ">= 200 DM"])
        savings_status = c2.selectbox("Estado de ahorros (savings_status)", ["no savings", "< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM"])
        credit_amount = c3.number_input("Cantidad de crédito solicitada (DM)", min_value=100, max_value=20000, value=2500)
        
        duration_months = c1.slider("Duración del préstamo (meses)", min_value=4, max_value=72, value=24)
        installment_rate = c2.slider("Tasa de pago a plazos (%)", min_value=1, max_value=4, value=2)
        existing_credits = c3.slider("Créditos existentes en el banco", min_value=1, max_value=4, value=1)
        
        credit_history = c1.selectbox("Historial crediticio", ["all paid duly", "all paid bank", "existing paid", "past delay", "critical account"])
        purpose = c2.selectbox("Propósito", ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs", "education", "vacation", "retraining", "business", "others"])
        installment_plans = c3.selectbox("Otros planes de cuotas", ["bank", "stores", "none"])

        st.subheader("Datos Demográficos y Laborales")
        c4, c5, c6 = st.columns(3)
        age = c4.slider("Edad (años)", min_value=18, max_value=80, value=35)
        personal_status = c5.selectbox("Estado civil y sexo", ["male: divorced/sep", "female: div/sep/mar", "male: single", "male: mar/wid", "female: single"])
        employment_since = c6.selectbox("Empleado desde hace", ["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"])
        
        job_type = c4.selectbox("Tipo de empleo", ["unskilled non-res", "unskilled res", "skilled official", "mgmt/highly qualif"])
        property_type = f = c5.selectbox("Propiedad", ["real estate", "life insurance", "car/other", "no property"])
        housing_type = c6.selectbox("Tipo de vivienda", ["rent", "own", "for free"])

        other_debtors = c4.selectbox("Otros deudores/Garante", ["none", "co-applicant", "guarantor"])
        residence_since = c5.slider("Tiempo en residencia actual (años)", min_value=1, max_value=4, value=2)
        dependents = c6.slider("Personas a cargo", min_value=1, max_value=2, value=1)
        
        telephone = c4.selectbox("Teléfono a su nombre", ["none", "yes"])
        foreign_worker = c5.selectbox("Trabajador extranjero", ["yes", "no"])

        submit_button = st.form_submit_button(label="🔍 Evaluar Riesgo")

    if submit_button:
        datos_cliente = {
            'checking_status': checking_status,
            'duration_months': duration_months,
            'credit_history': credit_history,
            'purpose': purpose,
            'credit_amount': credit_amount,
            'savings_status': savings_status,
            'employment_since': employment_since,
            'installment_rate': installment_rate,
            'personal_status': personal_status,
            'other_debtors': other_debtors,
            'residence_since': residence_since,
            'property_type': property_type,
            'age': age,
            'installment_plans': installment_plans,
            'housing_type': housing_type,
            'existing_credits': existing_credits,
            'job_type': job_type,
            'dependents': dependents,
            'telephone': telephone,
            'foreign_worker': foreign_worker
        }
        
        df_input = pd.DataFrame([datos_cliente])
        df_input = df_input.astype({col: 'category' for col in df_input.select_dtypes(['object', 'string']).columns})
        
        preprocessor = st.session_state['preprocessor']
        X_pred_proc = preprocessor.transform(df_input)
        df_pred_final = pd.DataFrame(X_pred_proc, columns=st.session_state['cols_names']).apply(pd.to_numeric)
        
        modelo = st.session_state['best_model']
        prediccion = modelo.predict(df_pred_final)[0]
        
        st.markdown("---")
        if prediccion == 0:
            st.success("✅ **CRÉDITO APROBADO:** El modelo clasifica al cliente como de BAJO RIESGO (Cliente Bueno).")
        else:
            st.error("❌ **CRÉDITO DENEGADO:** El modelo clasifica al cliente como de ALTO RIESGO (Posible Morosidad).")