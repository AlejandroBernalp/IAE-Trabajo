import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from xgboost import XGBClassifier
import time

# Importación del módulo de datos local
from datos import cargar_datos_desde_r

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Statlog (German Credit Data)", page_icon="🏦", layout="wide")

# ==========================================
# FUNCIONES CACHEADAS (Optimización)
# ==========================================
@st.cache_data(show_spinner="Extrayendo datos desde R...")
def obtener_datos():
    """Ejecuta el ETL solo una vez y guarda el DataFrame en caché."""
    df = cargar_datos_desde_r()
    df = df.astype({col: 'category' for col in df.select_dtypes(['object', 'string']).columns})
    df['class'] = df['class'].astype('category')
    return df

def calcular_coste_financiero(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        fp = cm[0, 1]  # Cliente bueno clasificado como malo (Coste 1)
        fn = cm[1, 0]  # Cliente malo clasificado como bueno (Coste 5)
        return (fp * 1) + (fn * 5)
    return 9999

# ==========================================
# INTERFAZ DE USUARIO - BARRA LATERAL
# ==========================================
st.sidebar.title("Navegación")
st.sidebar.markdown("Seleccione una de las opciones para explorar los datos, entrenar el modelo o realizar predicciones.")
opcion_menu = st.sidebar.radio(
    "Menú",
    ["Análisis Exploratorio", "Entrenamiento del Modelo", "Predicción de Crédito"]
)

# Carga de datos globales
try:
    df_global = obtener_datos()
except Exception as e:
    st.error(f"Error crítico al cargar los datos: {e}")
    st.stop()

# ==========================================
# PESTAÑA 1: ANÁLISIS EXPLORATORIO
# ==========================================
if opcion_menu == "Análisis Exploratorio":
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("Se presentan las distribuciones de los distintos atributos financieros y demográficos de los clientes.")

    col1, col2 = st.columns([1, 3])
    with col1:
        # Menú desplegable dinámico con las variables del dataset
        variables_disponibles = [col for col in df_global.columns if col != 'class']
        var_seleccionada = st.selectbox("Seleccione la variable a visualizar:", variables_disponibles)
    
    with col2:
        st.subheader(f"Distribución de: {var_seleccionada}")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Diferenciar entre gráficos para variables categóricas o numéricas
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
    st.markdown("En esta sección se entrena la malla de modelos mediante *GridSearchCV* utilizando una función de coste financiero para evaluar el rendimiento óptimo ante el riesgo de crédito.")
    
    if st.button("🚀 Iniciar Entrenamiento (GridSearch)"):
        with st.spinner("Entrenando macro-malla de algoritmos. Este proceso puede tardar unos minutos..."):
            
            # Preparación de variables categóricas para el preprocesador
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
            y = df_global['class'].astype(int).replace({1: 0, 2: 1}) # 0 = Bueno, 1 = Malo

            X_processed = preprocessor.fit_transform(X)
            cols_names = preprocessor.get_feature_names_out()
            df_final = pd.DataFrame(X_processed, columns=cols_names).apply(pd.to_numeric)

            # División de los datos
            X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42, stratify=y)
            cost_scorer = make_scorer(calcular_coste_financiero, greater_is_better=False)

            from sklearn.pipeline import Pipeline
            
            config_modelos = {
                "Logística": {
                    "model": LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42, max_iter=2000),
                    "params": {"clf__C": [0.01, 0.1, 1, 10]}
                },
                "SVM": {
                    "model": SVC(class_weight={0: 1, 1: 5}, random_state=42),
                    "params": {"clf__C": [0.1, 1, 10], "clf__kernel": ['linear', 'rbf']}
                },
                "Random Forest": {
                    "model": RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
                    "params": {
                        "clf__n_estimators": [100, 200],
                        "clf__max_depth": [4, 6, 8],
                        "clf__min_samples_leaf": [5, 10, 15]
                    }
                },
                "XGBoost": {
                    "model": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),
                    "params": {
                        "clf__learning_rate": [0.05, 0.1],
                        "clf__max_depth": [3, 4, 6],
                        "clf__n_estimators": [100, 200],
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
                
                # Predicciones y métricas
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
            
            # Guardar artefactos globales para la pestaña de predicción
            st.session_state['preprocessor'] = preprocessor
            st.session_state['cols_names'] = cols_names
            ganador = min(resultados_test, key=resultados_test.get)
            st.session_state['best_model'] = modelos_optimizados[ganador]
            st.session_state['best_model_name'] = ganador

            st.success(f"Entrenamiento finalizado. El mejor modelo recomendado es: **{ganador}** (Coste: {resultados_test[ganador]})")
            st.dataframe(df_resultados.style.highlight_min(subset=['Coste Financiero'], color='lightgreen'))

            # Gráfico multicriterio
            st.subheader("Comparativa de Rendimiento en Test")
            df_plot = df_resultados.melt(id_vars=["Algoritmo"], value_vars=["Accuracy", "Recall (Malos)", "F1-Score"], var_name="Métrica", value_name="Valor")
            fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
            sns.barplot(data=df_plot, x='Algoritmo', y='Valor', hue='Métrica', palette={"Accuracy": "#7FB3D5", "Recall (Malos)": "#E74C3C", "F1-Score": "#F39C12"}, ax=ax_perf)
            plt.ylim(0, 1.1)
            sns.despine()
            st.pyplot(fig_perf)

            # Importancia de variables del modelo ganador
            st.subheader(f"Importancia de Variables ({ganador})")
            final_clf = modelos_optimizados[ganador].named_steps['clf']
            
            if hasattr(final_clf, 'coef_'):
                importancia = final_clf.coef_.flatten()
            elif hasattr(final_clf, 'feature_importances_'):
                importancia = final_clf.feature_importances_
            else:
                importancia = None

            if importancia is not None:
                importance_df = pd.DataFrame({'Feature': df_final.columns, 'Importance': importancia})
                importance_df['Abs_Importance'] = importance_df['Importance'].abs()
                importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False).head(10)

                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='crest', legend=False, ax=ax_imp)
                plt.axvline(0, color='black', lw=1)
                sns.despine()
                st.pyplot(fig_imp)

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
        property_type = c5.selectbox("Propiedad", ["real estate", "life insurance", "car/other", "no property"])
        housing_type = c6.selectbox("Tipo de vivienda", ["rent", "own", "for free"])

        other_debtors = c4.selectbox("Otros deudores/Garante", ["none", "co-applicant", "guarantor"])
        residence_since = c5.slider("Tiempo en residencia actual (años)", min_value=1, max_value=4, value=2)
        dependents = c6.slider("Personas a cargo", min_value=1, max_value=2, value=1)
        
        telephone = c4.selectbox("Teléfono a su nombre", ["none", "yes"])
        foreign_worker = c5.selectbox("Trabajador extranjero", ["yes", "no"])

        submit_button = st.form_submit_button(label="🔍 Evaluar Riesgo")

    if submit_button:
        # Construcción del DataFrame con el cliente nuevo
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
        
        # Preprocesamiento idéntico al entrenamiento
        df_input = df_input.astype({col: 'category' for col in df_input.select_dtypes(['object', 'string']).columns})
        preprocessor = st.session_state['preprocessor']
        X_pred_proc = preprocessor.transform(df_input)
        df_pred_final = pd.DataFrame(X_pred_proc, columns=st.session_state['cols_names']).apply(pd.to_numeric)
        
        # Realización de la predicción
        modelo = st.session_state['best_model']
        prediccion = modelo.predict(df_pred_final)[0]
        
        st.markdown("---")
        if prediccion == 0:
            st.success("✅ **CRÉDITO APROBADO:** El modelo clasifica al cliente como de BAJO RIESGO (Cliente Bueno).")
        else:
            st.error("❌ **CRÉDITO DENEGADO:** El modelo clasifica al cliente como de ALTO RIESGO (Posible Morosidad).")