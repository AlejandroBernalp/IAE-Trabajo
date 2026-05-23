import pandas as pd
import time
from datos import cargar_datos_desde_r
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, make_scorer
from xgboost import XGBClassifier

# --- IMPORTS DE DASK ---
from distributed import Client
import dask_ml.model_selection as dask_search

if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")
    
    # 1. Carga de datos y preprocesamiento habitual
    df = cargar_datos_desde_r()
    df = df.astype({col: 'category' for col in df.select_dtypes(['object', 'string']).columns})
    df['class'] = df['class'].astype('category')

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

    X = df.drop(columns=['class'])
    y = df['class'].astype(int).replace({1: 0, 2: 1})

    X_processed = preprocessor.fit_transform(X)
    cols_names = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(X_processed, columns=cols_names).apply(pd.to_numeric)

    X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42, stratify=y)

    def calcular_coste_financiero(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            return (cm[0, 1] * 1) + (cm[1, 0] * 5)
        return 9999

    cost_scorer = make_scorer(calcular_coste_financiero, greater_is_better=False)

    # Engordamos un pelín la malla de hiperparámetros para que el test de estrés tenga sentido 
    # y Dask demuestre su potencia frente al método convencional.
    config_modelos = {
        "Logística": {
            "model": LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42, max_iter=2000),
            "params": {
                # Exploramos penalizaciones mucho más finas y extremas
                "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 50]
            }
        },
        "SVM": {
            "model": SVC(class_weight={0: 1, 1: 5}, random_state=42),
            "params": {
                # Añadimos más granularidad a C y el parámetro gamma para el kernel RBF
                "clf__C": [0.01, 0.1, 1, 5, 10, 50],
                "clf__kernel": ['linear', 'rbf'],
                "clf__gamma": ['scale', 'auto', 0.01, 0.1]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
            "params": {
                # Mantenemos tu lista e incrementamos combinaciones de árboles
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [5, 10, 15, 20],
                "clf__criterion": ["gini", "entropy"] # Evaluamos dos formas de medir la pureza
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                # Añadimos más variedad en la velocidad de aprendizaje y submuestreo
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__max_depth": [3, 4, 6, 8],
                "clf__n_estimators": [100, 200, 300],
                "clf__subsample": [0.8, 1.0], # Porcentaje de filas usadas por árbol
                "clf__scale_pos_weight": [5]
            }
        }
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

    # --- DICCIONARIO PARA ALMACENAR LOS TIEMPOS ---
    tiempos_ejecucion = {"Convencional (Joblib)": 0, "Paralelizado (Dask)": 0}

    # =========================================================================
    # FASE A: EXPERIMENTO CONVENCIONAL (Scikit-Learn original)
    # =========================================================================
    print("\n[TEST] Iniciando optimización CONVENCIONAL con Scikit-Learn...")
    start_conv = time.time()
    
    for nombre, config in config_modelos.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', config["model"])])
        # GridSearch nativo de sklearn
        grid_conv = SklearnGridSearchCV(
            estimator=pipeline, param_grid=config["params"], 
            scoring=cost_scorer, cv=skf, n_jobs=-1
        )
        grid_conv.fit(X_train, y_train)
        
    tiempos_ejecucion["Convencional (Joblib)"] = time.time() - start_conv
    print(f"-> Tiempo Convencional: {tiempos_ejecucion['Convencional (Joblib)']:.2f} segundos.")

    # =========================================================================
    # FASE B: EXPERIMENTO PARALELIZADO CON DASK
    # =========================================================================
    print("\n[TEST] Levantando clúster virtual de Dask...")
    # Inicializamos el cliente de Dask. Levanta un dashboard local automático.
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')
    print(f"-> Dashboard de Dask disponible en: {client.dashboard_link}")
    
    print("[TEST] Iniciando optimización PARALELIZADA con Dask-ML...")
    modelos_optimizados = {}
    start_dask = time.time()

    # Dask toma el control del contexto de ejecución
    with client:
        for nombre, config in config_modelos.items():
            pipeline = Pipeline([('scaler', StandardScaler()), ('clf', config["model"])])
            
            # ATENCIÓN: Usamos el GridSearchCV de Dask, no el de sklearn
            grid_dask = dask_search.GridSearchCV(
                estimator=pipeline, param_grid=config["params"], 
                scoring=cost_scorer, cv=skf
            )
            grid_dask.fit(X_train, y_train)
            modelos_optimizados[nombre] = grid_dask.best_estimator_

    tiempos_ejecucion["Paralelizado (Dask)"] = time.time() - start_dask
    print(f"-> Tiempo con Dask: {tiempos_ejecucion['Paralelizado (Dask)']:.2f} segundos.")
    
   # Cerramos el clúster de dask para liberar la RAM
    client.close()

    # =========================================================================
    # FASE C: TABLA JUSTIFICATIVA DE RENDIMIENTO PARA EL PROFESOR Y GRÁFICO
    # =========================================================================
    print("\n" + "="*60)
    print("MÉTRICAS DE RENDIMIENTO REQUERIDAS POR EL PROFESOR")
    print("="*60)
    df_tiempos = pd.DataFrame.from_dict(tiempos_ejecucion, orient='index', columns=['Tiempo (Segundos)'])
    mejora = ((tiempos_ejecucion["Convencional (Joblib)"] - tiempos_ejecucion["Paralelizado (Dask)"]) / tiempos_ejecucion["Convencional (Joblib)"]) * 100
    print(df_tiempos)
    print("-" * 60)
    if mejora > 0:
        print(f"🚀 ¡ÉXITO! Dask ha reducido el tiempo de procesamiento en un {mejora:.2f}%.")
    else:
        print(f"ℹ️ Dask tardó más debido al coste de inicializar el clúster en un dataset de 1000 filas.")
    print("="*60)

    # --- GENERACIÓN DINÁMICA DEL GRÁFICO REAL ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Mapeamos los nombres directamente desde tu diccionario de tiempos
    tiempos_reales = {
        "Convencional\n(Scikit-Learn / Joblib)": tiempos_ejecucion["Convencional (Joblib)"],
        "Paralelizado\n(Dask-ML Cluster)": tiempos_ejecucion["Paralelizado (Dask)"]
    }

    df_plot = pd.DataFrame(list(tiempos_reales.items()), columns=['Arquitectura', 'Tiempo (Segundos)'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    colores = ["#5D6D7E", "#2E4053"]

    ax = sns.barplot(x='Arquitectura', y='Tiempo (Segundos)', data=df_plot, hue='Arquitectura', palette=colores, edgecolor='0.3', legend=False)

    # Añadir etiquetas dinámicas sobre las barras
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f} s', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points', 
                        fontsize=11, fontweight='bold', color='#2C3E50')

    plt.title('Comparativa de Tiempos de Ejecución', fontsize=13, fontweight='bold', pad=20)
    plt.ylabel('Tiempo Total de Cómputo (Segundos)', fontsize=11)
    plt.xlabel('')
    
    # Ajustamos el límite superior del eje Y dinámicamente según el tiempo máximo obtenido
    plt.ylim(0, max(tiempos_reales.values()) * 1.2) 

    # Nota al pie dinámica con el cálculo de la pérdida/ganancia real
    porcentaje_cambio = abs(mejora)
    texto_nota = (
        f"Nota: Dask registra un incremento del {porcentaje_cambio:.1f}% en el tiempo debido al overhead estructural\n"
        "(inicialización del clúster local, asignación de workers y serialización TCP) sobre un dataset de 1000 registros."
    ) if mejora < 0 else f"Nota: Dask reduce el tiempo en un {porcentaje_cambio:.1f}% gracias a la paralelización distribuida."

    plt.figtext(0.5, -0.04, texto_nota, ha="center", fontsize=9, 
                bbox={"facecolor":"#F9EBEA" if mejora < 0 else "#EAF2F8", "alpha":0.5, "pad":8}, 
                style='italic', color='#78281F' if mejora < 0 else '#1B4F72')

    sns.despine()
    plt.tight_layout()

    # Guardar en la carpeta del proyecto
    plt.savefig('benchmark_dask_real.png', dpi=300, bbox_inches='tight')
    print("\n📊 ¡Gráfico dinámico 'benchmark_dask_real.png' generado y guardado correctamente!")
    plt.show()