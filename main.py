import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datos import cargar_datos_desde_r
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from xgboost import XGBClassifier

if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE MACHINE LEARNING ---")
    
    # 1. Carga de datos
    df = cargar_datos_desde_r()

    # Convertir todas las columnas de tipo objeto/str a categóricas
    df = df.astype({col: 'category' for col in df.select_dtypes(['object', 'string']).columns})
    df['class'] = df['class'].astype('category')

    # 2. Configuración del Preprocesador
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
    y = df['class'].astype(int).replace({1: 0, 2: 1}) # 0 = Bueno, 1 = Malo (Clase objetivo)

    X_processed = preprocessor.fit_transform(X)
    cols_names = preprocessor.get_feature_names_out()
    df_final = pd.DataFrame(X_processed, columns=cols_names).apply(pd.to_numeric)

    # 3. División del dataset (80/20)
    X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42, stratify=y)

    # --- 4. MÉTRICA DE COSTE FINANCIERO PERSONALIZADA ---
    def calcular_coste_financiero(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            fp = cm[0, 1]  # Cliente bueno clasificado como malo (Coste 1)
            fn = cm[1, 0]  # Cliente malo clasificado como bueno (Coste 5)
            return (fp * 1) + (fn * 5)
        return 9999

    # Scorer optimizado para que GridSearchCV busque MINIMIZAR el coste
    cost_scorer = make_scorer(calcular_coste_financiero, greater_is_better=False)

    # --- 5. DEFINICIÓN DE CONFIGURACIONES PARA LA MALLA (GRIDSEARCH) ---
    # Nota: Escalamos dentro de cada validación mediante Pipelines independientes
    from sklearn.pipeline import Pipeline

    config_modelos = {
        "Logística": {
            "model": LogisticRegression(class_weight={0: 1, 1: 5}, random_state=42, max_iter=1000),
            "params": {
                "clf__C": [0.01, 0.1, 1, 10]
            }
        },
        "SVM": {
            "model": SVC(class_weight={0: 1, 1: 5}, random_state=42),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ['linear', 'rbf']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10, None],
                "clf__min_samples_leaf": [1, 2, 4]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5],
                "clf__n_estimators": [100, 200],
                "clf__scale_pos_weight": [5] # Penalización financiera nativa en XGBoost
            }
        }
    }

    # --- 6. OPTIMIZACIÓN MULTIMODELO CON VALIDACIÓN CRUZADA ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    modelos_optimizados = {}
    print(f"\n{'Algoritmo':<20} | {'Mejores Parámetros':<45} | {'Mejor Coste CV':<15}")
    print("-" * 88)

    for nombre, config in config_modelos.items():
        # Creamos un pipeline que primero escala y luego ejecuta el clasificador
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', config["model"])
        ])
        
        # GridSearch usando tu métrica de coste financiero y la validación cruzada estratificada
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            scoring=cost_scorer,
            cv=skf,
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        # Extraemos los mejores parámetros quitando el prefijo 'clf__' para legibilidad
        mejores_params_limpios = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}
        coste_medio = -grid.best_score_
        
        print(f"{nombre:<20} | {str(mejores_params_limpios):<45} | {coste_medio:<15.2f}")
        
        # Guardamos el mejor modelo entrenado de cada tipo
        modelos_optimizados[nombre] = grid.best_estimator_

    # --- 7. EVALUACIÓN DE LOS GANADORES EN EL DATASET DE TEST ---
    print("\n" + "="*50)
    print("EVALUACIÓN DEFINITIVA EN EL CONJUNTO DE TEST (80/20)")
    print("="*50)
    
    resultados_test = {}
    
    for nombre, modelo_final in modelos_optimizados.items():
        # Evaluación en datos de prueba que el modelo NUNCA ha visto
        y_pred = modelo_final.predict(X_test)
        coste_test = calcular_coste_financiero(y_test, y_pred)
        resultados_test[nombre] = coste_test
        
        print(f"\n--- {nombre.upper()} ---")
        print(f"Coste Financiero en Test: {coste_test}")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))

    # --- 8. SELECCIÓN DEL MODELO GANADOR ---
    ganador = min(resultados_test, key=resultados_test.get)
    print("\n" + "="*50)
    print(f"🏆 EL MODELO RECOMENDADO PARA EL BANCO ES: {ganador} (Coste en Test: {resultados_test[ganador]})")
    print("="*50)

    # --- 9. GRÁFICO MULTICRITERIO DE LOS MODELOS GANADORES ---
    results_viz = []
    for nombre, modelo_final in modelos_optimizados.items():
        cv_results = cross_validate(modelo_final, df_final, y, cv=skf, scoring=['accuracy', 'recall', 'f1'])
        results_viz.append({'Modelo': nombre, 'Métrica': 'Accuracy', 'Valor': cv_results['test_accuracy'].mean()})
        results_viz.append({'Modelo': nombre, 'Métrica': 'Recall', 'Valor': cv_results['test_recall'].mean()})
        results_viz.append({'Modelo': nombre, 'Métrica': 'F1-Score', 'Valor': cv_results['test_f1'].mean()})

    df_plot = pd.DataFrame(results_viz)

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    palette = {"Accuracy": "#7FB3D5", "Recall": "#E74C3C", "F1-Score": "#F39C12"}
    
    ax = sns.barplot(data=df_plot, x='Modelo', y='Valor', hue='Métrica', palette=palette, edgecolor='0.3')
    plt.title('Métricas de Rendimiento de Modelos Optimizados via GridSearchCV', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')
    
    sns.despine()
    plt.tight_layout()
    plt.show()

    # --- 10. IMPORTANCIA DE LAS VARIABLES (USANDO EL GANADOR) ---
    # Extraemos el clasificador final del pipeline ganador
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
        importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False).head(15)

        plt.figure(figsize=(12, 6))
        palette_dict = {row['Feature']: ('#E74C3C' if row['Importance'] > 0 else '#3498db') if hasattr(final_clf, 'coef_') else '#5D6D7E'
                        for _, row in importance_df.iterrows()}

        sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette=palette_dict, legend=False, edgecolor='0.3')
        plt.title(f'Top 15 Variables Determinantes ({ganador} Optimizado)', fontsize=16, fontweight='bold')
        plt.axvline(0, color='black', lw=1)
        sns.despine()
        plt.tight_layout()
        plt.show()