import pandas as pd

# Importar el archivo generado en R
df = pd.read_csv("german_credit_clean.csv")

# Verificar que los nombres y valores están correctos
print(df.head())
print(df.info())

# Convertir todas las columnas de tipo objeto/str a categóricas
df = df.astype({col: 'category' for col in df.select_dtypes(['object', 'string']).columns})

# La columna 'class' suele ser el objetivo (1=bueno, 2=malo), 
# así que también conviene que sea categórica
df['class'] = df['class'].astype('category')

# Verificar el resultado final
print(df.info())

# Hay que cambiar el encoding de las variables categóricas para modelos que no entienden bien 
# Vamos a aplicar One Hot Encoding y Ordinal Encoding a las variables que tenga sentido. 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

# 1. Listas de columnas según su tratamiento
# He movido credit_history a nominal_cols porque el error venía de ahí
nominal_cols = [
    'credit_history', 'purpose', 'personal_status', 
    'other_debtors', 'property_type', 'installment_plans', 'housing_type'
]

ordinal_cols = ['checking_status', 'savings_status', 'employment_since', 'job_type']

binary_cols = ['telephone', 'foreign_worker']

# 2. Definir los órdenes para las ordinales (basado exactamente en la descripción UCI)
# IMPORTANTE: Los nombres deben coincidir EXACTAMENTE con lo que hay en el DF
ordinal_order = [
    ['no checking', '< 0 DM', '0-200 DM', '>= 200 DM'], # checking_status
    ['no savings', '< 100 DM', '100-500 DM', '500-1000 DM', '>= 1000 DM'], # savings_status
    ['unemployed', '< 1 year', '1-4 years', '4-7 years', '>= 7 years'], # employment_since
    ['unskilled non-res', 'unskilled res', 'skilled official', 'mgmt/highly qualif'] # job_type CORREGIDO
]

# 3. Construir el transformador
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_order), ordinal_cols),
        ('nom', OneHotEncoder(drop='first', sparse_output=False), nominal_cols),
        ('bin', OrdinalEncoder(), binary_cols)
    ],
    remainder='passthrough'
)

# 4. Transformar y reconstruir el DataFrame
X = df.drop(columns=['class']) # Suponiendo que 'class' es tu target
X_processed = preprocessor.fit_transform(X)

# Recuperar nombres de columnas
cols_names = preprocessor.get_feature_names_out()

# Crear el DF final y forzar tipos numéricos
df_final = pd.DataFrame(X_processed, columns=cols_names)
df_final = df_final.apply(pd.to_numeric)

print("¡Éxito! Todas las variables son numéricas ahora.")
print(df_final.info())

# MODELOS: 

# REGRESIÓN LOGÍSTICA

# Convertimos: Good (1) -> 0 , Bad (2) -> 1
# Así, el modelo predice la probabilidad de ser "Malo" (la clase de interés)
# Convertimos primero a entero para poder cambiar los valores libremente
y = df['class'].astype(int).replace({1: 0, 2: 1})

print("Distribución de la variable objetivo:")
print(y.value_counts())
# Deberías ver: 
# 0 (Good): 700
# 1 (Bad): 300
X = df_final # Tu DataFrame de 38 columnas

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. División en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Escalado (Fundamental para Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento del modelo
# Usamos class_weight='balanced' para que el modelo sepa que la clase 1 es más difícil/importante
model_log = LogisticRegression(class_weight={0: 1, 1: 5}) # Aplicamos la matriz de costes aquí
model_log.fit(X_train_scaled, y_train)

# 4. Predicciones
y_pred = model_log.predict(X_test_scaled)

# 5. Evaluación inicial
print(classification_report(y_test, y_pred))

#Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión Regresión logística:")
print(cm)

#Aunque el modelo tiene una precisión baja, solo hemos tenido 9 errores graves de tipo financiero (FN), logrando un coste total de 121. Sin la matriz de pesos, este número de errores graves sería mucho mayor

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# 1. Instanciar el modelo con el peso de la matriz de costes
# RandomForest usa 'class_weight' igual que la logística
model_rf = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 5}, random_state=42)

# 2. Entrenar (puedes usar X_train sin escalar si quieres, pero por orden seguiremos con el scaled)
model_rf.fit(X_train_scaled, y_train)

# 3. Predecir y evaluar
y_pred_rf = model_rf.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_rf)
print("Matriz de Confusión RF:")
print(cm)
print("--- REPORTE RANDOM FOREST ---")
print(classification_report(y_test, y_pred_rf))

# SVM
from sklearn.svm import SVC

# 1. Instanciar SVM con la matriz de costes
# En SVM el parámetro también se llama class_weight
model_svm = SVC(kernel='linear', class_weight={0: 1, 1: 5}, random_state=42)

# 2. Entrenar
model_svm.fit(X_train_scaled, y_train)

# 3. Predecir
y_pred_svm = model_svm.predict(X_test_scaled)

print("--- REPORTE SVM ---")
print(classification_report(y_test, y_pred_svm))
print("Matriz de Confusión SVM:")
print(confusion_matrix(y_test, y_pred_svm))

# PERCEPTRÓN MULTICAPA
from sklearn.neural_network import MLPClassifier

# 1. Instanciar la Red Neuronal
# Nota: MLPClassifier no tiene el parámetro 'class_weight' directamente como los otros.
# Para compensar el coste, una técnica común es el sobremuestreo o ajustar el dataset, 
# pero para no complicar el código ahora, vamos a ver cómo se porta "a pelo" 
# o usando un truco de arquitectura.
model_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# 2. Entrenar
model_nn.fit(X_train_scaled, y_train)

# 3. Predecir
y_pred_nn = model_nn.predict(X_test_scaled)

print("--- REPORTE RED NEURONAL ---")
print(classification_report(y_test, y_pred_nn))
print("Matriz de Confusión Red Neuronal:")
print(confusion_matrix(y_test, y_pred_nn))

# XGBOOST
from xgboost import XGBClassifier

# 1. Definir el peso basado en tu matriz de costes (5 a 1)
# Creamos un array de pesos para el entrenamiento
sample_weights = [5 if y == 1 else 1 for y in y_train]

# 2. Instanciar el modelo
model_xgb = XGBClassifier(random_state=42)

# 3. Entrenar usando los pesos de la matriz de costes
model_xgb.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# 4. Predicciones
y_pred_xgb = model_xgb.predict(X_test_scaled)

print("--- REPORTE XGBOOST ---")
print(classification_report(y_test, y_pred_xgb))
print("Matriz de Confusión XGBoost:")
print(confusion_matrix(y_test, y_pred_xgb))