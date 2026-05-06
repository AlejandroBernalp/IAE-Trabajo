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

