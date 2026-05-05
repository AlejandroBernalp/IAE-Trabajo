# 1. Instalar y cargar la librería
if (!require("ucimlrepo")) install.packages("ucimlrepo")
library(ucimlrepo)

if (!require("dplyr")) install.packages("dplyr")
library(dplyr)
# 2. Traer el dataset usando su ID
german_credit <- fetch_ucirepo(id = 144)

# 3. Extraer los datos en un solo DataFrame

X <- german_credit$data$original

#Compruebo si los tipos son los deseados

str(X)

#Las variables categóricas son de tipo character, y la variable class como entera

#Cambio nombres

nuevos_nombres <- c(
  "checking_status", "duration_months", "credit_history", "purpose", 
  "credit_amount", "savings_status", "employment_since", "installment_rate", 
  "personal_status", "other_debtors", "residence_since", "property_type", 
  "age", "installment_plans", "housing_type", "existing_credits", 
  "job_type", "dependents", "telephone", "foreign_worker", "class"
)

names(X) <- nuevos_nombres

X <- X %>%
  mutate(
    checking_status = recode(checking_status, 
                             "A11" = "< 0 DM", "A12" = "0-200 DM", 
                             "A13" = ">= 200 DM", "A14" = "no checking"),
    
    credit_history = recode(credit_history,
                            "A30" = "all paid duly", "A31" = "all paid bank",
                            "A32" = "existing paid", "A33" = "past delay", 
                            "A34" = "critical account"),
    
    purpose = recode(purpose,
                     "A40" = "car (new)", "A41" = "car (used)", 
                     "A42" = "furniture/equipment", "A43" = "radio/television",
                     "A44" = "domestic appliances", "A45" = "repairs", 
                     "A46" = "education", "A47" = "vacation", 
                     "A48" = "retraining", "A49" = "business", "A410" = "others"),
    
    savings_status = recode(savings_status,
                            "A61" = "< 100 DM", "A62" = "100-500 DM",
                            "A63" = "500-1000 DM", "A64" = ">= 1000 DM", 
                            "A65" = "no savings"),
    
    employment_since = recode(employment_since,
                              "A71" = "unemployed", "A72" = "< 1 year",
                              "A73" = "1-4 years", "A74" = "4-7 years", 
                              "A75" = ">= 7 years"),
    
    personal_status = recode(personal_status,
                             "A91" = "male: divorced/sep", "A92" = "female: div/sep/mar",
                             "A93" = "male: single", "A94" = "male: mar/wid", 
                             "A95" = "female: single"),
    
    other_debtors = recode(other_debtors,
                           "A101" = "none", "A102" = "co-applicant", 
                           "A103" = "guarantor"),
    
    property_type = recode(property_type,
                           "A121" = "real estate", "A122" = "life insurance",
                           "A123" = "car/other", "A124" = "no property"),
    
    installment_plans = recode(installment_plans,
                               "A141" = "bank", "A142" = "stores", 
                               "A143" = "none"),
    
    housing_type = recode(housing_type,
                          "A151" = "rent", "A152" = "own", 
                          "A153" = "for free"),
    
    job_type = recode(job_type,
                      "A171" = "unskilled non-res", "A172" = "unskilled res",
                      "A173" = "skilled official", "A174" = "mgmt/highly qualif"),
    
    telephone = recode(telephone,
                       "A191" = "none", "A192" = "yes"),
    
    foreign_worker = recode(foreign_worker,
                            "A201" = "yes", "A202" = "no")
  )

# 4. Ver resultado
head(X)

# Convertir todas las columnas de texto a factor
X <- X %>%
  mutate(across(where(is.character), as.factor))

#Convertir la clase a factor
X$class <- as.factor(X$class)

# Opcional: Verificar la estructura para confirmar los cambios
str(X)

# Exportar el dataframe a un archivo CSV para leerlo en Python
write.csv(X, "german_credit_clean.csv", row.names = FALSE)
