from ucimlrepo import fetch_ucirepo 

def load_credit_data(): 
    # fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    return statlog_german_credit_data


statlog_german_credit_data = load_credit_data()


# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
    
# print(statlog_german_credit_data.variables.loc[:,['name','description']]) 
# Le cambiamos el nombre a las variables para que 
# tengan un nombre más fácilmente interpretable
print(X.head())
new_column_names = {
    'Attribute1': 'checking_status',
    'Attribute2': 'duration_months',
    'Attribute3': 'credit_history',
    'Attribute4': 'purpose',
    'Attribute5': 'credit_amount',
    'Attribute6': 'savings_status',
    'Attribute7': 'employment_since',
    'Attribute8': 'installment_rate',
    'Attribute9': 'personal_status',
    'Attribute10': 'other_debtors',
    'Attribute11': 'residence_since',
    'Attribute12': 'property_type',
    'Attribute13': 'age',
    'Attribute14': 'installment_plans',
    'Attribute15': 'housing_type',
    'Attribute16': 'existing_credits',
    'Attribute17': 'job_type',
    'Attribute18': 'dependents',
    'Attribute19': 'telephone',
    'Attribute20': 'foreign_worker'
}

X.rename(columns=new_column_names, inplace=True)

print(X.head())

#Ahora cambiamos los valores de las variables cualitativas 
#por nombres que tengan sentido, en vez de A11, A311, etc.
# haciendo uso de la guía en la página web.

category_mappings = {
    'checking_status': {
        'A11': '< 0 DM',
        'A12': '0-200 DM',
        'A13': '>= 200 DM',
        'A14': 'no checking'
    },
    'credit_history': {
        'A30': 'all paid duly',
        'A31': 'all paid duly (this bank)',
        'A32': 'existing paid duly',
        'A33': 'past delay',
        'A34': 'critical account'
    },
    'purpose': {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    },
    'savings_status': {
        'A61': '< 100 DM',
        'A62': '100-500 DM',
        'A63': '500-1000 DM',
        'A64': '>= 1000 DM',
        'A65': 'no savings'
    },
    'employment_since': {
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1-4 years',
        'A74': '4-7 years',
        'A75': '>= 7 years'
    },
    'personal_status': {
        'A91': 'male: divorced/separated',
        'A92': 'female: divorced/sep/married',
        'A93': 'male: single',
        'A94': 'male: married/widowed',
        'A95': 'female: single'
    },
    'other_debtors': {
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'
    },
    'property_type': {
        'A121': 'real estate',
        'A122': 'life insurance',
        'A123': 'car/other',
        'A124': 'no property'
    },
    'installment_plans': {
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none'
    },
    'housing_type': {
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free'
    },
    'job_type': {
        'A171': 'unskilled non-resident',
        'A172': 'unskilled resident',
        'A173': 'skilled official',
        'A174': 'management/highly qualified'
    },
    'telephone': {
        'A191': 'none',
        'A192': 'yes'
    },
    'foreign_worker': {
        'A201': 'yes',
        'A202': 'no'
    }
}

# Aplicar el cambio a todo el DataFrame de una vez
X.replace(category_mappings, inplace=True)

print(X.head())

