import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


url = 'https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
df = pd.read_csv(url, delimiter=';')

# Eliminar duplicados
df.drop_duplicates()

# Función para sustituir valores desconocidos en variables categóricas por la moda
def sust_unk_cat(columna):

    moda = df[columna].mode().tolist()[0]

    def unknown_categorico(valor):
        if valor == 'unknown':
            return moda
        else:
            return valor

    df[columna] = df[columna].apply(unknown_categorico)

variables_categoricas = df.columns[df.dtypes == 'object'].tolist()
del variables_categoricas[-1]

for variable in variables_categoricas:
    sust_unk_cat(variable)

# Función para obtener los límites para considerar outliers
def limites_outliers(columna):
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    IQR = q3 - q1
    min_so = q1 - 1.5*IQR
    max_so = q3 + 1.5*IQR
    return [min_so, max_so]

lim_age = limites_outliers('age')
lim_duration = limites_outliers('duration')
lim_campaign = limites_outliers('campaign')

# Eliminar outliers
df = df.drop(df[(df['age'] < lim_age[0]) | (df['age'] > lim_age[1])].index)
df = df.drop(df[(df['duration'] < lim_duration[0]) | (df['duration'] > lim_duration[1])].index)
df = df.drop(df[(df['campaign'] < lim_campaign[0]) | (df['campaign'] > lim_campaign[1])].index)

# Convertir edad a categórico
age_groups = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
df['age_groups'] = age_groups.astype(str).astype(int)

df.drop('age', axis=1, inplace=True)

# Convertir las categorías 'basic.9y','basic.6y','basic.4y' a 'middle.school'
def sustituir_categoria(valor):
    if valor in ['basic.9y','basic.6y','basic.4y']:
        return 'middle.school'
    else:
        return valor

df['education'] = df['education'].apply(sustituir_categoria)

# Convertir la variable target a binario
y_dict = {'yes':1, 'no':0}
df['y'] = df['y'].map(y_dict)

# Codificar variables categóricas
month_dict = {'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].map(month_dict)

day_dict = {'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6}
df['day_of_week'] = df['day_of_week'].map(day_dict)

object_columns_list = list(df.select_dtypes(include='object').columns)
df = pd.get_dummies(df, columns = object_columns_list)

# Construir modelo
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

modelo_lr = LogisticRegression()
modelo_lr.fit(X_train, y_train)


# Evaluar modelo
y_pred = modelo_lr.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Guardar modelo
filename = 'modelo_regresion_logistica.sav'
pickle.dump(modelo_lr, open(filename, 'wb'))

print(f'El valor de accuracy del modelo es {acc :.4f}.')
print(f'El valor de F1 del modelo es {f1 :.4f}.')
print(f'El valor de recall del modelo es {recall :.4f}.')
print()
print('Se guardó el modelo.')