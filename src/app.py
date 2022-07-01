import pandas as pd

# Cargar csv
df = pd.read_csv('../data/raw/AB_NYC_2019.csv')

# Cambiar el tipo de las variables
df['last_review'] = df['last_review'].astype('datetime64[ns]')
df=df.astype({'name':'str','host_name':'str','neighbourhood_group':'category','neighbourhood':'category','room_type':'category'})

# Obtener el precio promedio (excluyendo los valores nulos) por zona y tipo de alojamiento
df_precio_no_nulo = df[df['price'] > 0]
df_precio_promedio = df_precio_no_nulo.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
pd.DataFrame(df_precio_promedio).sort_values(by='room_type')
dict_precios_promedio = df_precio_promedio.to_dict()
dict_precios_promedio

# Definir función para cambiar los precios nulos por el precio promedio por zona y tipo de alojamiento
def precio_nulo_a_promedio(fila):
	if fila['price'] > 0:
		return fila['price']
	else:
		return dict_precios_promedio[fila['room_type'], fila['neighbourhood_group']]

# Aplicar la función
df['price'] = df.apply(precio_nulo_a_promedio, axis=1)

# Cambiar los nan de reviews_per_month por 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Guardar csv
df.to_csv('../data/processed/AB_NYC_2019_processed.csv')