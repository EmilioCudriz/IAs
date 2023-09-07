import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




##Importando Datos
house_df = pd.read_csv("Precios_hogares.csv")



#VISUALIZACION
sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)



#correlacion
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(house_df.corr(), annot = True)


#Limpieza de datos
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']


X = house_df[selected_features]
y = house_df['price']


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
  



#Normalizando output
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)


#Entrenamiento
from sklearn.model_selection import train_test_split

X_train, X_teste, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size= 0.25)

#Definir modelo
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

model.compile(optimizer= 'Adam', loss= 'mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)


#Evaluando Modelo
epochs_hist.history.keys()



#Grafico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])



#########PREDICTION!!!!!!!!!!!!
""" 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement' """

print("Prediccion del precio de inmuebles")
print("")

bedrooms = int(input("Numero de Cuartos: "))
bathrooms = int(input("Numero de baños: "))
floors = int(input("Numero de pisos: "))
sqft_living = int(input("Area del inmueble (m^2): ")) 
sqft_lot = int(input("Area del lote (m^2): "))
sqft_above = int(input("Area de la terraza (m^2): "))
sqft_basement = int(input("Area del subterraneo/sotano (m^2): "))

X_test_1 = np.array([[ bedrooms, bathrooms, sqft_living , sqft_lot, floors, sqft_above, sqft_basement]])


scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)

#Haciendo prediccion
y_predict_1 = model.predict(X_test_scaled_1)
y_predict_1 = scaler.inverse_transform(y_predict_1)


print("El precio en dolares del inmueble es de: ",  y_predict_1.max())













