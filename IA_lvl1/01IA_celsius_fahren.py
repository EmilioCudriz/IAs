
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importar datos (Pasar de excel a PANDAS)

temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")
print(temperature_df)

#Visualizacion 

sns.scatterplot(x=temperature_df['Celsius'], y=temperature_df['Fahrenheit'])


#cargando SET de datos 
X_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']


#Crear el modelo de IA
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#Ver el modelo
model.summary()

#Creacion del compilado
model.compile(optimizer=tf.keras.optimizers.Adam(0.5), loss='mean_squared_error')


#Entrenar el modelo
epochs_hist = model.fit(X_train, y_train, epochs=145)

#Evaluando modelo
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss']) #grafica

#Mejorar grafica 
plt.title('Progrso de perdida de entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend('Training loss')

model.get_weights()

#Poner en practica el modelo 
Temp_c = int(input("Introduce los grados celsius a convertir: "))
Temp_F = model.predict([Temp_c])
Temp_F2 = float(9/5 * Temp_c + 32)
print("")

print("Por IA:")
print(Temp_F)
print("")
print("Por formula:")
print(Temp_F2)






