
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sales_df = pd.read_csv("datos_de_ventas.csv")

sns.scatterplot(x=sales_df['Temperature'], y=sales_df['Revenue'])


X_train = sales_df['Temperature']

y_train = sales_df['Revenue']

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))


model.summary()

#Creacion del compilado
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')


#Entrenamiento 
epochs_hist = model.fit(X_train, y_train, epochs=1000)

keys = epochs_hist.history.keys()
#print(keys)

plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de perdida')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend('Training loss')

weights = model.get_weights()
#print(weights)


Temp = float(input("Introduce la temperatura del dia: "))
ganancias = model.predict([Temp])
print("")

print("Las ventas del dia segun la IA seran de: ")
print(ganancias)



