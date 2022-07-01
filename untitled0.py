# Autocodificador para quitar ruido a una imagen

#Importamos Paquetes
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# CARGANDO SET DE DATOS
from tensorflow.keras.datasets import mnist

(X_Train,Y_Train),(X_Test,Y_Test)= mnist.load_data()

# VISUALIZACION DE DATOS
X_Train.shape
X_Test.shape

#Podemos ver que tenemos 60k de imagenes para entrenar y 10k para testear


# SELECCIONANDO IMAGEN ALEATORIA
i= random.randint(1,60000)

#va a elegir un numero entre 1 y 60.000

plt.imshow(X_Train[i], cmap= 'gray')

#ahora vamos a ver que siginfica la imagen
label= Y_Train[i]
print(label)


# AGREGANDO RUIDO A LAS IMAGENES 
# normalizamos las imagenes
X_Train = X_Train / 255
X_Test = X_Test / 255

#agregamos ruido
added_noise = np.random.randn(*(28,28))

#ponemos la cantidad de ruido en una variable
noise_factor= 0.3

#agregamos la cantidad de ruido 
added_noise = noise_factor * np.random.randn(*(28,28))

#mostramos la imagen con ruido
plt.imshow(added_noise, cmap='gray')

# LE AGREGAMOS RUIDO A UNA IMAGEN ALEATORIA
noise_factor= 0.2
sample_img = X_Train[321]
img_sample_noise= sample_img + noise_factor * np.random.randn(*(28,28))

#Mostramos la imagen con escalas de grises
plt.imshow(img_sample_noise,cmap= 'gray') 

#ahora veamos los valores maximos y minimos de la imagen
img_sample_noise.max()
img_sample_noise.min()
#podemos ver que los valores no van del 0 al 1
#asi que lo vamos a arreglar

img_sample_noise = np.clip(img_sample_noise, 0., 1.)

#ahora veamos como quedó
img_sample_noise.max()
img_sample_noise.min()

#veamos como se ve la imagen ahora con los rangos 0.0 y 1.0
plt.imshow(img_sample_noise,cmap= 'gray')


# HACEMOS EL MISMO PROCESO PARA TODAS LAS IMAGENES

X_Train_Noisy = []
noise_factor = 0.2

for img in X_Train:
    img_noise = img + noise_factor * np.random.randn(*(28,28))
    img_noise = np.clip(img_noise, 0., 1.)
    X_Train_Noisy.append(img_noise)



#COVERTIMOS NUESTRO SET DE DATOS EN UNA MATRIZ
X_Train_Noisy = np.array(X_Train_Noisy)

# Ahora vemos la forma que tiene 
X_Train_Noisy.shape

plt.imshow(X_Train_Noisy[22], cmap='gray')


#AHORA AGREGAMOS RUIDO A NUESTRO SET DE PRUEBA
X_Test_Noisy = []
noise_factor =0.4

for img in X_Test:
    img_noise = img + noise_factor * np.random.randn(*(28,28))
    img_noise = np.clip(img_noise, 0., 1.)
    X_Test_Noisy.append(img_noise)

#Ahora la covertimos en una matriz
X_Test_Noisy = np.array(X_Test_Noisy)

#Verificamos la forma de la matriz
X_Test.shape

#Veamos una imagen para verificar los cambios
plt.imshow(X_Test_Noisy[10], cmap='gray')


# Hasta ahora creamos un set de prueba y de entrenamiento con imagenes con ruido
# ahora vamos a crear nuestro autocodificador con deep learning


#Creacion del Modelo
autoencoder = tf.keras.models.Sequential()

#Armando capa convolucional
autoencoder.add(tf.keras.layers.Conv2D(16,(3,3), strides=1 , padding= "same", input_shape=(28,28,1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

autoencoder.add(tf.keras.layers.Conv2D(8,(3,3), strides=1 , padding= "same"))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

#imagen decodificada
autoencoder.add(tf.keras.layers.Conv2D(8,(3,3), strides=1 , padding= "same"))
                
#Armando decodificador                
autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(8, (3,3), strides=1,padding='same'))       

autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(1, (3,3), strides=1,activation='sigmoid',padding='same'))                


#Compilado
autoencoder.compile(loss='binary_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.001))

autoencoder.summary()

#1ra capa Conv2D : 
# Con 16 filtros, kernel de tamaño 3x3 que se mueven a 1 stride,
# Cuando padding="same"y strides=1, la salida tiene el mismo tamaño
# que la entrada.

# 2da capa MaxPooling2D:
# Esta capa aplica un filtro que reduce la dimensionalidad
# de la imagen, la hace más pequeña, en esta la convertimos
# en una de 2x2
# Van cogiendo grupos de 2x2 y nos quedamos con el máximo valor
# del grupo de cuatro píxeles.



#Entrenamiento
autoencoder.fit(X_Train_Noisy.reshape(-1,28,28,1),
                 X_Train.reshape(-1,28,28,1),
                 epochs=10,
                 batch_size=200)

#Evaluando el modelo
denoise_img= autoencoder.predict(X_Test_Noisy[:15].reshape(-1,28,28,1))
denoise_img.shape

#Visualizamos las imagenes decodificadas
fig, axes = plt.subplots(nrows=2, ncols=15, figsize=(30,6))

for img, row in zip([X_Test_Noisy[:15], denoise_img], axes):
    for imgs, ax in zip(img,row):
        ax.imshow(imgs.reshape((28,28)), cmap='gray')








































