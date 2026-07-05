import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist 
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=6)

model.save('Digit_Recognition.keras')

model = tf.keras.models.load_model('Digit_Recognition.keras')

loss, accuracy = model.evaluate(x_test,y_test)

print('Loss = ',loss)
print(f'Accuracy = {accuracy*100} %')

img_num=0
while os.path.isfile(f"digits/digit{img_num}.png"):
    try:
        img = cv.imread(f"digits/digit{img_num}.png", cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (28, 28))
        img = np.invert(img)
        
        img = img / 255.0
        img_input = np.expand_dims(img, axis=0)
        
        predict = model.predict(img_input)
        print(f"Digit {img_num} is predicted as: {np.argmax(predict)}")
        
        plt.imshow(img, cmap=plt.cm.binary)
        plt.title(f"Prediction: {np.argmax(predict)}")
        plt.show()
        
    except Exception as e:
        print(f"Error processing digit{img_num}.png: {e}")
        
    img_num += 1

