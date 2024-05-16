import numpy as np 
from tensorflow import keras
from keras.datasets import mnist
import matplotlib as plt
from sklearn.metrics import classification_report

#Tải dữ liệu huấn luyện và dữ liệu đánh giá cho mô hình
(X_train, y_train), (X_test, y_test)= mnist.load_data()


#Xây dựng cấu trúc mô hình
def DCNN1_model():
    #CNN layer
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5,5), input_shape=(28,28,1), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0,25))
    model.add(keras.layers.Conv2D(32, (3, 3), activation= 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0,25))
    model.add(keras.layers.Flatten())
     
    #MLP with 3 hidden layer
    model.add(keras.layers.Dense(375, activation = 'relu'))
    model.add(keras.layers.Dropout(0,25))
    model.add(keras.layers.Dense(225, activation = 'relu'))
    model.add(keras.layers.Dropout(0,25))
    model.add(keras.layers.Dense(135, activation = 'relu'))
    model.add(keras.layers.Dropout(0,25))
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    return model


#Huấn luyện mô hình
model = DCNN1_model()
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size= 256, verbose= 2)


#Lưu lại mô hình
model.save("HWR model")


#Tải mô hình có sẵn 
model2 = keras.models.load_model("HWR model")

#Dánh giá mô hình
pred = np.argmax(model2.predict(X_test), axis = 1)
print(classification_report(pred, y_test))