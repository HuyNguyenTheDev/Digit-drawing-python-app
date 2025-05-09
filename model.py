#Tải các thư viện cần thiết và bộ dữ liệu chữ số mnist
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

#Chia tập train và test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Chia tập train và validation từ tập train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Reshape kích thước dữ liệu cho phù hợp với bài toán
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#Chuẩn hóa dữ liệu về [0, 1]
X_train = X_train /255.0
X_val = X_val /255.0
X_test = X_test /255.0

#Gán nhãn one-hot encoding cho từng giá trị nhãn y
Y_train = to_categorical(y_train, 10)
Y_val = to_categorical(y_val, 10)
Y_test = to_categorical(y_test, 10)

#Xây dựng model deep learning cơ bản và nhẹ dùng cho bài toán phân loại chữ số
model = Sequential()
model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#Train model và thực hiện trực quan hóa kết quả dự đoán
H = model.fit(X_train, Y_train, epochs = 10, batch_size = 128, validation_data = (X_val, Y_val), verbose=1)
fig = plt.figure()
numEpoch = 10

plt.plot(np.arange(0, numEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy per epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#Lưu file mô hình
model.save('model.h5')