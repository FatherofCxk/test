# mnist数据集 28×28矩阵输入，batch_size != 1
import numpy as np 
from keras.layers import LSTM,Dense
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential

# gurantee the same result
np.random.seed(521)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:5000]
y_test = y_test[:5000]

x_train = np.reshape(x_train, (-1,28,28))
y_train = np_utils.to_categorical(y_train, num_classes=10)
x_test = np.reshape(x_test, (-1,28,28))
y_test = np_utils.to_categorical(y_test, num_classes=10)

FEATURES = 28
TIME_STEPS = 28

# define the model 
model = Sequential()
model.add(LSTM(
    units=50,
    input_shape=(TIME_STEPS, FEATURES),

))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=30, verbose=2)

score = model.evaluate(x_test,y_test)

print(score)