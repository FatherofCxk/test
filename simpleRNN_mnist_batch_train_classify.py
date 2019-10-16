# mnist数据集，使用train_on_batch方法分批训练

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001


# Data Preparation
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:1000].reshape(-1,28,28) / 255.0
y_train = y_train[:1000]
y_train = np_utils.to_categorical(y_train,num_classes=10)

x_test = x_test[:200].reshape(-1,28,28) / 255.0
y_test = y_test[:200]
y_test = np_utils.to_categorical(y_test,num_classes=10)


# Build Sequential RNN Model
model = Sequential()
model.add(SimpleRNN(
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
    units=CELL_SIZE,
    unroll=True,
))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy']
)
model.summary()

# Training
for step in range(20001):
    x_batch = x_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :, :]
    y_batch = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(x_batch, y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX>=x_train.shape[0] else BATCH_INDEX

    if step%500 == 0:
        # batch size of Evaluate must be the same as the model input size # 
        cost,accuracy = model.evaluate(x_test, y_test,batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test acc :', accuracy)
