import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,TimeDistributed
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import Adam

np.random.seed(521)

# define the sequential data
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_int = dict((c,i) for i,c in enumerate(alphabet))
int_to_char = dict((i,c) for i,c in enumerate(alphabet))
# print(char_to_int,'\n\n',int_to_char)

seq_len = 5
data_X = []
data_Y = []

for i in range(0,len(char_to_int)-seq_len,1):
    # char data
    seq_in = alphabet[i:i+seq_len]
    seq_out = alphabet[i+seq_len]
    # convert to num sequence
    data_X.append([char_to_int[char] for char in seq_in])
    data_Y.append(char_to_int[seq_out])
print(data_X,data_Y)

# input_shape : (samples, time_steps, features)
x = np.reshape(data_X,(len(data_X), seq_len, 1))
x = x/float(len(alphabet)) 

y = []
for data in data_X:
    for a in data:
        label_a = a+1
        y.append([[label_a]])
# y = np_utils.to_categorical(data_Y,num_classes=26)
y = np.array(y)
y = np.reshape(y,(len(data_X),seq_len,-1))

y = np_utils.to_categorical(y,num_classes=26)
# print(x,y)
# exit()
# define LSTM model 
model = Sequential()
model.add(LSTM(
    units=32,
    # INPUT_SHAPE (None, time_steps, features) PS:None is the arbiterary batch size
    input_shape=(x.shape[1],x.shape[2]),
    return_sequences=True,
    stateful=False,
))
model.add(TimeDistributed(Dense(y.shape[2],activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x, y, batch_size=1, epochs=500, verbose=2)

score = model.evaluate(x, y, verbose=1)

print(score)

# for data in data_X:
#     a = data
#     data = np.reshape(data,(1,seq_len,1))
#     data = data/float(len(alphabet))
#     pred = model.predict(data)
#     index = np.argmax(pred)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in a]

#     print(seq_in,'-->',result)