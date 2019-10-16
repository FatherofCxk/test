import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils

np.random.seed(521)

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

char_to_int = dict((lt,i) for i,lt in enumerate(alphabet))
int_to_char = dict((i,lt) for i,lt in enumerate(alphabet))

# print(letter_to_int,'\n\n',int_to_letter)

seq_len = 5
data_X = []
data_Y = []

for i in range(0,len(char_to_int)-seq_len,1):
    # char data
    seq_in = alphabet[i:i+seq_len]
    # seq_out = alphabet[i+seq_len]
    # convert to num sequence
    data_X.append([char_to_int[char] for char in seq_in])
    data_Y.append([char_to_int[char]+1 for char in seq_in])
# print(data_X,data_Y)

# input_shape : (samples, time_steps, features)
x = np.reshape(data_X,(len(data_X), seq_len, 1))
x = x/float(len(alphabet))
y = np_utils.to_categorical(data_Y,num_classes=26)
print(x.shape,y.shape)

# define LSTM model 
model = Sequential()
model.add(LSTM(
    units=32,
    input_shape=(x.shape[1],x.shape[2]),  # INPUT_SHAPE (None, time_steps, features) PS:None is the arbiterary batch size
    return_sequences=True,
    stateful=False,
))
model.add(TimeDistributed(Dense(26,activation='softmax')))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=1, epochs=1000, verbose=2)

# score = model.evaluate(x, y, verbose=1)

def predict_sequence(sequence):
    global char_to_int
    input_data = []
    input_data.append([[char_to_int[letter] for letter in sequence]])
    input_data = np.reshape(input_data,(1, 5, 1))
    return input_data

count = 10
out_letter = 'ABCDE'
while(count):
    print(out_letter,'\n-->')

    input_data = predict_sequence(out_letter)/float(len(alphabet))

    pred_result = model.predict(input_data)

    # print(pred_result.shape)  # (1, 5, 26)

    out_letter=''

    for i in range(0,seq_len):
        index = pred_result[0,i,:].argmax()
        out_letter += int_to_char[index]
    # print(out_letter)

    count -= 1
# print(score)

# for data in data_X:
#     a = data
#     data = np.reshape(data,(1,seq_len,1))   
#     data = data/float(len(alphabet))
#     pred = model.predict(data)
#     index = np.argmax(pred)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in a]

#     print(seq_in,'-->',result)