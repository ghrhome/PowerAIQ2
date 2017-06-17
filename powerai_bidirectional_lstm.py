'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import powerai_data
import sys


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

train_intention = True
if len(sys.argv) > 1:
    if sys.argv[1] == "a":
        train_intention = False


print('Loading data...')
(x_a_train, x_b_train, y_cate_train, y_inte_train), (
    x_a_test, x_b_test, y_cate_test, y_inte_test) = powerai_data.load_data()

if train_intention:
    print("Train Intention")
    (x_train, y_train), (x_test, y_test) = (x_b_train,
                                            y_inte_train), (x_b_test, y_inte_test)
else:
    print("Train Category")
    (x_train, y_train), (x_test, y_test) = (x_a_train,
                                            y_cate_train), (x_a_test, y_cate_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])
