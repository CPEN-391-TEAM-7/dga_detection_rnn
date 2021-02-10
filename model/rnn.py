import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


class RNN(object):

    def __init__(self):
        pass

    def fit(self, x_train, y_train, modelPath, epochs = 10, batch_size=100, max_features=68, maxlen=64, embedding_value=128, rnn_value=128):
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(max_features, embedding_value, input_length=maxlen))#128*64
        self.model.add(layers.SimpleRNN(rnn_value))
        self.model.add(layers.Dropout(0.4))# change later
        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.summary()
        x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        self.model.save(modelPath)
        
    def predict(self,x_test, resultPath, batch_size=100, modelPath=None):
        x_test = sequence.pad_sequences(x_test, maxlen=64)
        if modelPath != None:
            self.model = load_model(modelPath, complile="False")
        y_test = self.model.predict(x_test, batch_size=batch_size).tolist()
        file = open(resultPath, 'w+')
        for index in y_test:
            y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            file.write(str(y) + '\n')
            
    def predict_p(self,x_test, modelPath=None):
        x_test = sequence.pad_sequences(x_test, maxlen=64)
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        y_test = self.model.predict(x_test, batch_size=1).tolist()
        for index in y_test:
            y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            print(str(y) + '\n')

