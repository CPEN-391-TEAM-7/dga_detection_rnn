import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


class RNN(object):

    def __init__(self):
        pass

    def fit(self, x_train, y_train, modelPath, epochs = 20, batch_size=100, max_features=68, maxlen=64, embedding_value=8, rnn_value=8):
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(max_features, embedding_value, input_length=maxlen))#128*64
        self.model.add(layers.SimpleRNN(rnn_value))
        self.model.add(layers.Dropout(0.4))# change later
        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation('relu'))
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
        self.model.summary()
        keras.utils.plot_model(self.model, show_shapes=True)
        model2 = keras.Model(self.model.input, self.model.layers[-4].output)
        model2.summary()
        y2 = model2.predict(x_test, batch_size=1)
        print(y2)
        y_test = self.model.predict(x_test, batch_size=1).tolist()
        for index in y_test:
            y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            print(str(y) + '\n')
    
    def save_model_bin(self, binFilePath, modelPath=None):
        from fxpmath import Fxp
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        with open(binFilePath, 'wb') as binFile:
            for layer in self.model.layers:
                g=layer.get_config()
                h=layer.get_weights()
                print(g)
                #binFile.write(json.dumps(g).encode(encoding="ascii",errors="unknown char"))
                # embedding = 1 * 68 * 8
                # simple rnn = 8 * 8, 8 * 8, 8 * 1
                # drop out: none
                # dense: 8 * 1, 1 * 1
                # activation: sigmoid
                for i in h:
                    i = np.array(i)
                    for index, x in np.ndenumerate(i):
                        print(x)
                        h_fxp = Fxp(x, signed=True, n_word=16, n_frac=12)
                        print(h_fxp.bin())
                        binFile.write(h_fxp.bin().encode(encoding="ascii",errors="unknown char"))

            
    def save_model_txt(self, txtPath, modelPath=None):
        import json
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        with open(txtPath, 'w') as txtFile:
            for layer in self.model.layers:
                g=layer.get_config()
                h=layer.get_weights()
                print(type(g))
                print(type(h))
                txtFile.write(json.dumps(g))
                txtFile.write("\n")
                txtFile.write(str(h))
                txtFile.write("\n")
