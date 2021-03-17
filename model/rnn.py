import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


class RNN(object):

    def __init__(self):
        pass

    def fit(self, x_train, y_train, modelPath, epochs = 10, batch_size=100, max_features=39, maxlen=32, embedding_value=4, rnn_value=32):
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(max_features, embedding_value, input_length=maxlen))
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
        x_test = sequence.pad_sequences(x_test, maxlen=32)
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        y_test = self.model.predict(x_test, batch_size=batch_size).tolist()
        file = open(resultPath, 'w+')
        for index in y_test:
            y = round(float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']')))
            file.write(str(y) + '\n')

    def predict_dense_out(self,x_test, x_names, resultPath, batch_size=100, modelPath=None):
        x_test = sequence.pad_sequences(x_test, maxlen=32)
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        model2 = keras.Model(self.model.input, self.model.layers[-2].output)
        y2 = model2.predict(x_test, batch_size=batch_size).tolist()
        file = open(resultPath, 'w+')
        for index, string in enumerate(y2):
            y = float(str(string).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            file.write(x_names[index] + ',' + str(y) + '\n')

    def predict_p(self,x_test, modelPath=None):
        x_test = sequence.pad_sequences(x_test, maxlen=32)
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        self.model.summary()
        # dot_img_file = 'model_2.png'
        # keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
        model2 = keras.Model(self.model.input, self.model.layers[-2].output)
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
            largest_inaccuracy = 0
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
                        h_fxp = Fxp(x, signed=True, n_word=16, n_frac=8)
                        difference = abs(h_fxp.get_val()-x)
                        if difference>largest_inaccuracy:
                            largest_inaccuracy = difference
                        print(h_fxp.bin())
                        binFile.write(h_fxp.bin().encode(encoding="ascii",errors="unknown char"))
            print("largest difference")
            print(str(largest_inaccuracy))

            
    def save_model_txt(self, txtPath, modelPath=None):
        import json
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        with open(txtPath, 'w') as txtFile:
            for layer in self.model.layers:
                g=layer.get_config()
                h=layer.get_weights()
                # print(type(g))
                # print(type(h))
                txtFile.write(json.dumps(g))
                txtFile.write("\n")
                txtFile.write(str(h))
                txtFile.write("\n")

    def save_model_txt_binary(self, txtPath, modelPath = None):
        from fxpmath import Fxp
        import math
        import json
        if modelPath != None:
            self.model = load_model(modelPath, compile="False")
        with open(txtPath, 'w') as txtFile:
            for layer in self.model.layers:
                g=layer.get_config()
                h=layer.get_weights()
                txtFile.write(json.dumps(g))
                txtFile.write("\n")
                for i in h:
                    i = np.array(i)
                    for index, x in np.ndenumerate(i):
                        if g["name"] == "dropout":
                            continue
                        #if g["name"] == "embedding":
                        #     print(index)
                        #     row = math.floor(index / 4)
                        #     col = index % 4
                        #     txtFile.write("row:"+str(row)+"col:"+col)
                        # else:
                        #     row = math.floor(index / 32)
                        #     col = index % 32
                        #     txtFile.write("row:"+str(row)+"col:"+col)
                        if len(index) > 1:
                            row = index[0]
                            col = index[1]
                            txtFile.write("row:" + str(row) + "col:" + str(col))
                        else:
                            row = index[0]
                            txtFile.write("row:" + str(row))
                        h_fxp = Fxp(x, signed=True, n_word=16, n_frac=8)
                        txtFile.write("val:"+h_fxp.bin())
                        txtFile.write("\n")
