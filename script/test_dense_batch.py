import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from model.rnn import RNN
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
trainDataPath = 'test_data/binary_test_1000.txt'
modelPath = 'model_export/rnn_binary_model_0317_rnn32.h5'
resultPath = 'test_result/denseout_test_1000.txt'

charList = {}
confFilePath = './conf/charList.txt'

with open(confFilePath, 'r') as confFile:
    lines = confFile.read().split('\n')
ii = 0
for line in lines:
    temp = line.strip('\n').strip('\r').strip(' ')
    charList[temp] = ii
    ii += 1

max_features = ii
x_data_sum = []
x_names = []
#
with open(trainDataPath, 'r') as trainFile:
    lines = trainFile.read().split('\n')

for line in lines:
    if line.strip('\n').strip('\r').strip(' ') == '':
        continue
    x_data = []
    x = line.strip('\n').strip('\r').strip(' ').split(',')[0]
    x_names.append(x)
    for char in x:
        try:
            x_data.append(charList[char])
        except:
            print('unexpected char' + ' : ' + char)
            print(line)
            x_data.append(0)

    x_data_sum.append(x_data)

x_data_sum = np.array(x_data_sum)

rnn_binary  = RNN()
rnn_binary.predict_dense_out(x_data_sum, x_names, resultPath, modelPath=modelPath)

