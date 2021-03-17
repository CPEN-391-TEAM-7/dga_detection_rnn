import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from model.rnn import RNN
import datetime
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

starttime = datetime.datetime.now()

trainDataPath = 'data/binary_training.txt'
modelPath = 'model_export/rnn_binary_model_0317_rnn32.h5'

charList = {}
confFilePath = './conf/charList.txt'

with open(confFilePath, 'r') as confFile:
    lines = confFile.read().split('\n')
ii = 0
for line in lines:
    temp = line.strip('\n').strip('\r').strip(' ')
    charList[temp] = ii
    ii += 1

print(len(charList))
max_features = ii
x_data_sum = []
y_data_sum = []
#
with open(trainDataPath, 'r') as trainFile:
    lines = trainFile.read().split('\n')

for line in lines:
    if line.strip('\n').strip('\r').strip(' ') == '':
        continue
    x_data = []
    x = line.strip('\n').strip('\r').strip(' ').split(',')[0]
    y = int(line.strip('\n').strip('\r').strip(' ').split(',')[1])
    for char in x:
        try:
            x_data.append(charList[char])
        except:
            print('unexpected char' + ' : ' + char)
            print(line)
            x_data.append(0)

    x_data_sum.append(x_data)
    y_data_sum.append(y)

x_data_sum = np.array(x_data_sum)
y_data_sum = np.array(y_data_sum)

rnn_binary  = RNN()
rnn_binary.fit(x_data_sum, y_data_sum, modelPath)
endtime = datetime.datetime.now()
print('=== starttime : ',starttime)
print('=== endtime   : ',endtime)

