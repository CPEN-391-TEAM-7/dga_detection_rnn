from model.rnn import RNN
import numpy as np

import os
import argparse

modelPath = 'model_export/rnn_binary_model_0209.h5'
charListPath = 'conf/charList.txt'

def getCharList(charListPath):
    charList = {}

    with open(charListPath, 'r') as confFile:
        lines = confFile.read().split('\n')
    ii = 1
    for line in lines:
        temp = line.strip('\n').strip('\r').strip(' ')
        if temp != '':
            charList[temp] = ii
            ii += 1
    return charList
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--domain', required=True)
    
    io_args = parser.parse_args()
    domain = io_args.domain
    
    charList = getCharList(charListPath)
    
    x_data = []
    for char in domain:
        try:
            x_data.append(charList[char])
        except:
            print('unexpected char' + ' : ' + char)
            x_data.append(0)
    
    rnn_binary_model = RNN()
    rnn_binary_model.predict_p([x_data], modelPath=modelPath)
    
    
