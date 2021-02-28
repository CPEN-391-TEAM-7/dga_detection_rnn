# embedding = 1 * 68 * 8
# simple rnn = 8 * 8, 8 * 8, 8 * 1
# drop out: none
# dense: 8 * 1, 1 * 1
# activation: sigmoid
import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from math_help.fxp import Fxp

model_path = "model_export/model_bininary_0228.bin"
input_string = "google.com"

class RNN_Foward_Propagation(object):

    def __init__(self, model_path):
    
        self.raw_binary = None
        self.embedding_layer = None
        self.rnn_wh = None
        self.rnn_wx = None
        self.rnn_wy = None
        self.dense_1 = None
        self.dense_2 = None
        
        with open(model_path, mode='rb') as file:
            self.raw_binary = file.read()
            self.parse_weight()
            
    #get f4.12 byte object
    def get_byte_number(self, index):
        return Fxp(self.raw_binary[index*16:(index+1)*16])
        
    def forward_prop(self, input_string):
        pass
        
    def parse_weight(self):
        self.parse_emedding_layer()
        self.parse_rnn_layer()
        self.parse_dense_layer()
    
    def parse_emedding_layer(self):
        self.embedding_layer = [];
        for i in range(68):
            tmp = []
            for j in range(8):
                tmp.append(self.get_byte_number(i*8+j))
            self.embedding_layer.append(tmp)
        
    def parse_rnn_layer(self):
        self.rnn_wh = []
        self.rnn_wx = []
        self.rnn_wy = []
        for i in range(8):
            tmp_wh = []
            tmp_wx = []
            for j in range(8):
                tmp_wh.append(self.get_byte_number(544+i*8+j))
                tmp_wx.append(self.get_byte_number(608+i*8+j))
            self.rnn_wh.append(tmp_wh)
            self.rnn_wx.append(tmp_wx)
            self.rnn_wy.append(self.get_byte_number(672+i))
    
    def parse_dense_layer(self):
        self.dense_1 = []
        self.dense_2 = []
        for i in range(8):
            self.dense_1.append(self.get_byte_number(680+i))
        self.dense_2.append(self.get_byte_number(688))
        

    #compare fixed point and float point number
    def verify_fixed_point(self, fpn, ftn):
        print(fpn.value())
        print(ftn)
        print("inaccuracy = "+str((fpn.value()-ftn)/ftn*100)+"%")
    
    def tanh(self, number):
        pass
    
    def sigmoid(self, number):
        pass
        
    def matrix_mul(self, X, Y):
        result = [[ 0 for x in range(len(X))] for y in range(len(Y[0])) ]
        for i in range(len(X)):
            for j in range(len(Y[0])):
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]
        return result
                
rnn = RNN_Foward_Propagation(model_path)




