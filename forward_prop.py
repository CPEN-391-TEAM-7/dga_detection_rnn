# embedding = 1 * 68 * 8
# simple rnn = 8 * 8, 8 * 8, 8 * 1
# drop out: none
# dense: 8 * 1, 1 * 1
# activation: sigmoid
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

    #get f8.8 byte object
    def get_byte_number(self, index):
        return Fxp(self.raw_binary[index*16:(index+1)*16])
        
    def forward_prop(self):
        pass
        
    def parse_weight(self):
        pass
    
    def parse_emedding_layer(self):
        #
        pass
        
    #compare fixed point and float point number
    def verify_fixed_point(self, fpn, ftn):
        pass
    
    def tanh(self, number):
        pass
    
    def sigmoid(self, number):
        pass
            
rnn = RNN_Foward_Propagation(model_path)

#rnn.parse_weight()
a = rnn.get_byte_number(0)
a.value()


