# embedding = 1 * 68 * 8
# simple rnn = 8 * 8, 8 * 8, 8 * 1
# drop out: none
# dense: 8 * 1, 1 * 1
# activation: reLu
import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from math_help.fxp import Fxp

model_path = "model_export/model_bininary_0301.bin"
tanh_table_path = "conf/tanh_table.bin"
charListPath = 'conf/charList.txt'
input_string = "qwdsdivwefqwdoamsxc.com"

class RNN_Foward_Propagation(object):

    def __init__(self, model_path, tanh_table_path, charListPath):

        self.raw_binary = None
        self.embedding_layer = None
        self.rnn_w = None
        self.rnn_u = None
        self.rnn_b = None
        self.dense_w = None
        self.dense_b = None
        self.tanh_table = None
        self.charList = None

        with open(model_path, mode='rb') as file:
            self.raw_binary = file.read()
            self.parse_weight()

        self.parse_tanh_table(tanh_table_path)
        self.parse_char_list(charListPath)

    def forward_prop(self, input_string):
        x_data = []
        for char in input_string:
            try:
                x_data.append(self.charList[char])
            except:
                print('unexpected char' + ' : ' + char)
                x_data.append(0)
        x2 = self.process_embedding(x_data)
        # self.print_layer(x2)
        prev_out = [[Fxp([0]*16)]*8] #initialize hidden state
        for input in x2:
            self.expand(input)
            hidden = self.matrix_add(self.matrix_mul(self.transpose(input), self.rnn_w), self.rnn_b)
            # hidden = self.tanh_matrix(self.matrix_add(self.matrix_add(self.matrix_mul(self.rnn_w, input),
            #                                                           self.matrix_mul(self.rnn_u, hidden)),
            #                                           self.rnn_b))
            prev_out = self.tanh_matrix(self.matrix_add(hidden, self.matrix_mul(prev_out, self.rnn_u)))
        x3 = self.matrix_add(self.matrix_mul(prev_out, self.dense_w), self.dense_b)
        # for x in x3:
        #     self.print_layer(x)
        # x4 = []
        # for input in x3:
        #     nxt = self.matrix_add(self.matrix_mul(self.transpose(input), self.dense_w), self.dense_b)
        #     x4.append(nxt)
        x4 = self.reLu_matrix(x3)
        print(x4[0][0].value())

    def process_embedding(self, x_data):
        result = []
        for i in range(64-len(x_data)):
            result.append(self.embedding_layer[0].copy())
        for i in x_data:
            result.append(self.embedding_layer[i+1].copy())
        return result

    def print_layer(self, matrix):
        result = []
        for i in matrix:
            tmp = []
            for j in i:
                tmp.append(j.value())
            result.append(tmp)
        print(result)

    def transpose(self, matrix):
        result = []
        for i in range(len(matrix)):
            result.append(matrix[i][0])
        return [result]

    def expand(self, matrix):
        for i in range(len(matrix)):
            matrix[i] = [matrix[i]]

    def parse_char_list(self, charListPath):
        self.charList = {}
        with open(charListPath, 'r') as confFile:
            lines = confFile.read().split('\n')
        ii = 0
        for line in lines:
            temp = line.strip('\n').strip('\r').strip(' ')
            if temp != '':
                self.charList[temp] = ii
                ii += 1

    def get_char_vector(self, index):
        result = [Fxp([0]*16)]*68
        result[index] = Fxp([0,0,0,1]+[0]*12)
        return result

    #get f4.12 byte object
    def get_byte_number(self, index):
        return Fxp(self.raw_binary[index*16:(index+1)*16])

    def parse_tanh_table(self, tanh_table_path):
        with open(tanh_table_path, mode="rb") as file:
            raw_tanh = file.read()
            self.tanh_table = []
            for i in range(10):
                entry = []
                entry.append(Fxp(raw_tanh[2*i*16:(2*i+1)*16]))
                entry.append(Fxp(raw_tanh[(2*i+1)*16:(2*i+2)*16]))
                self.tanh_table.append(entry)

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
        self.rnn_w = []
        self.rnn_u = []
        self.rnn_b = []
        tmp = []
        for i in range(8):
            tmp_wh = []
            tmp_wx = []
            for j in range(8):
                tmp_wh.append(self.get_byte_number(544+i*8+j))
                tmp_wx.append(self.get_byte_number(608+i*8+j))
            self.rnn_w.append(tmp_wh)
            self.rnn_u.append(tmp_wx)
            tmp.append(self.get_byte_number(672 + i))
        self.rnn_b.append(tmp)

    def parse_dense_layer(self):
        self.dense_w = []
        self.dense_b = []
        for i in range(8):
            self.dense_w.append([self.get_byte_number(680 + i)])
        self.dense_b.append([self.get_byte_number(688)])

    #compare fixed point and float point number
    def verify_fixed_point(self, fpn, ftn):
        print(fpn.value())
        print(ftn)
        print("inaccuracy = "+str((fpn.value()-ftn)/ftn*100)+"%")

    def tanh_matrix(self, matrix):
        result = []
        for i in matrix:
            if not type(i) == list:
                result.append(self.tanh(i))
            else:
                result.append(self.tanh_matrix(i))
        return result

    def reLu_matrix(self, matrix):
        result = []
        for i in matrix:
            if not type(i) == list:
                result.append(self.reLu(i))
            else:
                result.append(self.reLu_matrix(i))
        return result

    # def tanh(self, fxp_num):
    #     diff = abs(self.tanh_table[0][0].value() - fxp_num.value()) + 1
    #     result = None
    #     for entry in self.tanh_table:
    #         newdiff = abs(entry[0].value() - fxp_num.value())
    #         if newdiff > diff:
    #             return result
    #         else:
    #             result = entry[1]
    #             diff = newdiff
    #     return result

    def tanh(self, fxp_num):
        from fxpmath import Fxp as Fxp2
        import numpy as np
        xTanh = np.tanh(fxp_num.value())
        xTanh_fxp = Fxp2(xTanh, signed=True, n_word=16, n_frac=12)
        return Fxp(xTanh_fxp.bin())

    def reLu(self, fxp_num):
        if fxp_num.value() < 0:
            return Fxp([0]*16)
        else:
            return fxp_num

    def matrix_mul(self, X, Y):
        result = [[Fxp([0]*16) for y in range(len(Y[0]))] for x in range(len(X))]
        for i in range(len(X)):
            for j in range(len(Y[0])):
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]
        return result

    def matrix_add(self, X, Y):
        result = [[Fxp([0]*16) for y in range(len(X[0]))] for x in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X[0])):
                result[i][j] += X[i][j] + Y[i][j]
        return result


def test_tanh():
    rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)
    for i in rnn.dense_w:
        print(i.value())
        print(rnn.tanh(i).value())

def test_matrix():
    rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)
    a = [[1,2,3,4,5],[6,7,8,9,10]]
    b = [[1,2,3], [4,5,6], [7,8,9] ,[10,11,12], [13,14,15]]
    c = [[1,1,1,1,1],[1,1,1,1,1]]
    print(rnn.matrix_mul(a,b))
    print(rnn.matrix_add(a,c))

rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)
rnn.forward_prop(input_string)

# test_matrix()
