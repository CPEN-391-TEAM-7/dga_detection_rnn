import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from script.forward_prop import RNN_Foward_Propagation
from math_help.fxp import Fxp

model_path = "model_export/rnn_binary_model_0307_rnn32.bin"
tanh_table_path = "conf/tanh_table.bin"
charListPath = 'conf/charList.txt'
input_string = "google.com"

def test_val():
    rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)

    for i in range(50):
        a = rnn.get_byte_number(i)
        print(a.value())
        
def test_mul():
    rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)

    for i in range(50):
        a = rnn.get_byte_number(i)
        b = rnn.get_byte_number(i+1)
        print("test "+ str(i))
        print(a.value())
        print(b.value())
        print("Real Value "+str(a.value()*b.value()))
        print("Test Value "+str((a*b).value()))
        print("\n")

def test_add():
    rnn = RNN_Foward_Propagation(model_path, tanh_table_path, charListPath)

    for i in range(50):
        a = rnn.get_byte_number(i)
        b = rnn.get_byte_number(i+1)
        print("test "+ str(i))
        print(a.value())
        print(b.value())
        print("Real Value "+str(a.value()+b.value()))
        print("Test Value "+str((a+b).value()))
        print("\n")

test_mul()




