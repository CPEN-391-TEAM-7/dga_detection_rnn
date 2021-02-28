import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from forward_prop import RNN_Foward_Propagation
from math_help.fxp import Fxp

def test_val():
    model_path = "model_export/model_bininary_0228.bin"
    rnn = RNN_Foward_Propagation(model_path)

    for i in range(50):
        a = rnn.get_byte_number(i)
        print(a.value())
        
def test_mul():
    model_path = "model_export/model_bininary_0228.bin"
    rnn = RNN_Foward_Propagation(model_path)

    for i in range(50):
        a = rnn.get_byte_number(i)
        b = rnn.get_byte_number(i+1)
        print("test "+ str(i))
        print(a.value())
        print(b.value())
        print("Real Value "+str(a.value()*b.value()))
        print("Test Value "+str((a*b).value()))
        print("\n")
        
test_mul()




