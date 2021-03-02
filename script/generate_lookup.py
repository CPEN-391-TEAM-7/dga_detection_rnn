import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')
from fxpmath import Fxp
import numpy as np

tanh_table_path = "conf/tanh_table.bin"
with open(tanh_table_path, 'wb') as binFile:
    for entry in range(10):
        xVal = -4 + entry * 8 / 9
        print(xVal)
        x_fxp = Fxp(xVal, signed=True, n_word=16, n_frac=12)
        print(x_fxp.bin())
        xTanh = np.tanh(xVal)
        print(xTanh)
        xTanh_fxp = Fxp(xTanh, signed=True, n_word=16, n_frac=12)
        print(xTanh_fxp.bin())
        binFile.write(x_fxp.bin().encode(encoding="ascii", errors="unknown char"))
        binFile.write(xTanh_fxp.bin().encode(encoding="ascii", errors="unknown char"))