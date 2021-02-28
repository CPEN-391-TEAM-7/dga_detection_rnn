from model.rnn import RNN

model_path = "model_export/rnn_binary_model_0210.h5"
export_path = "model_export/model_bininary_0228.bin"
export_tt_path = "model_export/model_bininary_0210.txt"

model = RNN()

#model.save_model_txt(export_tt_path, modelPath=model_path)
model.save_model_bin(export_path, modelPath=model_path)

