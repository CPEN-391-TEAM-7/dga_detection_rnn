import sys
sys.path.insert(1, '/Users/jingyuan/Desktop/dga/dga_detection_rnn')

dataPath = 'data/binary_training.txt'
resultPath = 'test_result/binary_test.txt'

with open(dataPath, "r") as f1:
    data = f1.read().split('\n')

with open(resultPath, "r") as f2:
    result = f2.read().split('\n')


wrong = 0
correct = 0
for i, line in enumerate(data):
    if len(line) == 0 or line[0] == '#' or line[0] == ' ':
        continue
    if line.split(',')[1] != result[i]:
        wrong += 1
        print(line.split(',')[0]+' '+result[i])
    else:
        correct += 1

print("total accuracy")
print(str(correct/(wrong+correct)*100)+'%')
