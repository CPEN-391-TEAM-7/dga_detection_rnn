import csv
import random

dataPath = './data/dga.txt'
resultPath = './data/binary_training.txt'

data_out = []

#DGA is marked with 1, whitelist websites are marked with 0
with open(dataPath, "r") as f:
    data = f.read().split('\n')

for line in data:
    if len(line) == 0 or line[0] == '#' or line[0] == ' ':
        continue
    data_out.append(line.split('\t')[1].lower().strip(' ') + ',1')

batch_size = len(data_out)
print(batch_size)

with open('./data/majestic_million.csv') as whiteList:
    readCSV = csv.reader(whiteList, delimiter=',')
    for row in readCSV:
        if row[1] == 'Domain':
            continue
        data_out.append(row[2]+',0')
        if len(data_out) >= batch_size*2 :
            break

random.shuffle(data_out)

f_result = open(resultPath, "a")
for a in data_out:
    f_result.write(a + "\n")
