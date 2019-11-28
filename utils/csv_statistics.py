import json

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())

csv_path = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\top\\train_frontal_6.csv'
statistics = []
pathologies = []
cont = 0

# types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
#          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
#          'Fracture', 'Support_Devices']

with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cont = 0
    for row in csv_reader:
        if cont ==0:
            pathologies = row[1:]
            statistics = [[0,0] for x in pathologies]
        else:
            data = row[1:]
            for i in range(len(data)):
                if int(data[i]) == 1:
                    statistics[i][0] += 1
                else:
                    statistics[i][1] += 1
        # if cont>10: break
        cont+=1
    csv_file.close()

dict = {}
for e in range(len(pathologies)):
    dict[pathologies[e]] = statistics[e]
# print(json.dumps(dict, indent = 4))

positive = [x[0] for x in statistics]
negative = [x[1] for x in statistics]
ind = np.arange(len(pathologies))
width = 0.3
p1 = plt.bar(ind, positive, width)
p2 = plt.bar(ind, negative, width,
             bottom=positive)

plt.ylabel('Tests')
plt.title('Positive and negative samples by pathologie')
x_labels = [p[0:5] for p in pathologies]
plt.xticks(ind, x_labels)
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))

plt.show()

