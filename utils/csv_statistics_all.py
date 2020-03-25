import json

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())

csv_path = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\original\\train.csv'
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
            pathologies = row[6:-1]
            statistics = [[0, 0, 0] for x in pathologies]
        else:
            data = row[6:-1]
            for i in range(len(data)):
                if len(data[i]) > 0:
                    if float(data[i]) == 1.0:
                        statistics[i][0] += 1
                    elif float(data[i]) == -1.0:
                        statistics[i][1] += 1
                    else:
                        statistics[i][2] += 1
                else:
                    # pass
                    statistics[i][2] += 1
        # if cont>10: break
        cont+=1
    csv_file.close()

dict = {}
for e in range(len(pathologies)):
    dict[pathologies[e]] = statistics[e]
print(json.dumps(dict, indent = 4))
# exit()

positive = [x[0] for x in statistics]
negative = [x[2] for x in statistics]
uncertain = [x[1] for x in statistics]
ind = np.arange(len(pathologies))
width = 0.3

plt.figure(figsize=(26, 10))
p1 = plt.bar(ind, positive, width)
p2 = plt.bar(ind, negative, width,
             bottom=positive)
p3 = plt.bar(ind, uncertain, width,
             bottom=np.array(positive)+np.array(negative))
plt.ylabel('Tests', fontsize=14)
plt.title('Positive, negative and uncertain samples by pathology', fontsize=20)
# x_labels = [p[0:5] for p in pathologies] Names shorter
x_labels = pathologies
plt.xticks(ind, x_labels, fontsize=12)
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Positive', 'Negative', 'Uncertain'), loc=1, fontsize=18)
plt.show()

