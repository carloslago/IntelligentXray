import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())

option = 'frontal'
csv_path = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\original\\train.csv'
csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\top\\train_'+option+'_5.csv'

pathologies = []
cont = 0
cont_front = 0
cont_lat = 0
# img_frontales = []
# img_laterales = []
positive = 1.0
uncertain = -1.0
negative = 0.0
imgs = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cont = 0
    for row in csv_reader:
        del row[1:5]
        if cont ==0:
            pathologies = row[1:]
            for i in range(len(pathologies)):
                pathologies[i] = pathologies[i].replace(' ', '_')
            imgs.append(pathologies)
        else:
            data = row
            certain = 0
            if option in data[0]:
                for i in range(1,len(data)):
                    if len(data[i]) == 0:
                        data[i] = 0
                    elif int(float(data[i])) == positive:
                        data[i] = 1
                        certain +=1
                    elif int(float(data[i])) == uncertain:
                        data[i] = 0
                    else:
                        certain+=1
                        data[i] = 0
            if certain>4:
                imgs.append(data)
            certain = 0
        # if cont>100: break
        cont+=1
    csv_file.close()

# print(len(imgs))
with open(csv_write, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(imgs)
    csvFile.close()
