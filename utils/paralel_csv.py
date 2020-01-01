import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())

option = 'valid'
csv_path = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\original\\%s.csv'%option
csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\paralel\\%s_paralel_frontal.csv'%option
csv_write2 = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\paralel\\%s_paralel_lateral.csv'%option

default_frontal = 'CheXpert-v1.0-small/averages/median_frontal.jpg'
default_lateral = 'CheXpert-v1.0-small/averages/median_lateral.jpg'

# u_zeros = [0, 2, 3, 4, 6, 10, 12, 13]
# u_ones = [1, 5, 7, 8, 9, 11]
u_ones = []

pathologies = []
cont = 0
positive = 1.0
uncertain = -1.0
negative = 0.0
imgs_frontal = []
imgs_lateral = []
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cont = 0
    for row in csv_reader:
        del row[1:5]
        if cont ==0:
            pathologies = row
            for i in range(1,len(pathologies)):
                pathologies[i] = pathologies[i].replace(' ', '_')
            imgs_frontal.append(pathologies)
            imgs_lateral.append(pathologies)
        else:
            data = row
            certain = 0
            for i in range(1,len(data)):
                if len(data[i]) == 0:
                    data[i] = 0
                elif int(float(data[i])) == positive:
                    data[i] = 1
                    certain +=1
                elif int(float(data[i])) == uncertain:
                    if i in u_ones:
                        data[i] = 1
                    else:
                        data[i] = 0
                else:
                    certain+=1
                    data[i] = 0
            if certain>4:
                if 'frontal' in row[0]:
                    imgs_frontal.append(data)
                else:
                    imgs_lateral.append(data)
            certain = 0
        # if cont>100: break
        cont+=1
    csv_file.close()

final_frontal = []
final_lateral = []

cont = 0
for i in imgs_frontal:
    if cont == 0:
        final_frontal.append(i)
        final_lateral.append(i)
    else:
        final_frontal.append(i)
        path_test = i[0].split('view')[0]
        found = 0
        for e in imgs_lateral:
            if path_test in e[0] and found==0:
                found = 1
                final_lateral.append(e)
        if found == 0:
            data = [default_lateral]+i[1:]
            final_lateral.append(data)
    cont+=1

for i in imgs_lateral:
    if i not in final_lateral:
        data = [default_frontal] + i[1:]
        final_frontal.append(data)
        final_lateral.append(i)

with open(csv_write, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(final_frontal)
    csvFile.close()
with open(csv_write2, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(final_lateral)
    csvFile.close()