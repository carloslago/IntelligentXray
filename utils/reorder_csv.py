import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())


types = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']
types_index = []


option = 'all'
known_frontal = 2
known_lateral = 1
csv_path = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\original\\train.csv'
# csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\pathologies\\train_'+option+'_'+str(known_frontal)\
#             +'_'+str(known_lateral)+'.csv'
csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\pathologies\\train_'+option+'_'+str(known_frontal)\
            +'_'+str(known_lateral)+'_mix.csv'
# csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\original\\valid_all.csv'
# csv_write = str(p.parents[0]) + '\\CheXpert\\CheXpert-v1.0-small\\csv\\pathologies\\valid_all.csv'

# u_zeros = [0, 2, 3, 4, 6, 10, 12, 13]
# u_ones = [1, 5, 7, 8, 9, 11]
u_zeros = [i for i in range(14)]
u_ones = []

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
            temp = row
            for i in range(0,len(temp)):
                formatted = temp[i].replace(' ', '_')
                if formatted in types or formatted=="Path":
                    pathologies.append(formatted)
                    types_index.append(i)
            imgs.append(pathologies)
        else:
            data = row
            new = [data[0]]
            known = known_frontal
            if "frontal" in data[0]: known = known_frontal
            else: known = known_lateral
            certain = 0
            if option in data[0] or option=="all":
                for i in range(1,len(data)):
                    if i in types_index:
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
                        new.append(data[i])
                    else:
                        pass
                if certain>=known and certain!=5:
                    imgs.append(new)
            certain = 0
        # if cont>100: break
        cont+=1
    csv_file.close()

print(cont)
print(len(imgs))
# exit()
with open(csv_write, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(imgs)
    csvFile.close()
