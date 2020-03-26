import numpy as np
import os
import csv
from pathlib import Path, PureWindowsPath
from shutil import copy

p = PureWindowsPath(Path().absolute())

csv_path = "dataset/metadata.csv"
csv_write = "dataset/cleaned.csv"
def clean_csv():
    imgs = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cont = 0
        for row in csv_reader:
            if cont == 0:
                header = ["path", "covid", "no-finding", "pneuomonia"]
                imgs.append(header)
            else:
                present = 0
                normal = 1
                if row[6] != "Axial":
                    if row[4] == "COVID-19":
                        present = 1
                        normal = 0
                    else:
                        present = 0
                        normal = 1
                    path = "dataset/images/"+row[10]
                    imgs.append([path, present, normal, 0])
            cont += 1

    with open(csv_write, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(imgs)
        csvFile.close()

def append_imgs():
    option = "PNEUMONIA"
    if option == "NORMAL": diagnosis = [0,1,0]
    else: diagnosis = [0,0,1]
    path_files = str(p.parents[0]) + '\\Pneumonia\\dataset\\train\\'+option+'\\'
    path_write = "dataset/images/"
    cont = 0
    imgs = []
    for file in os.listdir(path_files):
        if cont > 120: break
        copy(path_files+file, path_write)
        imgs.append([path_write+file, diagnosis[0], diagnosis[1], diagnosis[2]])
        cont += 1

    with open('dataset/'+option+'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(imgs)
        csvFile.close()

def main():
    # clean_csv()
    append_imgs()

main()