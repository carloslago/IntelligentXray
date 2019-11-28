from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from mlxtend.plotting import plot_confusion_matrix
import csv
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix

types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

datagen = ImageDataGenerator(rescale=1. / 255)
valid_set = pd.read_csv("CheXpert-v1.0-small/csv/original/valid_all.csv")
validation_generator = datagen.flow_from_dataframe(
    dataframe=valid_set,
    directory="",
    x_col="Path",
    y_col=types,
    classes=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(224, 224),
    batch_size=8)

# print(valid_set.head)
valid_set = valid_set.drop(["Path"], axis=1)
path_model = os.path.join('saved_models/best_model_all_11_23_2019_12_05_57.h5')
model = load_model(path_model)

Y_pred = model.predict_generator(validation_generator, len(valid_set) / 8)
# print(Y_pred)
# y_pred = np.argmax(Y_pred, axis=1)
y_pred = np.around(Y_pred)

y_test = valid_set.values

cm = multilabel_confusion_matrix(y_test, y_pred)
for i in range(len(cm)):
    good = cm[i][0][0]+cm[i][1][1]
    total = sum(cm[i][0]) + sum(cm[i][1])
    print("Pathologie %s - %.2f" % (types[i], good/total))
    print(cm[i])
    print('\n')
    # plt.figure()
    # plot_confusion_matrix(cm[i])
    # plt.xticks(range(2), ['Normal', types[i]], fontsize=16)
    # plt.yticks(range(2), ['Normal', types[i]], fontsize=16)
    # plt.show()





'''
csv_valid = os.path.join("CheXpert-v1.0-small/csv/original/valid_lateral.csv")
tot_acc = 0
with open(csv_valid) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cont = 0
    for row in csv_reader:
        if cont > 0:
            path_to = os.path.join(row[0])
            img = image.load_img(path_to, target_size=(224, 224))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            res = model.predict(img_tensor)[0]
            should_be = row[1:]
            print(res)
            print(should_be)
            correct = 0
            for i in range(len(res)):
                if round(res[i]) == int(should_be[i]):
                    correct+=1
            acc = correct/len(res)
            tot_acc += acc
            print(acc)
            print('\n')
        else:
            pass
        cont += 1

tot_acc /= cont
print(tot_acc)
'''