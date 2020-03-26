import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
import csv
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import collections

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    shuffle=False,
    batch_size=8)

# print(valid_set.head)
valid_set = valid_set.drop(["Path"], axis=1)
path_model = os.path.join('saved_models/best_model_all_03_25_2020_21_35_48.h5')
model = tf.keras.models.load_model(path_model, compile=False)
# print(model.summary())

Y_pred = model.predict_generator(validation_generator, len(valid_set) / 8)
# print(Y_pred)
# y_pred = np.argmax(Y_pred, axis=1)
y_pred = np.around(Y_pred)

y_test = valid_set.values

cm = multilabel_confusion_matrix(y_test, y_pred)
tot = 0
accs = {}

for i in range(len(types)): types[i] = types[i][0:15]
for i in range(len(cm)):
    good = cm[i][0][0]+cm[i][1][1]
    total = sum(cm[i][0]) + sum(cm[i][1])
    acc = good/total
    accs[types[i]] = acc
    tot += acc
    # print("Pathologie %s - %.2f" % (types[i], acc))
    # print(cm[i])
    # print('\n')

    # plt.figure()
    # plot_confusion_matrix(cm[i])
    # plt.xticks(range(2), ['Normal', types[i]], fontsize=16)
    # plt.yticks(range(2), ['Normal', types[i]], fontsize=16)
    # plt.show()


sorted_acc = sorted(accs.items(), key=lambda k: k[1])
# sorted_acc = accs.items()
accs = collections.OrderedDict(sorted_acc)

print("Overall acc %.5f"%(float(tot/len(types))))
print(list(accs.values()))

plt.barh(list(accs.keys()), list(accs.values()), align='center')
plt.yticks(list(accs.keys()))
plt.xlabel('Accuracy')
plt.title('Model performance by pathology')
plt.xlim(right=1, left=0)

plt.show()







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