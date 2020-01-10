import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

BATCH_SIZE = 32
datagen = ImageDataGenerator(rescale=1. / 255)
set = pd.read_csv("CheXpert-v1.0-small/csv/top/train_top_lateral.csv")
generator = datagen.flow_from_dataframe(
    dataframe=set,
    directory="",
    x_col="Path",
    y_col=types,
    classes=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(224, 224),
    shuffle=True,
    batch_size=BATCH_SIZE)

model = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False)

steps = len(set) / BATCH_SIZE
feature_list = []
labels_list = []
cont = 1
for thing in generator:
    for i in range(len(thing[0])):
        img = thing[0][i]
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        res = model.predict(img_tensor)
        feature = np.array(res)
        feature_list.append(feature.flatten())
        labels_list.append(thing[1][i])
    cont+=1
    if cont>steps: break

feature_list = np.array(feature_list)
labels_list = np.array(labels_list)

print("TO TRAINING")
#SVC or LogisticRegression problem transformation methods including Binary Relevance, Label Powerset and Classifier Chains
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(feature_list, labels_list)
clf = OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=3000, verbose=1)).fit(feature_list, labels_list)
score = clf.score(feature_list, labels_list)
print(score)