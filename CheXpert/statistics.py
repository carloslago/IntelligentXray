from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from mlxtend.plotting import plot_confusion_matrix
import csv

path_model = os.path.join('saved_models/best_model_lateral_11_23_2019_00_41_44')
model = load_model(path_model)

csv_valid = os.path.join("CheXpert-v1.0-small/csv/original/valid_lateral.csv")
# normal = 0
# neumonia = 0
# should_be = 0
# false_positives = 0
# false_negatives = 0
with open(csv_valid) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cont = 0
    for row in csv_reader:
        path_to = row[0]
        img = image.load_img(path_to, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        res = model.predict(img_tensor)
        should_be = row[1:]
        print(res)
        print(should_be)
        correct = 0
        for i in range(len(res)):
            if round(res[i]) == should_be[i]:
                correct+=1
        acc = correct/len(res)
        print(acc)
        print('\n')


# normal = 364
# neumonia = 1039
# false_positives = 108
# false_negatives = 37
'''
print("Normal predicted: %d, Neumonia predicted: %d" % (normal, neumonia))
print("False positives: %d, False negatives: %d" % (false_positives, false_negatives))
total_normal = normal + false_positives - false_negatives
total_pneumonia = neumonia + false_negatives - false_positives

#Confusion matrix
cm = np.array([[normal-false_negatives, false_positives], [false_negatives, neumonia-false_positives]])
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

'''