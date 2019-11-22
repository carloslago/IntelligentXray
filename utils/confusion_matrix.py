from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from mlxtend.plotting import plot_confusion_matrix


model = load_model('best_network.h5')

img_path = os.path.join('chest_xray/test')
normal = 0
neumonia = 0
should_be = 0
false_positives = 0
false_negatives = 0
for subdir, dirs, files in os.walk(img_path):
    if "NORMAL" in subdir or "PNEUMONIA" in subdir:
        if "PNEUMONIA" in subdir: should_be=1
        else: should_be=0
        for file in files:
            path_to = os.path.join(subdir, file)
            img = image.load_img(path_to, target_size=(224, 224), grayscale=True)
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            res = model.predict(img_tensor)
            if res>0.5:
                if should_be==0: false_positives+=1
                neumonia += 1
            else:
                if should_be==1: false_negatives+=1
                normal += 1

# normal = 364
# neumonia = 1039
# false_positives = 108
# false_negatives = 37

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