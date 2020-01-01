import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.preprocessing import image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries


def img_to_tensor(img):
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

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
    shuffle=False,
    batch_size=8)


path_model = os.path.join('saved_models/best_model_lateral_11_30_2019_12_42_06.h5')
model = tf.keras.models.load_model(path_model, compile=False)

x,y = generator.next()
for e in range(0,4):
    img = x[e]
    # plt.imshow(x[e] / 2 + 0.5)
    # plt.imshow(img)
    # plt.show()
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img, model.predict, labels=types, hide_color=0, num_samples=1000)
    # img_tensor = img_to_tensor(img)
    # res = model.predict(img_tensor)
    # res = np.around(res)
    # res = res.astype(int)[0]
    for i in explanation.top_labels:
        print("Predicted: %d, Original: %d, Pathologie: %s"%(int(3), int(y[e][i]),types[i]))
        temp, mask = explanation.get_image_and_mask(i, positive_only=False, num_features=5,
                                                    hide_rest=False
                                                    # ,min_weight=0.05
                                                     )
        plt.title(types[i])
        plt.imshow(x[e] / 2 + 0.5)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.show()
    break

    # res = model.predict(img_tensor)
    # acc = 0
    # res = np.around(res)
    # res = res.astype(int)[0]
    # for e in range(len(res)):
    #     if int(res[e]) == int(y[i][e]):
    #         acc += 1
    # acc /= len(y[i])
    # print(res)
    # print(y[i])
    # print(acc)
    # print('\n')


