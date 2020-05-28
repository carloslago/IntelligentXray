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
from tensorflow.keras.utils import plot_model


def img_to_tensor(img):
    img = image.load_img(img, target_size=(224, 224), color_mode="rgb")
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    temp = img_tensor[0]
    return img_tensor, temp


types = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']
true = [0, 0, 1, 0, 0]


path_model = os.path.join('saved_models/best_model_all_05_10_2020_13_06_50.h5')
model = tf.keras.models.load_model(path_model, compile=False)

tensor, img = img_to_tensor("CheXpert-v1.0-small/to_explain/consolidation_lateral.jpeg")
plt.imshow(img)
plt.show()
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img, model.predict, labels=types, hide_color=0, num_samples=1000, top_labels=5)
res = model.predict(tensor)
res = np.around(res)
res = res.astype(int)[0]
for i in explanation.top_labels: # Gets more interesting label positions
    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5,
                                                hide_rest=False
                                                ,min_weight=0.08
                                                 )
    plt.title("Predicted: %d, Original: %d, Pathologie: %s"%(int(res[i]), int(true[i]),types[i]))
    # plt.imshow(x[e] / 2 + 0.5)
    plt.imshow(img)
    plt.imshow(mark_boundaries(temp, mask))
    plt.show()


