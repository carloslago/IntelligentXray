import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

train_dir = os.path.join('dataset/train')
test_dir = os.path.join('dataset/test')
validation_dir = os.path.join('dataset/val')


train_names = os.listdir(train_dir)

datagen = ImageDataGenerator(rescale=1/255)
training_datagen = ImageDataGenerator(rescale=1/255, rotation_range=30, zoom_range=0.3)

train_generator = training_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)

test_generator = datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)

validation_generator = datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# print(model.summary())



model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-5),
              metrics=['acc'])

# Fitting the model, in this case using the test_generator to plot afterwards the results.
history = model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=67,
                    validation_steps=22,
                    epochs= 15,
                    verbose = 1
)

model.save('saved_models/t_network.h5')


# Plot training and test acc and loss
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




'''ROC curve
model = load_model('best_network.h5')

X,Y = test_generator.next()
y_pred_keras = model.predict(X).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()'''