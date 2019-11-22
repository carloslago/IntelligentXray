
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras import models, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path, PureWindowsPath

p = PureWindowsPath(Path().absolute())
model = str(p.parents[0]) + '\\Pneuomonia\\saved_models\\first_network.h5'



train_dir = os.path.join('dataset/ChinaSet_AllFiles/train')
test_dir = os.path.join('dataset/ChinaSet_AllFiles/test')

train_names = os.listdir(train_dir)

datagen = ImageDataGenerator(rescale=1 / 255)
training_datagen = ImageDataGenerator(rescale=1/255, zoom_range=0.3, rotation_range=30)

train_generator = datagen.flow_from_directory(
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

model = load_model(model)

for layer in model.layers:
    if "core" not in str(layer): # Set convnet layers not trainable
        layer.trainable = True


# Changing dense layers
hidden = keras.layers.Dense(512, activation='sigmoid', name='my_dense3')(model.layers[-4].output)
# hidden2 = Dropout(0.5)(hidden)
new_layer = keras.layers.Dense(1, activation='sigmoid', name='my_dense')(hidden)
inp = model.input
# out = new_layer(model.layers[-2].output)
model2 = Model(inp, new_layer)

model2.summary()

model2.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-5),
              metrics=['acc'])

history = model2.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=8,
                    validation_steps=3,
                    epochs= 10,
                    verbose = 1
)


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
