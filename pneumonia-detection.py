# %%
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import tensorflow as tf

# %%
train_dir = r'C:\Users\lenovo_laptop\MinorProjectFolder\archive\chest_xray\chest_xray\train'
val_dir = r'C:\Users\lenovo_laptop\MinorProjectFolder\archive\chest_xray\chest_xray\val'
test_dir = r'C:\Users\lenovo_laptop\MinorProjectFolder\archive\chest_xray\chest_xray\test'

# %%
IMG_HEIGHT = 224
IMG_WIDTH = 224

BATCH_SIZE = 32

# %%
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                               rotation_range = 20,
                                                               brightness_range= (1.2, 1.5),
                                                               horizontal_flip = True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

# %%
train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size =(IMG_HEIGHT, IMG_WIDTH),
                                              class_mode = 'binary',
                                              batch_size = BATCH_SIZE)
val_data = val_datagen.flow_from_directory(val_dir,
                                              target_size =(IMG_HEIGHT, IMG_WIDTH),
                                              class_mode = 'binary',
                                              batch_size = BATCH_SIZE)
test_data = test_datagen.flow_from_directory(test_dir,
                                              target_size =(IMG_HEIGHT, IMG_WIDTH),
                                              class_mode = 'binary',
                                              batch_size = BATCH_SIZE)

# %%
mobilenet = tf.keras.applications.MobileNetV2(input_shape = (IMG_HEIGHT,IMG_WIDTH, 3),
                                              include_top = False,
                                              weights = 'imagenet',
                                              pooling='avg')

mobilenet.trainable = False

# %%
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

pretrained_model = mobilenet(inputs, training=False)

dense = tf.keras.layers.Dense(1024, activation="relu")(pretrained_model)
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(dense)

model = tf.keras.Model(inputs, outputs)


print(model.summary())

# %%
# EPOCHS = 10

# model.compile(optimizer = 'adam',
#               loss = 'binary_crossentropy',
#              metrics = ['accuracy',
#                        tf.keras.metrics.AUC(name='auc')])

# history = model.fit(train_data,
#                    validation_data = val_data,
#                    batch_size=BATCH_SIZE,
#                    epochs = EPOCHS,
#                    callbacks = None)

# %%
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
             metrics = ['accuracy',
                       tf.keras.metrics.AUC(name='auc')])
EPOCHS = 20
filepath = r'C:\Users\lenovo_laptop\MinorProjectFolder\archive\chest_xray\temporary\model-ep{epoch:03d}-loss{loss:.3f}-accuracy{accuracy:.3f}.h5'


checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit(train_data,
          validation_data = val_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS, 
          callbacks=[checkpoint])

# %%
true_labels = test_data.labels
pred_labels = np.squeeze(np.array(model.predict(test_data) >= 0.5, dtype =np.int))

cm = confusion_matrix(true_labels, pred_labels)

# %%
test_data.class_indices

# %%
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='mako', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(np.arange(2) + 0.5, ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(np.arange(2) + 0.5, ['Normal', 'Pneumonia'], fontsize=16)
plt.show()

# %%
results = model.evaluate(test_data, verbose = 0)

# %%
accuracy = results[1]
auc = results[2]

# %%
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)

# %%
print("Accuracy: {:.2f}".format(accuracy))
print("AUC: {:.2f}".format(auc))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# %%



