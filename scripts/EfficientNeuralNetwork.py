import os 
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import voxelmorph as vxm
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

def enet_block(inputs, filters, kernel_size, batch_norm=True, dropout=None):
    x = inputs
    if batch_norm:
        x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if dropout:
        x = Dropout(dropout)(x)
    return x

def enet(inputs, num_classes):
    x = inputs
    x = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    x = enet_block(x, 64, 3, batch_norm=True, dropout=0.1)
    x = enet_block(x, 64, 3, batch_norm=True)
    x = MaxPooling2D()(x)
    x = enet_block(x, 128, 3, batch_norm=True, dropout=0.1)
    x = enet_block(x, 128, 3, batch_norm=True)
    x = MaxPooling2D()(x)
    x = enet_block(x, 256, 3, batch_norm=True, dropout=0.1)
    x = enet_block(x, 256, 3, batch_norm=True)
    x = MaxPooling2D()(x)
    x = enet_block(x, 512, 3, batch_norm=True, dropout=0.1)
    x = enet_block(x, 512, 3, batch_norm=True)
    x = MaxPooling2D()(x)
    x = enet_block(x, 1024, 3, batch_norm=True, dropout=0.1)
    x = enet_block(x, 1024, 3, batch_norm=True)
    x = UpSampling2D()(x)
    x = enet_block(x, 512, 3, batch_norm=True)
    x = enet_block(x, 512, 3, batch_norm=True)
    x = UpSampling2D()(x)
    x = enet_block(x, 256, 3, batch_norm=True)
    x = enet_block(x, 256, 3, batch_norm=True)
    x = UpSampling2D()(x)
    x = enet_block(x, 128, 3, batch_norm=True)
    x = enet_block(x, 128, 3, batch_norm=True)
    x = UpSampling2D()(x)
    x = enet_block(x, 64, 3, batch_norm=True)
    x = enet_block(x, 64, 3, batch_norm=True)
    x = Conv2D(num_classes, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)
    return x

input_shape = (256, 256, 3)
num_classes = 1

inputs = Input(shape=input_shape)
outputs = enet(inputs, num_classes)
model = Model(inputs, outputs)

model_dir = '/content/drive/MyDrive/data'
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=50)
epochs = 201
loss = 'binary_crossentropy'

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
model.fit(Xtrain, ytrain, validation_split=0.2, batch_size=5, verbose=True, epochs=epochs, callbacks=callbacks)
model.save(save_filename.format(epoch=epochs))
