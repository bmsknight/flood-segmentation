from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose, concatenate
import keras.backend as K
import spektral
from spektral.layers import  GATConv, ChebConv
from keras.layers import Reshape
from spektral.utils import sp_matrix_to_sp_tensor
# Define custom layers to be used in the model
#custom_objects = {'GATConv': GATConv, 'ChebConv': ChebConv}

# Encoder
input_shape = (256, 256, 3)
num_filters = 64

inputs = Input(input_shape)
conv1 = Conv2D(num_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(num_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(num_filters*2, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(num_filters*2, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(num_filters*4, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(num_filters*4, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(num_filters*8, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(num_filters*8, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(num_filters*10, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(num_filters*10, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv5)
print(f'conv5 shape {conv5.shape}')
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

#between encoder and decoder
pool5_shape = pool5.shape.as_list()
num_nodes = pool5_shape[1] * pool5_shape[2]
n_features = pool5_shape[3] 
channels = num_filters*10
a = np.ones((num_nodes, num_nodes))
pool5_reshaped = Reshape((-1, num_filters*10))(pool5)
a_in = sp_matrix_to_sp_tensor(a)

#Graph Attention Networks
gatconv = spektral.layers.GATConv(channels, attn_heads=4,
                                  concat_heads=True,dropout_rate=0.5,
                                  return_attn_coef=False, add_self_loops=True,
                                  activation='relu', use_bias=True,
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  attn_kernel_initializer='glorot_uniform', kernel_regularizer=None,
                                  bias_regularizer=None, attn_kernel_regularizer=None,
                                  activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None, attn_kernel_constraint=None)([pool5_reshaped, a_in])
print(f'gatconv shape {gatconv.shape}')
#A Chebyshev convolutional layer
chebconv = spektral.layers.ChebConv(channels, K=4, 
                                    activation='relu',use_bias=True,
                                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                    kernel_regularizer=None, bias_regularizer=None,
                                    activity_regularizer=None, kernel_constraint=None,
                                    bias_constraint=None)([gatconv, a_in])
chebconv = tf.keras.layers.LayerNormalization()(chebconv)
com = CenterOfMass()(chebconv)
chebconv = Reshape((pool5_shape[1], pool5_shape[2], num_filters*10))(com)
print(f'chebconv shape after reshape {chebconv.shape}')

# Decoder
up_filters = num_filters*10
up6 = Conv2DTranspose(up_filters, 16, strides=(2,2), padding='same', kernel_initializer='he_normal')(chebconv)
print(up6.shape)
up6 = concatenate([up6, conv5])
conv6 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(up6)
conv6 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv6)
print(f'conv6:{conv6}')

up_filters = num_filters*8
up7 = Conv2DTranspose(up_filters, 16, strides=(2,2), padding='same', kernel_initializer='he_normal')(conv6)
up7 = concatenate([up7, conv4])
conv7 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(up7)
conv7 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv7)
print(f'conv7:{conv7}')


up_filters = num_filters*4
up8 = Conv2DTranspose(up_filters, 2, strides=(2,2), padding='same', kernel_initializer='he_normal')(conv7)
up8 = concatenate([up8, conv3])
conv8 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(up8)
conv8 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv8)
print(f'conv8:{conv8}')


up_filters = num_filters*2
up9 = Conv2DTranspose(up_filters, 2, strides=(2,2), padding='same', kernel_initializer='he_normal')(conv8)
up9 = concatenate([up9, conv2])
conv9 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(up9)
conv9 = Conv2D(up_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv9)
conv9.shape


up10 = Conv2DTranspose(num_filters, 2, strides=(2,2), padding='same', kernel_initializer='he_normal')(conv9)
up10 = concatenate([up10, conv1])
conv10 = Conv2D(num_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(up10)
conv10 = Conv2D(num_filters, 3, activation='ReLU', padding='same', kernel_initializer='he_normal')(conv10)
print(f'conv10:{conv10}')
outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
modelUNet = Model(inputs=inputs, outputs=outputs)
modelUNet.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
print(modelUNet.summary())
