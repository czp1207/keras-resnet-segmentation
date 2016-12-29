from keras.models import Model
from keras.layers import *
from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def PSPNET(input_shape = None, batch_shape = None):
    if batch_shape:
        img_input = Input(batch_shape = batch_shape)
    else:
        img_input = Input(input_shape = input_shape)
    conv1_1_s2 = Convolution2D(64, 3, 3, init="he_normal", border_mode="same", subsample=(2, 2), name='conv1_1_3x3_s2')(img_input)
    conv1_1_s2_bn = (mode=0, axis=3, momentum=0.95)(conv1_1_s2)
    conv1_1_s2_bn = Activation('relu')(conv1_1_s2_bn)
    
     
