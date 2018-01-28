from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense,Conv2D,Flatten ,Conv3D , MaxPooling3D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.utils import to_categorical


def create_model():
    inp = Input(shape=(32, 32, 32, 4))

    Conv_3d_1 = Conv3D( filters=6, kernel_size = (6, 6, 6), strides = (1, 1, 1), padding = 'valid',
                        activation = 'relu', use_bias = True)(inp)

    max_pool_1 = MaxPooling3D( pool_size = (2, 2, 2), strides = (1, 1, 1), padding = 'valid')(Conv_3d_1)

    Conv_3d_2 = Conv3D(filters = 3, kernel_size = (4, 4, 4), strides = (1, 1, 1), padding = 'valid',
                       activation = 'relu', use_bias = True)(max_pool_1)

    max_pool_2 = MaxPooling3D( pool_size = (2, 2, 2), strides = (1, 1, 1), padding = 'valid')(Conv_3d_2)

    Conv_3d_3 = Conv3D(filters=2, kernel_size=(4, 4, 4), strides=(1, 1, 1), padding='valid',
                       activation='relu', use_bias=True)(max_pool_2)

    max_pool_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid')(Conv_3d_3)

    flatten1 = Flatten()(max_pool_3)

    out = Dense(2, activation='sigmoid')(flatten1)

    model = Model(inputs=inp, outputs=out)
    
    return model