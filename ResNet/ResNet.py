# -*- coding: utf-8 -*-
import keras.backend as K
from keras import layers, optimizers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, \
    ZeroPadding2D, Input
from keras.models import Model

__author__ = 'Rainer Arencibia'
__copyright__ = 'Copyright 2017, Rainer Arencibia'
__credits__ = 'ImageNet Team, ResNet Team, Keras Team and TensorFlow Team'
__license__ = 'MIT'
__version__ = '1.00'
__status__ = 'Prototype'


def block_without_shortcut(model, filter, stage, block, index=1):

    conv_name_base = 'res' + str(stage) + block + str(index) + '_branch'
    bn_name_base = 'bn' + str(stage) + block + str(index) + '_branch'

    x = Conv2D(filter, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2a')(model)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter * 4, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, model])
    x = Activation('relu')(x)
    return x


def block_with_shortcut(model, filter, stage, block, strides=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(model)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter*4, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter*4, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1')(model)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class ResNet:

    def __init__(self):

        self.model = None

    def build_resnet(self, input_shape=None, num_outputs=1000, layers=None, weights_path=None):
        """
        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels) TensorFlow Format!!
            num_outputs: The number of outputs at final softmax layer
            layers: Number of layers for every network 50, 101, 152
            weights_path: URL to the weights of a pre-trained model.
            optimizer: An optimizer to compile the model, if None sgd+momentum by default.
        Returns:
            A compile Keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple like (nb_rows, nb_cols, nb_channels)")

        input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=197,
                                          data_format=K.image_data_format(), include_top=True)
        img_input = Input(shape=input_shape)
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

        nb_filters = 64
        stage = 2
        for e in layers:
            for i in range(e):
                if i == 0:
                    x = block_with_shortcut(x, nb_filters, stage=stage, block='a', strides=2 if stage >= 3 else 1)
                else:
                    x = block_without_shortcut(x, nb_filters, stage=stage, block='b', index=i)
            stage += 1
            nb_filters *= 2

        x = AveragePooling2D((7, 7), strides=(1, 1), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(units=num_outputs, activation='softmax', name='fc1000')(x)
        self.model = Model(inputs=img_input, outputs=x, name='ResNet Model')

        if weights_path is not None:
            model.load_weights(weights_path)

        return self.model

    def compile(self, optimizer='sgd'):

        optimizer_dicc = {'sgd': optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                          'rmsprop': optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                          'adagrad': optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
                          'adadelta': optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                          'adam': optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)}

        model.compile(optimizer=optimizer_dicc[optimizer], loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, image_batch):
        import numpy as np

        predictions = []
        for image in image_batch:
            pred = model.predict(image)
            predictions.append(pred)

        return np.asarray(predictions)

    """ FLOPs:
    50 layers  -> 3.8 x 10^9
    101 layers -> 7.6 x 10^9
    152 layers -> 11.3 x 10^9
    """
    def build_resnet_50(self, input_shape, num_outputs, weights):
        return self.build_resnet(input_shape, num_outputs, layers=[3, 4, 6, 3], weights_path=weights)

    def build_resnet_101(self, input_shape, num_outputs, weights, opt):
        return self.build_resnet(input_shape, num_outputs, layers=[3, 4, 23, 3], weights_path=weights)

    def build_resnet_152(self, input_shape, num_outputs, weights, opt):
        return self.build_resnet(input_shape, num_outputs, layers=[3, 8, 36, 3], weights_path=weights)


def process(url='./url/2/image.jpg'):

    import cv2
    import numpy as np
    img = cv2.imread(url)
    img = cv2.resize(img, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(np.asarray(img, dtype='float64'))
    return img  # img.shape (1, 224, 224, 3) type float64 ; BGR format by default from OpenCV

if __name__ == '__main__':

    img = process(url='./input/lena.png')
    resnet = ResNet()
    model = resnet.build_resnet_50(input_shape=img[0].shape, num_outputs=1000,
                                    weights='./weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.compile(optimizer='sgd')
    # './weights/resnet_v1_152.ckpt'

    from utils.Show import Show
    Show.show_predictions(model, img)
