# -*- coding: utf-8 -*-
import numpy as np
import keras.backend as K
from keras import layers, optimizers
from keras.models import Model, model_from_json
from keras.preprocessing import image
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, BatchNormalization

__author__ = 'Rainer Arencibia'
__copyright__ = 'Copyright 2017, Vidrovr'
__credits__ = 'Vidrovr Team'
__license__ = 'MIT'
__version__ = '0.99'
__status__ = 'Prototype'

""" FLOPs:
50 layers  -> 3.8 x 10^9
101 layers -> 7.6 x 10^9
152 layers -> 11.3 x 10^9
"""


def tf_dim_ordering_sort_format(input_shape):

    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS

    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
        shape = (input_shape[1], input_shape[2], input_shape[0])
        return Input(shape=shape)


def block_without_shortcut(input_tensor, kernel_size, filters, stage, block):

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def block_with_shortcut(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=CHANNEL_AXIS, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class Resnet(object):

    def __init__(self):
        pass

    @staticmethod
    def build_resnet(input_shape, num_outputs=1000, include_top=True, pooling=None, weights_path=None):
        """
        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels) TensorFlow Format!!
            num_outputs: The number of outputs at final softmax layer
            include_top: Is include the top fully connected layer.
            pooling: if include_top is false, give a different option of pooling. 'avg', 'max', none by default.
            weights_path: URL to the weights of a pre-trained model.
        Returns:
            A Keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple like (nb_rows, nb_cols, nb_channels)")

        input_shape = tf_dim_ordering_sort_format(input_shape)

        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False, input_shape=input_shape)
        # output image shape = (112x112)
        x = BatchNormalization(axis=CHANNEL_AXIS, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = block_with_shortcut(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = block_without_shortcut(x, 3, [64, 64, 256], stage=2, block='b')
        x = block_without_shortcut(x, 3, [64, 64, 256], stage=2, block='c')
        # output image shape = (56x56)

        x = block_with_shortcut(x, 3, [128, 128, 512], stage=3, block='a')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='b')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='c')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='d')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='e')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='f')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='g')
        x = block_without_shortcut(x, 3, [128, 128, 512], stage=3, block='h')
        # output image shape = (28x28)

        x = block_with_shortcut(x, 3, [256, 256, 1024], stage=4, block='a')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='b')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='c')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='d')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='e')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='f')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='g')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='h')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='i')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='j')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='k')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='l')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='m')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='n')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='o')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='p')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='q')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='r')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='s')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='t')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='u')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='v')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='w')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='x')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='y')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='z')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='a1')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='b2')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='c3')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='d4')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='e5')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='f6')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='g7')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='h8')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='i9')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='j10')
        x = block_without_shortcut(x, 3, [256, 256, 1024], stage=4, block='k11')
        # output image shape = (14x14)

        x = block_with_shortcut(x, 3, [512, 512, 2048], stage=5, block='a')
        x = block_without_shortcut(x, 3, [512, 512, 2048], stage=5, block='b')
        x = block_without_shortcut(x, 3, [512, 512, 2048], stage=5, block='c')
        # output image shape = (7x7)

        x = AveragePooling2D((7, 7), strides=(1, 1), name='avg_pool')(x)
        if include_top:
            x = Flatten()(x)
            x = Dense(units=num_outputs, activation='softmax', name='fc1000')(x)
        else:
            # 4D tensor output pooling == None
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)  # 2D tensor output pooling == None
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)      # Apply Global Max

        if input_shape is not None:
            input_shape = get_source_inputs(input_shape)

        model = Model(input_shape, x, name='ResNet-152')

        if weights_path is not None:
            model.load_weights(weights_path)

        """
        Optimizers... We can keep the best two/one optimizers after testing them.
        """
        # SGD + Momentum and Adagrad are usually the best in most cases: better results.
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        adagrad = optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
        # RMSProp The only different with Adagrad is the way to calculate Gt term = Exp Decay Ave.
        rmsprop = optimizers.RMSprop(lr=0.001, rh0=0.9, epsilon=1e-08, decay=0.0)
        # Good for minimizing the error but not to good accuracy.
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # Never* the best but never* fail.
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        opts = [sgd, adagrad, rmsprop, adam, adadelta]

        models = []
        for opt in opts:
            models.append(model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']))

        return np.asarray(models)  # 5 keras models, each one compile with a different optimizer. Purpose => TEST!

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, weights):
        return Resnet.build_resnet(input_shape, num_outputs, include_top=True, pooling=None, weights_path=weights)


if __name__ == '__main__':

    img_path = './input/b.png'
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    img = preprocess_input(img)

    resnet = Resnet()
    # Return 5 models as Numpy arrays. Every model have a different optimizer.
    models = resnet.build_resnet_152(input_shape=(224, 224, 3), num_outputs=1000, weights=None)
    for model in models:
        preds = model.predict(img)
        print('{}'.format(model.metrics_names[1]), 'Prediction: ', decode_predictions(preds))

    x_train, x_test, y_train, y_test = None, None, None, None  # train_test_split method
    """
    Training...
    """
    for model in models:
        model.fit(x_train, y_train, nb_epoch=100, batch_size=256, verbose=2)

    """ *****
    Saving 2 DISK... BACKUP BACKUP BACKUP: Model Architecture (.JSON) and Weights (.HDF5)
    """
    for i, model in enumerate(models):
        # serialize model to JSON
        model_json = model.to_json()
        with open('./output/model_' + str(i) + '.json', 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights('./output/model_' + str(i) + '.h5')

    """ *****
    Loading to RAM: Model Architecture and Weights.. 0-SGD, 1-Adagrad, 2-RMSProp, 3-Adam, 4-Adadelta
    """
    models = []
    for i in range(5):
        # load json and create model
        json_file = open('./output/model_' + str(i) + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('./output/model_' + str(i) + '.h5')
        models.append(model)

        #  models[i] = resnet.build_resnet_152(input_shape=(224, 224, 3), num_outputs=1000, weights='./output/model_' +
        # str(i) + '.h5')

    """
    Evaluate and Scores of models.
    """
    if __name__ == '__main__':
        for model in models:
            score = model.evaluate(x_test, y_test, verbose=2)
            print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    # Ideas: Offer the Vidrovr application to video makers to use in their videos in Youtube, Vimeo, etc..
    # Feature: Analyze a Soccer or Football game "voice" cut the best moments of the games. Highlights.
    # with a window of -20 second of the signal and +20 more seconds from the signal (replays include)
