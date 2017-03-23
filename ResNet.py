# -*- coding: utf-8 -*-
import numpy as np
import keras.backend as K
from keras import layers, optimizers
from keras.models import Model, model_from_json, Sequential
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, BatchNormalization, ZeroPadding2D, Merge, merge

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


def block_without_shortcut(model, filter, stage, block, index=1):

    union_model = model
    conv_name_base = 'res' + str(stage) + block + str(index) + '_branch'
    bn_name_base = 'bn' + str(stage) + block + str(index) + '_branch'

    model.add(Conv2D(filter, kernel_size=(1, 1), name=conv_name_base + '2a'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2a'))
    model.add(Activation('relu'))

    model.add(Conv2D(filter, kernel_size=(3, 3), padding='same', name=conv_name_base + '2b'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2b'))
    model.add(Activation('relu'))

    model.add(Conv2D(filter*4, kernel_size=(1, 1), name=conv_name_base + '2c'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2c'))

    # ??? Merge two Sequential models in Keras. Try Dense or Flatten
    # addition = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM) Caffe Layers = L
    # merged = merge([model, union_model], mode='sum')
    model_merged = Sequential()
    model_merged.add(Merge([model, union_model], mode='sum'))
    model_merged.add(Activation('relu'))
    return model_merged


def block_with_shortcut(model, filter, stage, block):

    shortcut_model = model
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    model.add(Conv2D(filter, kernel_size=(1, 1), name=conv_name_base + '2a'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2a'))
    model.add(Activation('relu'))

    model.add(Conv2D(filter, kernel_size=(3, 3), padding='same', name=conv_name_base + '2b'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2b'))
    model.add(Activation('relu'))

    model.add(Conv2D(filter*4, kernel_size=(1, 1), name=conv_name_base + '2c'))
    model.add(BatchNormalization(axis=3, name=bn_name_base + '2c'))

    shortcut_model.add(Conv2D(filter*4, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '1'))
    shortcut_model.add(BatchNormalization(axis=3, name=bn_name_base + '1'))

    # ??? Merge two Sequential models in Keras. Try Dense or Flatten
    # addition = Caffe Layers.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    # merged = Merge([model, shortcut_model], mode='sum')
    # merged = merge([model, shortcut_model], mode='sum')
    # from keras.layers.merge import add, concatenate
    model_merged = Sequential()
    model_merged.add(Merge([model, shortcut_model], mode='sum'))
    model_merged.add(Activation('relu'))
    return model_merged


class Resnet(object):

    def __init__(self):
        pass

    @staticmethod
    def build_resnet(input_shape=None, num_outputs=1000, layers=None, weights_path=None):
        """
        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels) TensorFlow Format!!
            num_outputs: The number of outputs at final softmax layer
            layers: Number of layers for every network 50, 101, 152
            weights_path: URL to the weights of a pre-trained model.
        Returns:
            A Keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple like (nb_rows, nb_cols, nb_channels)")

        from keras.applications.imagenet_utils import _obtain_input_shape
        input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=197,
                                          data_format=K.image_data_format(), include_top=True)
        # img_input = Input(shape=input_shape)

        model = Sequential()
        model.add(ZeroPadding2D((3, 3), input_shape=input_shape))
        model.add(Conv2D(64, (7, 7), strides=(2, 2), name='conv1'))
        model.add(BatchNormalization(axis=3, name='bn_conv1'))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1'))

        nb_filters = 64
        stage = 2
        # Check the merge/concatenate layers... 107 layers against 7 layers created.
        for e in layers:  # layers = [3, 8, 36, 3]
            for i in range(e):  # 3
                if i == 0:
                    model = block_with_shortcut(model, nb_filters, stage=stage, block='a')
                else:
                    model = block_without_shortcut(model, nb_filters, stage=stage, block='b', index=i)

            stage += 1
            nb_filters *= 2

        model.add(AveragePooling2D((7, 7), strides=(1, 1), name='avg_pool'))
        # fc = L.InnerProduct(glb_pool, num_output=1000)
        model.add(Flatten())
        model.add(Dense(num_outputs, name='fc1000'))
        model.add(Activation('softmax'))

        if weights_path is not None:
            model.load_weights(weights_path)

        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, weights):
        return Resnet.build_resnet(input_shape, num_outputs, layers=[3, 4, 6, 3], weights_path=weights)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, weights):
        return Resnet.build_resnet(input_shape, num_outputs, layers=[3, 4, 23, 3], weights_path=weights)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, weights):
        return Resnet.build_resnet(input_shape, num_outputs, layers=[3, 8, 36, 3], weights_path=weights)


if __name__ == '__main__':

    import cv2
    img = cv2.imread('./input/b.png')
    img = cv2.resize(img, dsize=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(np.asarray(img, dtype='float64'))  # shape (1, 224, 224, 3) type float64 ; BGR format by cv2

    resnet = Resnet()
    model = resnet.build_resnet_50(input_shape=img[0].shape, num_outputs=1000,
                                   weights=None)  # './weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    # Show summary of model
    print("[INFO] Summary of the model...")
    model.summary()
    """
    config = model.get_config()
    print("[INFO] Config: ")
    for i, conf in enumerate(config):
        print(i, conf.__str__())

    import time
    start = time.time()
    print('Start Backup: ')
    # serialize model to JSON
    model_json = model.to_json()
    with open('./output/model_design' + '.json', 'w') as json_file:
        json_file.write(model_json)
    stop = time.time() - start
    print('Time to Backup: ', stop)
    """
    # serialize weights to HDF5
    # model.save_weights('./output/model_weights' + '.h5', overwrite=True)

    # preds = model.predict(img)
    # print('{}'.format(model.metrics_names[1]), 'Prediction: ', decode_predictions(preds))

    """
    Training...
    x_train, x_test, y_train, y_test = None, None, None, None  # train_test_split method
    # for model in models:
    model.fit(x_train, y_train, nb_epoch=1, batch_size=256, verbose=2)


    # Saving 2 DISK... BACKUP BACKUP BACKUP: Model Architecture (.JSON) and Weights (.HDF5)
    for i, model in enumerate(models):
        # serialize model to JSON
        model_json = model.to_json()
        with open('./output/model_' + str(i) + '.json', 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights('./output/model_' + str(i) + '.h5')


    # Loading to RAM: Model Architecture and Weights.. 0-SGD, 1-Adagrad, 2-RMSProp, 3-Adam, 4-Adadelta
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

    # Evaluate and Scores of models.
    if __name__ == '__main__':
        for model in models:
            score = model.evaluate(x_test, y_test, verbose=2)
            print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    # Ideas: Offer the Vidrovr application to video makers to use in their videos in Youtube, Vimeo, etc..
    # Feature: Analyze a Soccer or Football game "voice" cut the best moments of the games. Highlights.
    # with a window of -20 second of the signal and +20 more seconds from the signal (replays include)
    """
