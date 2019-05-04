import numpy as np
from keras.preprocessing import image
from Papers2Code.ZFNet.ZFNet import ZFNet
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ZFNet()

# Do this for all the images
# Your final np array will have shape (N, 224, 224, 3) which is 224x224 pixels x3 colors xN images
img = image.load_img(path='Papers2Code/ZFNet/input/lena.png', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# This just creates an empty model
model.build(input_shape = (img.shape[1], img.shape[2], img.shape[3]))

# model.model.summary() just to sanity check

# don't forget model.compile() try out the different optimizers
# model.model.fit() use your usual x_train, y_train, stuff

# This code works
preds = model.predict(img)

preds

# Everything below you need to handle on your own.
# print('Predicted: ', decode_predictions(preds[0][0]))

# # The first/best prediction: Bigger confidence.
# (id, label, confidence) = preds[0][0]
# print('Label: ', label, 'Confidence: {0:.2f}%'.format(confidence*100))
