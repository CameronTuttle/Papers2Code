# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing import image
from ImageNet import ImageNet
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ImageNet()
img = image.load_img(path='./data/me.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

preds = model.predict(img)
print('Predicted: ', decode_predictions(preds))

# The first/best prediction: Bigger confidence.
(id, label, confidence) = preds[0][0]
print('Label: ', label, 'Confidence: {0:.2f}%'.format(confidence*100))
