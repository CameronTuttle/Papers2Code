# -*- coding: utf-8 -*-
from keras.preprocessing import image as image_utils
from src.imagenet_utils import decode_predictions, preprocess_input
from src.resnet50 import ResNet50
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image needed.")
args = vars(ap.parse_args())

original_img = cv2.imread(args['image'])
image = image_utils.load_img(args['image'], target_size=(224, 224))
image = image_utils.img_to_array(image)

image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

model = ResNet50(weights="imagenet")

pred = model.predict(image)
preds = decode_predictions(pred)

# The first/best prediction: Bigger confidence.
(id, label, confidence) = preds[0][0]
print('Label: ', label, 'Confidence: {0:.2f}%'.format(confidence*100))
