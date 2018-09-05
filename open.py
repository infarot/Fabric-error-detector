# -*- coding: utf-8 -*-
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import os
from PIL import Image

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 184) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='sgd',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="fabric-classifier.tfl")
model.load("fabric-classifier.tfl")


def predict(im):
    while True:
        img = scipy.misc.imresize(im, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
        # Predict
        prediction = model.predict([img])
        # Check the result.
        is_defect = np.argmax(prediction[0]) == 0
        if is_defect:
            typ = True
            print("Defect found")
        else:
            typ = False
            print("Defect not found")
        return typ


#for image_path in TEST_IMAGE_PATHS:
#    image = Image.open(image_path)
#    predict(image)

